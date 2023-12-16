# Adapted from SAM's MaskDecoder and DETR
# https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py
# https://github.com/facebookresearch/detr/blob/main/models/detr.py

'''
NOTE: This is an older, incomplete implementation of BoxDecoder; the file box_decoder.py contains the
current, complete implementation. I am saving this implementation as well because it is closer to the
MaskDecoder, and if in the future we decide we want to predict a set number of boxes per-prompt (as 
MaskDecoder does) rather than a set number of boxes per-image (as the current BoxDecoder does), it will 
probably be easier to work back from this implementation than the current one in box_decoder.py.
'''

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
from ..utils import box_ops


class BoxDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        transformer_dim: int,
        num_boxes: int = 100,
        box_head_depth: int = 3,
        box_head_hidden_dim: int = 256,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts boxes given an image and prompt embeddings, using a
        transformer architecture. May take a larger box and split it
        into multiple smaller boxes.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict boxes
          num_boxes (int): the number of boxes to predict within a region
            of interest
          iou_head_depth (int): the depth of the MLP used to predict
            box quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict box quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_boxes = num_boxes

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_box_tokens = num_boxes
        self.box_tokens = nn.Embedding(self.num_box_tokens, transformer_dim)

        self.reduce_image_embedding = CNN(
            in_channels=transformer_dim * 2, out_channels=transformer_dim)  
        self.box_prediction_head = MLP(
            transformer_dim, box_head_hidden_dim, 4, box_head_depth, sigmoid_output=True)
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_box_tokens, iou_head_depth)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict boxes given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted boxes
          torch.Tensor: batched predictions of box quality
        """
        outputs = self.predict_boxes(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Prepare output
        return outputs

    def predict_boxes(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts boxes. See 'forward' for more details."""
        # Reduce image embedding channels if two images were embedded together (i.e. RGB and Multi images)
        if image_embeddings.shape[1] != self.transformer_dim:
            image_embeddings = self.reduce_image_embedding(image_embeddings)

        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.box_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-box
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        box_tokens_out = hs[:, 1 : (1 + self.num_box_tokens), :]

        # Create output dictionary
        outputs = dict()

        # Predict boxes using the box tokens
        outputs['pred_boxes'] = self.box_prediction_head(box_tokens_out)

        # Generate box quality predictions
        outputs['pred_logits'] = self.iou_prediction_head(iou_token_out)

        return outputs


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class CNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.conv2 = nn.Conv2d(in_channels * 2, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x

# Taken from DETR
# https://github.com/facebookresearch/detr/blob/main/models/detr.py#L37
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results