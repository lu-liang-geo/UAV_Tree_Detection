# Adapted from SAM's MaskDecoder and DETR
# https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py
# https://github.com/facebookresearch/detr/blob/main/models/detr.py

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
        num_boxes: int = 200,
        num_classes: int = 1,
        box_head_depth: int = 3,
        box_head_hidden_dim: int = 256,
        class_head_depth: int = 3,
        class_head_hidden_dim: int = 256,
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
          class_head_depth (int): the depth of the MLP used to predict
            box class
          class_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict box class
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_box_tokens = num_boxes
        self.num_classes = num_classes + 1  # num_classes + 1 for no-object
        self.box_tokens = nn.Embedding(self.num_box_tokens, transformer_dim)

        self.reduce_image_embedding = CNN(
            in_channels=transformer_dim * 2, out_channels=transformer_dim)  
        self.box_prediction_head = MLP(
            transformer_dim, box_head_hidden_dim, 4, box_head_depth, sigmoid_output=True)
        self.class_prediction_head = MLP(
            transformer_dim, class_head_hidden_dim, self.num_classes, class_head_depth)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor = None,
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
        # TODO: either implement a way to read dense prompt embeddings or remove from BoxDecoder
        if dense_prompt_embeddings is not None:
          raise ValueError('BoxDecoder does not currently support dense prompt embeddings from masks.')

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

        # Reshape sparse prompt embeddings so that multiple box prompts are concatenated in
        # the 1st dimension rather than the 0th
        sparse_prompt_embeddings = sparse_prompt_embeddings.reshape(1,-1,256)

        # Concatenate output tokens
        tokens = torch.cat((self.box_tokens.weight[None], sparse_prompt_embeddings), dim=1)

        # Run the transformer, extract transformed box tokens
        hs, _ = self.transformer(image_embeddings, image_pe, tokens)
        box_tokens_out = hs[:, :self.num_box_tokens, :]

        # Predict bounding boxes and classes from transformed box tokens
        pred_boxes = self.box_prediction_head(box_tokens_out)
        pred_logits = self.class_prediction_head(box_tokens_out)

        # Return outputs in dictionary
        outputs = {'pred_boxes' : pred_boxes, 'pred_logits' : pred_logits}

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