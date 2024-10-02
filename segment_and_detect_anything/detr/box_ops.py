# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code copied from DETR.
# Source code at: https://github.com/facebookresearch/detr/blob/main/util/box_ops.py

# Additional functions (c) 2023 William Locke, marked below:
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import numpy as np
import supervision as sv
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


# New functions added by William Locke below
def box_area(box):
    t_box = box.T
    return (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])


def box_overlap(
	boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:

    area_a = box_area(boxes_a)
    area_b = box_area(boxes_b)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
    	np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return area_inter / area_a[:, None]


def external_box_suppression(
   detections,
   inter_threshold = 0.75,
   ignore_categories = False):
    '''
    Unlike Non-Max Suppression, this function orders boxes by size (smallest to largest)
    and eliminates larger boxes that contain smaller boxes over the inter_threshold, where
    inter_threshold signifies the percent of the smaller box's total area that is contained
    in the larger box. So if a smaller box has more than e.g. 75% of its area encompassed by
    a larger box, that larger box is suppressed.

    params:
    detections:   Supervision Detections object containing Nx6 array, where N is the number of
                  predicted boxes, the first 4 columns are the xy coordinates of the top-left
                  and bottom-right corners respectively of each box, the 5th column is the
                  confidence, and the 6th column is the box label.

    inter_threshold:  The percent of a given box's total area that is contained within another box
                      to activate external_box_suppression.

    ignore_categories:  If False, external_box_suppression will only be applied to boxes with the
                        same label, otherwise will be applied between all boxes regardless of label.

    returns:
    keep: Index of the detections to keep after non-max suppression.
    '''

    boxes = detections.xyxy
    confidence = detections.confidence
    categories = detections.class_id
    areas = box_area(boxes)

    rows = boxes.shape[0]

    sort_index = areas.argsort()
    boxes = boxes[sort_index]
    confidence = confidence[sort_index]
    categories = categories[sort_index]

    overlaps = box_overlap(boxes, boxes)
    overlaps = overlaps - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(overlaps, categories)):
        if not keep[index]:
            continue

        if ignore_categories:
            condition = (iou > inter_threshold)
        else:
            condition = (iou > inter_threshold) & (categories == category)
        keep = keep & ~condition

    return keep


def box_overlap_max(
	boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:

    area_a = box_area(boxes_a)
    area_b = box_area(boxes_b)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
    	np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    area_inter_norm1 = area_inter / area_a[:, None]
    area_inter_norm2 = area_inter / area_a[None, :]
    area_inter_norm = np.maximum(area_inter_norm1, area_inter_norm2)

    return area_inter_norm


def custom_nms(
   detections,
   inter_threshold = 0.75,
   ignore_categories = False):
    '''
    Like regular Non-Max Suppression, but uses the inter_threshold of External Box Suppression
    rather than an IoU Threshold like standard NMS.

    params:
    detections:   Supervision Detections object containing Nx6 array, where N is the number of
                  predicted boxes, the first 4 columns are the xy coordinates of the top-left
                  and bottom-right corners respectively of each box, the 5th column is the
                  confidence, and the 6th column is the box label.

    inter_threshold:  The percent of a given box's total area that is contained within another box
                      to activate custom non-max suppression.

    ignore_categories:  If False, external_box_suppression will only be applied to boxes with the
                        same label, otherwise will be applied between all boxes regardless of label.

    returns:
    keep: Index of the detections to keep after non-max suppression.
    '''
    boxes = detections.xyxy
    confidence = detections.confidence
    categories = detections.class_id

    rows = boxes.shape[0]

    sort_index = np.flip(confidence.argsort())
    boxes = boxes[sort_index]
    confidence = confidence[sort_index]
    categories = categories[sort_index]

    overlaps = box_overlap_max(boxes, boxes)
    overlaps = overlaps - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (inter, category) in enumerate(zip(overlaps, categories)):
        if not keep[index]:
            continue

        if ignore_categories:
            condition = (inter > inter_threshold)
        else:
            condition = (inter > inter_threshold) & (categories == category)
        keep = keep & ~condition

    return keep