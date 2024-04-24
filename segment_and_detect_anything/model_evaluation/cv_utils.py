'''
File: cv_utils.py
Author: Ishir Garg (ishirgarg@berkeley.edu)
Date: 3/18/24

Generic functions for various image processing and annotation tasks
'''

import supervision as svn
from supervision import Detections
from typing import List
import torch
import numpy as np
import torchvision

class TreeDetections:
    def __init__(self, xyxy: np.ndarray, confidence: np.ndarray = None):
        '''
        Args:
            xyxy: A numpy array of bboxes with shape (N, 4), where N is number of bboxes
            confidence: A numpy array of shape (N,) where N is confidence scores for boxes
        '''
        assert (len(xyxy) == 0) or xyxy.shape[1] == 4, f"xyxy should have shape (N, 4), instead has shape {xyxy.shape}"
        assert (confidence is None) or (xyxy.shape[0] == confidence.shape[0]), "xyxy and confidence should have same length"

        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = np.array([0] * len(xyxy)) # All class ids are 0s, indicating a tree

    def to_svn_detections(self):
        '''Returns this DetectionSet object as a supervision.core.Detections object'''
        # If we have no detections, return a "null" detection for purposes of plotting
        if (len(self.xyxy) == 0):
            return Detections(
                xyxy=np.array([[0, 0, 0, 0]]),
                confidence=np.array([0]),
                class_id=np.array([0])
            )
        
        return Detections(
            xyxy=self.xyxy, 
            confidence=self.confidence, 
            class_id=self.class_id
        )

def plot_img_with_annotations(image, detections : TreeDetections, size : tuple):
    '''Plots image with annotations
    
    Args:
        image : A numpy array in RGB format representing the image
        detections : The detections for the image
        size: Size of the plotted image; tuple containing two integers, x and y dimensions
    '''
    box_annotator = svn.BoxAnnotator()
    
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections.to_svn_detections(), labels=None, skip_label=True)
    svn.plot_image(annotated_frame, size)

def box_cxcywh_to_xyxy(x):
    '''Converts a list of bounding boxes from cxcywh to xyxy format
    cxcywh stores the center x and y coordinates, and the width and height of the bbox
    xyxy stores the top left corner x, y coordinates and the bottom right corner x, y coordinates
    
    Args:
        x : The list of bounding boxes box in cxcywh format, must have shape (N, 4) where N is number of bboxes
    '''
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def compute_nms_iou(bboxes: np.ndarray, scores: np.ndarray, nms_threshold: float):
    '''Performs non-maximal suppression on bounding boxes
    
    Args:
        bboxes: List of bboxes to perform NMS on, has shape (N, 4) where N is initial number of boxes
        scores: Confidence scores of bboxes
        nms_threshold: Threshold of IoU to perform suppression at
    Returns:
        List of bboxes of shape (N, 4) after NMS, where N is final number of boxes
        List of scores of shape (N, 4) after NMS
    '''
    if (len(bboxes) == 0):
        return bboxes.copy(), scores.copy()
    
    assert bboxes.shape[1] == 4, "Array of bounding boxes must have shape (N, 4)"
    assert scores.shape[0] == bboxes.shape[0], "Number of confidence scores must equal number of boxes"

    keep_ind = torchvision.ops.nms(torch.Tensor(bboxes), torch.Tensor(scores), nms_threshold).numpy()
    return bboxes[keep_ind].copy(), scores[keep_ind].copy()


def compute_nms_iomin(bboxes: np.ndarray, scores: np.ndarray, nms_threshold: float):
    '''Performs non-maximal suppression on bounding boxes
    
    Args:
        bboxes: List of bboxes to perform NMS on, has shape (N, 4) where N is initial number of boxes
        scores: Confidence scores of bboxes
        nms_threshold: Threshold of IoU to perform suppression at
    Returns:
        List of bboxes of shape (N, 4) after NMS, where N is final number of boxes
        List of scores of shape (N, 4) after NMS
    '''
    if (len(bboxes) == 0):
        return bboxes.copy(), scores.copy()
    
    assert bboxes.shape[1] == 4, "Array of bounding boxes must have shape (N, 4)"
    assert scores.shape[0] == bboxes.shape[0], "Number of confidence scores must equal number of boxes"

    bboxes = bboxes.copy()
    scores = scores.copy()


    sort_index = np.flip(scores.argsort())
    bboxes = bboxes[sort_index]
    scores = scores[sort_index]

    # Compute mutual iomins
    overlaps = np.empty((len(bboxes), len(bboxes)))
    for i in range(len(bboxes)):
        for j in range(len(bboxes)):
            overlaps[i][j] = bbox_iomin(bboxes[i], bboxes[j])
    overlaps -= np.eye(len(overlaps))

    keep = np.ones(len(bboxes), dtype=bool)

    # Filter out unwanted boxes
    for index, inter in enumerate(overlaps):
        if not keep[index]:
            continue

        condition = (inter > nms_threshold)
        keep = keep & ~condition

    return bboxes[keep], scores[keep]
    
def bbox_iou(box1: np.ndarray, box2: np.ndarray):
    '''Computes Intersection over Union for two bounding boxes
    
    Args:
        box1: Bounding box in xyxy format
        box2: Bounding box in xyxy format

    Returns:
        IoU score
    '''
    x_diff = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_diff = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection = x_diff * y_diff

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection

    return intersection / union_area if union_area > 0 else 0

def bbox_iomin(box1: np.ndarray, box2: np.ndarray):
    '''Computes Intersection over Minimum Box Area for two bounding boxes
    
    Args:
        box1: Bounding box in xyxy format
        box2: Bounding box in xyxy format

    Returns:
        IoMin score
    '''
    x_diff = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_diff = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection = x_diff * y_diff

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    min_area = min(area1, area2)

    return intersection / min_area if min_area > 0 else 0