'''
File: metrics_utils.py
Author: Ishir Garg (ishirgarg@berkeley.edu)
Date: 3/19/24

Generic functions for computing various computer vision-related metrics
'''
from torchmetrics.detection import MeanAveragePrecision
import scipy
from typing import List
import torch
import numpy as np
from cv_utils import TreeDetections, bbox_iou

def compute_map(detections : List[TreeDetections], annotations : list):
    '''Computes the mean average precision of a set of detections
    
    Args:
        detections : List of detections for each image
        annotations : Array containing annotations for each image

    Returns:
        The computed mean average precision
    '''
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        iou_thresholds=None, # Uses default [0.05, 0.1, ..., 1]
        rec_thresholds=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], # Uses default [0, 0.01, 0.02, ..., 1]
        max_detection_thresholds=[100, 1000, 10000],
        class_metrics=False,
        extended_summary=False,
        average="macro",
        backend="pycocotools")
    
    predictions = [{
        "boxes" : torch.Tensor(detections[i].xyxy),
        "scores" : torch.Tensor(detections[i].confidence),
        "labels" : torch.Tensor(detections[i].class_id).int()
    } for i in range(len(detections))]

    targets = [{
        "boxes" : torch.Tensor(annotations[i]),
        "labels" : torch.zeros(len(annotations[i])).int()
    } for i in range(len(annotations))]
    
    metric.update(predictions, targets)

    return metric.compute()["map_50"]

def bbox_iou(box1: np.ndarray, box2: np.ndarray):
    x_diff = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_diff = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection = x_diff * y_diff

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection

    return intersection / union_area if union_area > 0 else 0

def hungarian_matching(detections : np.ndarray, annotations : np.ndarray):
    '''Computes the Hungarian matching for a set of detections and annotations
    
    Args:
        detections : Array of detection bboxes for an image, has shape (N, 4) where N is number of bboxes
        annotations : Array containing annotations for each image, has shape (N,)
        iou_threshold : Iou Threshold to use for determining true positives

    Returns:
        An array of two-element tuples representing the matchings; the first index is the bbox in the detections array, and the second index is the bbox in the annotations array
    '''

    assert detections.shape[1] == 4
    assert annotations.shape[1] == 4

    cost_matrix = np.zeros((len(detections), len(annotations)))
    for i in range(len(detections)):
        for j in range(len(annotations)):
            cost_matrix[i, j] = bbox_iou(detections[i], annotations[j])

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
    return zip(row_ind, col_ind)

def compute_precision_recall(detections : List[np.ndarray], annotations : np.ndarray, iou_threshold):
    '''Computes the precision and recall of a set of detections using Hungarian matching and IoU
    
    Args:
        detections : List of detections for each image
        annotations : Array containing annotations for each image
        iou_threshold : Iou Threshold to use for determining true positives

    Returns:
        precision, recall : The computed precision and recall
    '''
    assert len(detections) == len(annotations), "Number of detections should equal number of annotations (both are equal to number of images)"

    precisions = []
    recalls = []

    for i in range(len(detections)):
        # Precision recall edge cases
        # Zero boxes detected, zero boxes in annotation
        if len(detections[i]) == 0 and len(annotations[i]) == 0:
            precisions.append(1)
            recalls.append(1)
            continue
        if len(detections[i]) == 0 or len(annotations[i]) == 0:
            print(f"{len(detections[i])} detections and {len(annotations[i])} annotations for this image, not included in metric computation")
            continue

        matchings = hungarian_matching(detections[i], annotations[i])
        iou_scores = [bbox_iou(detections[i][pair[0]], annotations[i][pair[1]]) for pair in matchings]
        iou_scores = np.array(iou_scores) > iou_threshold

        precisions.append(sum(iou_scores) / len(detections[i]))
        recalls.append(sum(iou_scores) / len(annotations[i]))

    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def compute_metrics(detections : List[TreeDetections], annotations : np.ndarray, iou_threshold) -> dict:
    '''Computes the mean average precision, precision, recall, and f1 score of a set of detections
    
    Args:
        detections : List of detections for each image
        annotations : Array containing annotations for each image
        iou_threshold : Iou Threshold to use for determining true positives

    Returns:
        dictionary containing 4 key-value pairs:
            "map" : the mean average precision
            "precision" : the precision
            "recall" : the recall
            "f1" : the f1 score
    '''

    map = compute_map(detections, annotations)
    precision, recall = compute_precision_recall([detection.xyxy for detection in detections], annotations, iou_threshold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall != 0) else 0

    return {
        "map" : map,
        "precision" : precision,
        "recall" : recall,
        "f1" : f1
    }