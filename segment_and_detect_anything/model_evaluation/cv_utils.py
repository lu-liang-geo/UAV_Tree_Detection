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

class TreeDetections:
    def __init__(self, xyxy: np.ndarray, confidence: np.ndarray = None):
        '''
        Args:
            xyxy: A numpy array of bboxes with shape (N, 4), where N is number of bboxes
            confidence: A numpy array of shape (N,) where N is confidence scores for boxes
        '''
        assert xyxy.shape[1] == 4, f"xyxy should have shape (N, 4), instead has shape {xyxy.shape}"
        assert (confidence is None) or (xyxy.shape[0] == confidence.shape[0]), "xyxy and confidence should have same length"

        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = np.array([0] * len(xyxy)) # All class ids are 0s, indicating a tree

    def to_svn_detections(self):
        '''Returns this DetectionSet object as a supervision.core.Detections object'''
        # If we have no detections, return a "null" detection for purposes of plotting
        if (len(self.xyxy) == 0) or (self.xyxy is None):
            return Detections(
                xyxy=np.array([[0, 0, 0, 0]]),
                confidence=np.array(0),
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
