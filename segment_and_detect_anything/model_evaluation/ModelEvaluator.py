'''
File: ModelEvaluator.py
Author: Ishir Garg (ishirgarg@berkeley.edu)
Date: 3/18/24

Class for evaluating any PyTorch model on the NEON dataset
'''

from typing import List
from supervision import Detections
import torch
import cv2
import numpy as np
from RGBNeonDataset import RGBNEONTreeDataset
import cv_utils
from cv_utils import TreeDetections
from metrics_utils import compute_metrics
import abc

class ModelEvaluator:
    def __init__(self, image_path, annotation_path):
        '''
        Args:
            model : A PyTorch model to be evaluated
            image_path : File path to RGB images
            annotation_path : File path to annotations
            input_bgr_format: If true, indicates that the model takes in BGR images, else, it is assumed that the model takes in RGB images

        '''
        self.device = ModelEvaluator._load_device()
        self.dataset = ModelEvaluator._load_dataset(image_path, annotation_path)
        self.model = self.load_model()

        # Load data for future evaluation
        self.rgb_images = [data["rgb"] for data in self.dataset]
        self.annotations = [data["annotation"] for data in self.dataset]

    def dataset_len(self):
        '''Returns length of the dataset used for evaluation'''
        return len(self.dataset)

    @abc.abstractmethod
    def load_model(self):
        '''Returns a PyTorch model with loaded weights'''
        raise "Abstract method should be implemented in child class"

    def _load_device():
        '''Returns a GPU PyTorch device if one exists, else returns the CPU device'''
        return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    def _load_dataset(image_path : str, annotation_path : str):
        '''Returns the NEON dataset
        
        Args:
            image_path : File path to RGB images
            annotation_path : File path to annotations
        '''
        return RGBNEONTreeDataset(image_path, annotation_path, check_values=True)

    def _filter_bboxes(bboxes, scores, confidence_threshold):
        '''Filter out bboxes that have low confidence

        Args:
            bboxes : numpy array of bounding boxes, has shape (N, 4) where N is number of bboxes
            scores : numpy array of confidence scores, has shape (N, 1) where N is number of boxes
            confidence_threshold : Confidence threshold to filter out bboxes
        '''

        assert(bboxes.shape[1] == 4), "Bboxes should have be in xyxy format"
        assert(bboxes.shape[0] == scores.shape[0]), "Number of bboxes should equal number of scores"

        keep_indices = np.nonzero(scores > confidence_threshold)
        return bboxes[keep_indices], scores[keep_indices]
    
    @abc.abstractmethod
    def predict_image(self, model, rgb_image) -> dict:
        '''Returns the output of the model given an image in unnormalized RGB format; some recommended considerations to make when implementing this function are
        - Does the model want BGR or RGB images (rgb_image is in RGB format)
        - Does the model want (C, W, H) or (W, H, C) format (rgb_image is in (W, H, C) format)
        - Does the model perform better/require normalized RGB inputs

        Args:
            rgb_image : The image to be passed into the model; is in RGB format

        Returns:
            A dict with the following key-value pairs:
                "bboxes" : bboxes in xyxy format, numpy array of size (N, 4) where N is the number of boxes
                "scores" : confidence scores for each bounding box
        '''
        raise "Abstract method should be implemented in child class"
    
    def plot_image_annotations(self, image_index : int, size : tuple):
        '''Plots an image with predicted bboxes and annotations
        
        Args:
            image_index : The index of the image to plot in our rgb image array
            size : Size of the plot, must be tuple of length 2 characterizing the length and width of the plot
        '''
        assert hasattr(self, 'detections_bboxes'), "Must call self.evaluate_model() at least once before plotting results"
        assert (image_index >= 0 and image_index < len(self.dataset)), f"image_index {image_index} out of bounds for dataset of size {len(self.dataset)}"
        assert(len(size) == 2), "Size must be 2-element tuple consisting of length and width"

        image = self.rgb_images[image_index]
        annotations = self.annotations[image_index]
        detections = self.detections_bboxes[image_index]

        cv_utils.plot_img_with_annotations(image, detections, size)
        cv_utils.plot_img_with_annotations(image, TreeDetections(annotations), size)

    def _postprocess_bboxes(self, bboxes, scores, confidence_threshold, nms_threshold, nms_type):
        if len(bboxes) > 0:
            # Filter low confidence boxes
            bboxes, scores = ModelEvaluator._filter_bboxes(bboxes, scores, confidence_threshold)
            # Apply NMS
            if nms_type == "iomin":
                nms_func = cv_utils.compute_nms_iomin
            elif nms_type == "iou":
                nms_func = cv_utils.compute_nms_iou
            else:
                raise "nms_type must be either iomin or iou"
            
            bboxes, scores = nms_func(bboxes, scores, nms_threshold)
        return bboxes, scores
    
    def eval_and_plot_image_annotations(self, rgb_image, annotations: np.ndarray, confidence_threshold: float, nms_threshold: float, size : tuple, nms_type: str):
        '''Evaluates an image to generate predicted bboxes, the plots image with annotations
        
        Args:
            rgb_image : Image in RGB format
            annotations : Array of annotations, has shape (N, 4) where N is number of annotated bboxes for the image
            confidence_threshold: Confidence threshold for bbox filtering
            nms_threshold: Non-maximal supression threshold for bbox filtering
            size : Size of the plot, must be tuple of length 2 characterizing the length and width of the plot
            nms_type: Specify either "iou" or "iomin" to choose Intersection-over-union or Intersection-over-minimum-area for Non-maximal suppression
        '''
        detection = self.predict_image(self.model, rgb_image)
        bboxes, scores = self._postprocess_bboxes(detection["bboxes"], detection["scores"], confidence_threshold, nms_threshold, nms_type)

        cv_utils.plot_img_with_annotations(rgb_image, TreeDetections(xyxy=bboxes, confidence=scores), size)
        cv_utils.plot_img_with_annotations(rgb_image, TreeDetections(annotations), size)
        
    def evaluate_model(self, confidence_threshold: float, iou_threshold: float, nms_threshold: float, nms_type: str) -> tuple[List[TreeDetections], dict]:
        '''Returns a dictionary containing bounding boxes, confidence scores, and metrics
        
        Args:
            confidence_threshold : Confidence threshold to filter out low-confidence bboxes
            iou_threshold: IoU threshold to count matched bboxes as a true positive
            nms_threshold: Non-maximal supression threshold for bbox filtering
            nms_type: Specify either "iou" or "iomin" to choose Intersection-over-union or Intersection-over-minimum-area for Non-maximal suppression

        Returns:
            A list of detections, one Detections object for each image
            A dictionary containing 3 key-value pairs:
                "bboxes" : Array of bounding boxes for each image
                "scores" : Array of confidence scores for each image
                "metrics": A dictionary containing 4 key-value pairs:
                    "map" : the computed mean average precision
                    "precision" : the computed precision
                    "recall" : the computed recall
                    "f1" : the computed f1-score
        '''
        
        # Input data into model
        with torch.no_grad():
            detections = [self.predict_image(self.model, img) for img in self.rgb_images]

        detections_bboxes = [detection["bboxes"] for detection in detections]
        detections_scores = [detection["scores"] for detection in detections]

        # Filter out bboxes with low confidence
        for i in range(len(detections_bboxes)):
            assert len(detections_bboxes[i]) == len(detections_scores[i]), "Number of detected bboxes and scores should be the same"

            detections_bboxes[i], detections_scores[i] = self._postprocess_bboxes(detections_bboxes[i], detections_scores[i], confidence_threshold, nms_threshold, nms_type)
        
        # Format detections into Detections objects
        formatted_detections = []
        for i in range(len(detections_bboxes)):
            # Format into TreeDetections object
            fmt_detection = TreeDetections(xyxy=np.array(detections_bboxes[i]), 
                                    confidence=np.array(detections_scores[i]))
            formatted_detections.append(fmt_detection)
        
        # Compute metrics
        metrics = compute_metrics(formatted_detections, self.annotations, iou_threshold)

        # Cache results for plotting
        self.detections_bboxes = formatted_detections
        self.detections_scores = detections_scores
        self.metrics = metrics

        return {
            "bboxes" : detections_bboxes, 
            "scores" : detections_scores, 
            "metrics" : metrics
        }