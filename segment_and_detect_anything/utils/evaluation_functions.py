import os
import rasterio
import numpy as np
import geopandas as gpd
from supervision.detection.utils import box_iou_batch, polygon_to_mask
from scipy.optimize import linear_sum_assignment


def get_masks(filename, rgb_folder, shape_folder, img_size=(400,400)):
  '''
  Transform shapefiles into masks.

  params:
    filename (str): Name of the image, minus extension
    rgb_folder (str): Path to saved RGB tif image.
    shape_folder (str): Path to saved shapefile.
    img_size (tuple): RGB image size in pixels (H x W)

  returns:
    masks (ndarray): Boolean mask with TRUE indicating the polygon area
                     and FALSE elsewhere
  '''
  polygons = []
  with rasterio.open(os.path.join(rgb_folder, filename+'.tif')) as rast_img:
    # Open corresponding shapefile, convert to image's CRS, clip to image bounds
    shapes = gpd.read_file(os.path.join(shape_folder, filename, filename+'.shp'))
    shapes = shapes.to_crs(rast_img.crs)
    shapes = gpd.clip(shapes, rast_img.bounds)
    # Convert shape coordinates to pixel indices, eliminate neighbors that share the same indices
    for i in range(len(shapes)):
      shape = shapes.loc[i, 'geometry']
      coords = list(shape.exterior.coords)
      pixels = [rast_img.index(x,y) for x,y in coords]
      pixels = [pixels[0]] + [pixels[i] for i in range(1,len(pixels)) if pixels[i] != pixels[i-1]]
      pixels = [(p[1],p[0]) for p in pixels]
      polygons.append(np.array(pixels, dtype='int'))
  true_masks = np.array([polygon_to_mask(poly, (img_size[1],img_size[0])) for poly in polygons], dtype=bool)

  return true_masks


def mask_iou_batch(masks_true, masks_pred):
    '''
    Calculate IoU between all pairs of two batches of masks, return matrix of IoUs.
    '''
    intersection = np.logical_and(masks_true[:,None,...], masks_pred).sum(axis=(2,3))
    union = np.logical_or(masks_true[:,None,...], masks_pred).sum(axis=(2,3))
    iou_matrix = np.divide(intersection, union, out=np.zeros(union.shape), where=union!=0)
    return iou_matrix


def boxes_to_masks(boxes, shape):
    '''
    Convert bounding boxes to masks for purposes of computing IoU between boxes and masks.
    '''
    masks = []
    for i, box in enumerate(boxes):
        masks.append(np.full(shape, False))
        masks[i][round(box[1]):round(box[3])+1, round(box[0]):round(box[2])+1] = True
    if len(masks) > 0:
        return np.stack(masks)
    else:
        return np.empty(shape)[np.newaxis]

def hungarian_matching(truths, preds, threshold=0.0):
    '''
    Compute Hungarian Matching between ground truth masks or bounding boxes
    and predicted masks or bounding boxes.

    parameters:
        truths (array): Ground truth bounding boxes or masks
        preds (array): Predicted bounding boxes or masks
        threshold (float): Only return indexes with an IoU greater than the given threshold

    returns:
        true_idx, pred_idx (floats): corresponding indices for the truths and preds with greatest IoU
    '''
    # If both preds and truths are bounding boxes, calculate IoU between boxes and feed into hungarian algorithm
    if truths.ndim == 2 and preds.ndim == 2:
        iou_matrix = box_iou_batch(truths, preds)
        true_idx, pred_idx = linear_sum_assignment(iou_matrix, maximize=True)
        idx = np.array([i for i in range(len(true_idx)) if iou_matrix[true_idx[i], pred_idx[i]] >= threshold], dtype='int')
        return true_idx[idx], pred_idx[idx]

    # If only one is bounding boxes, convert to mask to compute IoU
    elif truths.ndim == 2:
        truths = boxes_to_masks(truths, preds.shape[1:])

    elif preds.ndim == 2:
        preds = boxes_to_masks(preds, truths.shape[1:])

    # Calculate IoU between masks, feed into hungarian algorithm
    iou_matrix = mask_iou_batch(truths, preds)
    true_idx, pred_idx = linear_sum_assignment(iou_matrix, maximize=True)
    idx = np.array([i for i in range(len(true_idx)) if iou_matrix[true_idx[i], pred_idx[i]] >= threshold], dtype='int')
    return true_idx[idx], pred_idx[idx]


def compute_iou(truths, preds, reduction='none'):
    '''
    Calculate IoU between matched masks, return all or average IoU.

    params:
        truths (ndarray): Numpy array of shape (T,H,W) for ground truth masks, where T is the number of identified trees, 
                          H and W are the height and width of the image.
        preds (ndarray): Numpy array of shape (T,H,W) for predicted masks, where T is the number of identified trees,
                         H and W are the height and width of the image.
        reduction (str): If 'none', return metrics for each tree, if "mean" return average metrics.
    
    return:
        IoU (ndarray)
    '''
    intersection = np.logical_and(truths, preds).sum(axis=(1,2))
    union = np.logical_or(truths, preds).sum(axis=(1,2))
    iou = np.divide(intersection, union, out=np.zeros(union.shape), where=union!=0)


    # If reduction is 'mean', return the mean IoU, otherwise return all IoUs
    if reduction == 'none':
        return iou
    elif reduction == 'mean':
        return np.mean(iou)
    else:
        raise ValueError(f"reduction should be either 'mean' or 'none'.")


def segmentation_metrics(truths, preds, reduction='none'):
    '''
    Calculate segmentation metrics between matched masks, return all or average.

    params:
        truths (ndarray): Numpy array of shape (T,H,W) for ground truth masks, where T is the number of identified trees, 
                          H and W are the height and width of the image.
        preds (ndarray): Numpy array of shape (T,H,W) for predicted masks, where T is the number of identified trees,
                         H and W are the height and width of the image.
        reduction (str): If 'none', return metrics for each tree, if "mean" return average metrics.
    
    return:
        precision, recall, f1 (ndarrays)
    '''
    tp = np.logical_and(truths, preds).sum(axis=(1,2))
    fp = np.logical_and(np.logical_not(truths), preds).sum(axis=(1,2))
    fn = np.logical_and(truths, np.logical_not(preds)).sum(axis=(1,2))

    precision = np.divide(tp, (tp+fp), out=np.zeros(tp.shape), where=(tp+fp)!=0)
    recall = np.divide(tp, (tp+fn), out=np.zeros(tp.shape), where=(tp+fn)!=0)
    f1 = np.divide((2*precision*recall), (precision+recall), out=np.zeros(recall.shape), where=(precision+recall)!=0)

    if reduction == 'none':
        return precision, recall, f1
    elif reduction == 'mean':
        return np.mean(precision), np.mean(recall), np.mean(f1)
    else:
        raise ValueError(f"reduction should be either 'mean' or 'none'.")


def detection_metrics(truths, preds, threshold=0.5):
    '''
    Calculate detection metrics between matched masks above threshold.

    params:
        truths (ndarray): Numpy array of shape (T,H,W) for ground truth masks, where T is the number of identified trees, 
                          H and W are the height and width of the image.
        preds (ndarray): Numpy array of shape (T,H,W) for predicted masks, where T is the number of identified trees,
                         H and W are the height and width of the image.
        threshold (float): IoU threshold for prediction and ground truth to be considered a "True Positive"
    
    return:
        precision, recall, f1 (ndarrays)
    '''
    true_idx, pred_idx = hungarian_matching(truths, preds, threshold=threshold)
    tp = len(true_idx)
    fp = len(preds) - len(pred_idx)
    fn = len(truths) - len(true_idx)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def per_tree_metrics(truths, preds, detection_threshold=0.5, segmentation_threshold=0.5):
    '''
    Calculate detection (object-based) and segmentation (pixel-based) metrics across all trees, optionally across multiple
    images. This means if one image has 1 tree and another image has 10 trees, metrics are calculated across all 11 trees
    rather than separately on the 1 tree and the 10 trees and then averaged together.

    params:
        truths (list): List of sv.Detections objects. Each Detections object includes the ground truth bounding boxes 
                       and masks for a single image.
        preds (list): List of sv.Detections objects. Each Detections object includes the prompt bounding boxes and
                      predicted masks for a single image.
        detection_threshold (float): Minimum IoU between predicted and ground truth masks to count as a True Positive
                                     for detection metrics
        segmentation_threshold (float): Minimum IoU between predicted and ground truth boxes to include a prediction in
                                        the segmentation metrics
    
    return:
        object-based precision, recall, and f1, pixel-based precision, recall, f1, and iou (tuple): Metrics across all trees.
    '''
    # List of ground truths and list of predictions should be same length (i.e. same number of images with
    # ground truth labels and predicted labels)
    assert len(truths) == len(preds)

    detect_tp = []
    detect_fp = []
    detect_fn = []
    segment_precision = []
    segment_recall = []
    segment_f1 = []
    segment_iou = []

    # Iterate thru each image
    for i in range(len(truths)):
        true_boxes = truths[i].xyxy
        true_masks = truths[i].mask
        pred_boxes = preds[i].xyxy
        pred_masks = preds[i].mask

        # Match ground truth and predicted masks for given image above detection_threshold
        true_idx, pred_idx = hungarian_matching(true_masks, pred_masks, threshold=detection_threshold)

        # Record detection (tree-wise) TP, FP, and FN
        detect_tp.append(len(true_idx))
        detect_fp.append(len(pred_boxes) - len(pred_idx))
        detect_fn.append(len(true_boxes) - len(true_idx))

        # Match ground truth and predicted boxes for given image above segmentation_threshold
        # to select corresponding masks to run segmentation metrics on
        segment_true_idx, segment_pred_idx = hungarian_matching(true_boxes, pred_boxes, threshold=segmentation_threshold)
        true_masks = true_masks[segment_true_idx]
        pred_masks = pred_masks[segment_pred_idx]

        # Calculate segmentation (pixel-wise) TP, FP, and FN
        segment_tp = np.logical_and(true_masks, pred_masks).sum(axis=(1,2))
        segment_fp = np.logical_and(np.logical_not(true_masks), pred_masks).sum(axis=(1,2))
        segment_fn = np.logical_and(true_masks, np.logical_not(pred_masks)).sum(axis=(1,2))

        # Use pixel-wise TP, FP, and FN to calculate tree-wise segmentation metrics
        precision = np.divide(segment_tp, (segment_tp+segment_fp), out=np.zeros(segment_tp.shape), where=(segment_tp+segment_fp)!=0)
        recall = np.divide(segment_tp, (segment_tp+segment_fn), out=np.zeros(segment_tp.shape), where=(segment_tp+segment_fn)!=0)
        f1 = np.divide((2*precision*recall), (precision+recall), out=np.zeros(recall.shape), where=(precision+recall)!=0)
        iou = np.divide(segment_tp, (segment_tp+segment_fp+segment_fn), out=np.zeros(segment_tp.shape), where=(segment_tp+segment_fp+segment_fn)!=0)
        segment_precision += list(precision)
        segment_recall += list(recall)
        segment_f1 += list(f1)
        segment_iou += list(iou)

    # Calculate detection precision, recall, and f1 for all trees across all images
    detect_tp = sum(detect_tp)
    detect_fp = sum(detect_fp)
    detect_fn = sum(detect_fn)
    detect_precision = detect_tp / (detect_tp + detect_fp) if (detect_tp + detect_fp) > 0 else 0
    detect_recall = detect_tp / (detect_tp + detect_fn) if (detect_tp + detect_fn) > 0 else 0
    detect_f1 = (2 * detect_precision * detect_recall) / (detect_precision + detect_recall) if (detect_precision + detect_recall) > 0 else 0

    # Average segmentation precision, recall, f1, and IoU for all trees across all images
    segment_precision = sum(segment_precision) / len(segment_precision) if len(segment_precision) > 0 else 0
    segment_recall = sum(segment_recall) / len(segment_recall) if len(segment_recall) > 0 else 0
    segment_f1 = sum(segment_f1) / len(segment_f1) if len(segment_f1) > 0 else 0
    segment_iou = sum(segment_iou) / len(segment_iou) if len(segment_iou) > 0 else 0

    return np.array([detect_precision, detect_recall, detect_f1,
                     segment_precision, segment_recall, segment_f1, segment_iou])
    

def per_tree_std(truths, preds, segmentation_threshold=0.5):
    '''
    Calculate standartd deviation of segmentation (pixel-based) metrics across all trees, optionally across multiple
    images. This means if one image has 1 tree and another image has 10 trees, metrics are calculated across all 11
    trees rather than separately on the 1 tree and the 10 trees and then averaged together.

    params:
        truths (list): List of sv.Detections objects. Each Detections object includes the ground truth bounding boxes 
                       and masks for a single image.
        preds (list): List of sv.Detections objects. Each Detections object includes the prompt bounding boxes and
                      predicted masks for a single image.
        segmentation_threshold (float): Minimum IoU between predicted and ground truth boxes to include a prediction in
                                        the segmentation metrics
    
    return:
        pixel-based precision, recall, f1, and iou (tuple): Standard deviations of these metrics across all trees.
    '''
    # List of ground truths and list of predictions should be same length (i.e. same number of images with
    # ground truth labels and predicted labels)
    assert len(truths) == len(preds)
    
    segment_precision = []
    segment_recall = []
    segment_f1 = []
    segment_iou = []

    for i in range(len(truths)):
        true_boxes = truths[i].xyxy
        true_masks = truths[i].mask
        pred_boxes = preds[i].xyxy
        pred_masks = preds[i].mask

        # Match ground truth and predicted boxes for given image above segmentation_threshold
        # to select corresponding masks to run segmentation metrics on
        segment_true_idx, segment_pred_idx = hungarian_matching(true_boxes, pred_boxes, threshold=segmentation_threshold)
        true_masks = true_masks[segment_true_idx]
        pred_masks = pred_masks[segment_pred_idx]

        # Calculate segmentation (pixel-wise) TP, FP, and FN
        segment_tp = np.logical_and(true_masks, pred_masks).sum(axis=(1,2))
        segment_fp = np.logical_and(np.logical_not(true_masks), pred_masks).sum(axis=(1,2))
        segment_fn = np.logical_and(true_masks, np.logical_not(pred_masks)).sum(axis=(1,2))

        # Use pixel-wise TP, FP, and FN to calculate tree-wise segmentation metrics
        precision = np.divide(segment_tp, (segment_tp+segment_fp), out=np.zeros(segment_tp.shape), where=(segment_tp+segment_fp)!=0)
        recall = np.divide(segment_tp, (segment_tp+segment_fn), out=np.zeros(segment_tp.shape), where=(segment_tp+segment_fn)!=0)
        f1 = np.divide((2*precision*recall), (precision+recall), out=np.zeros(recall.shape), where=(precision+recall)!=0)
        iou = np.divide(segment_tp, (segment_tp+segment_fp+segment_fn), out=np.zeros(segment_tp.shape), where=(segment_tp+segment_fp+segment_fn)!=0)
        segment_precision += list(precision)
        segment_recall += list(recall)
        segment_f1 += list(f1)
        segment_iou += list(iou)

    return np.std(segment_precision), np.std(segment_recall), np.std(segment_f1), np.std(segment_iou)


def box_mask_metrics(truths, preds, detection_threshold=0.5, segmentation_threshold=0.5):
    '''
    Calculate object-based metrics for both bounding boxes and masks. Used for an experiment in the
    paper comparing both.

    params:
        truths (list): List of sv.Detections objects. Each Detections object includes the ground truth
                    bounding boxes and masks for a single image.
        preds (list): List of sv.Detections objects. Each Detections object includes the predicted
                    bounding boxes and masks for a single image.
        detection_threshold (float): Minimum IoU between predicted and ground truth masks to count
                                    as a True Positive for detection metrics
        segmentation_threshold (float): Minimum IoU between predicted and ground truth boxes to include
                                        a prediction in the segmentation metrics
    returns:
        bounding box precision, recall, and f1, mask precision, recall, and f1 (tuple)
    '''
    # List of ground truths and list of predictions should be same length (i.e. same number of images with
    # ground truth labels and predicted labels)
    assert len(truths) == len(preds)

    box_tp, box_fp, box_fn = [],[],[]
    mask_tp, mask_fp, mask_fn = [],[],[]

    # Iterate thru each image
    for i in range(len(truths)):
        true_boxes = truths[i].xyxy
        true_masks = truths[i].mask
        pred_boxes = preds[i].xyxy
        pred_masks = preds[i].mask

        # Match ground truth and predicted boxes for given image above detection_threshold
        true_idx, box_idx = hungarian_matching(true_boxes, pred_boxes, threshold=detection_threshold)

        # Record box (tree-wise) TP, FP, and FN
        box_tp.append(len(true_idx))
        box_fp.append(len(pred_boxes) - len(box_idx))
        box_fn.append(len(true_boxes) - len(true_idx))

        # Match ground truth and predicted masks for given image above detection_threshold
        true_idx, mask_idx = hungarian_matching(true_masks, pred_masks, threshold=detection_threshold)

        # Record mask (tree-wise) TP, FP, and FN
        mask_tp.append(len(true_idx))
        mask_fp.append(len(pred_masks) - len(mask_idx))
        mask_fn.append(len(true_masks) - len(true_idx))

    # Calculate box and mask precision, recall, and f1 for all trees across all images
    box_tp = sum(box_tp)
    box_fp = sum(box_fp)
    box_fn = sum(box_fn)

    mask_tp = sum(mask_tp)
    mask_fp = sum(mask_fp)
    mask_fn = sum(mask_fn)   

    box_precision = box_tp / (box_tp + box_fp) if (box_tp + box_fp) > 0 else 0
    box_recall = box_tp / (box_tp + box_fn) if (box_tp + box_fn) > 0 else 0
    box_f1 = (2 * box_precision * box_recall) / (box_precision + box_recall) if (box_precision + box_recall) > 0 else 0

    mask_precision = mask_tp / (mask_tp + mask_fp) if (mask_tp + mask_fp) > 0 else 0
    mask_recall = mask_tp / (mask_tp + mask_fn) if (mask_tp + mask_fn) > 0 else 0
    mask_f1 = (2 * mask_precision * mask_recall) / (mask_precision + mask_recall) if (mask_precision + mask_recall) > 0 else 0
 

    return np.array([box_precision, box_recall, box_f1,
                     mask_precision, mask_recall, mask_f1])