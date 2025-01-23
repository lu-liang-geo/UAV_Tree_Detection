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
    intersection = np.logical_and(masks_true[:,None,...], masks_pred).sum(axis=(2,3))
    union = np.logical_or(masks_true[:,None,...], masks_pred).sum(axis=(2,3))
    iou_matrix = np.divide(intersection, union, out=np.zeros(union.shape), where=union!=0)
    return iou_matrix


def boxes_to_masks(boxes, shape):
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
    precisions_per_tree = []
    recalls_per_tree = []
    f1_per_tree = []

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
    true_idx, pred_idx = hungarian_matching(truths, preds, threshold=threshold)
    tp = len(true_idx)
    fp = len(preds) - len(pred_idx)
    fn = len(truths) - len(true_idx)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1