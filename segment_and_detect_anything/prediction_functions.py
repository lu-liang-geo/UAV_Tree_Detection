import numpy as np

def segment_boxes(sam_predictor, boxes, threshold=0.0) -> np.ndarray:
    result_mask = []
    for box in boxes:
        mask, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=False
        )
        if scores > threshold:
            result_mask.append(mask.astype('bool'))

    if len(result_mask) > 0:
        return np.concatenate(result_mask)
    else:
        return np.empty(shape=(0, mask.shape[0], mask.shape[1]))

def segment_points(sam_predictor, points, labels, threshold=0.0, iterative=True, return_iterations=False):
    result_mask = []
    for i, label in enumerate(labels):
        # If there is only one set of coordinates (i.e. points.shape[0]==1),
        # always use those same coordinates. If there are multiple sets of
        # coordinates (i.e. points.shape[0]>1), use different set for each label.
        # This is used for individually sampled vs collectively sampled points.
        if points.shape[0] == 1:
            tree_points = points[0]
        else:
            tree_points = points[i]

        if not iterative:
            mask, scores, logits = sam_predictor.predict(
                point_coords=tree_points,
                point_labels=label,
                multimask_output=False)
        else:
            # First Prediction,without logits
            mask, scores, logits = sam_predictor.predict(
                point_coords=tree_points[:1],
                point_labels=label[:1],
                multimask_output=False)
            if return_iterations:
                result_mask.append(mask.astype('bool'))
            # Subsequent Predictions, with logits
            for j in range(1, len(label)):
                mask, scores, logits = sam_predictor.predict(
                    point_coords=tree_points[:j+1],
                    point_labels=label[:j+1],
                    mask_input=logits,
                    multimask_output=False)
                if return_iterations:
                    result_mask.append(mask.astype('bool'))

        if not return_iterations:
            if scores > threshold:
                result_mask.append(mask.astype('bool'))

    if len(result_mask) > 0:
        return np.concatenate(result_mask)
    else:
        return np.empty(shape=(0, mask.shape[0], mask.shape[1]))

def segment_box_points(sam_predictor, boxes, points, labels, threshold=0.0, iterative=True, return_iterations=False):
    result_mask = []
    for i, box in enumerate(boxes):
        tree_points = points[i]
        tree_labels = labels[i]
        # If not iterative, use all prompts at once
        if not iterative:
            mask, scores, logits = sam_predictor.predict(
                box=box,
                point_coords=tree_points,
                point_labels=tree_labels,
                multimask_output=False)
        # Else run through prompts iteratively
        else:
            # First Prediction, using box
            mask, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=False)
            if return_iterations:
                result_mask.append(mask.astype('bool'))
            # Subsequent Predictions using box, new points, and previous mask
            for j in range(len(tree_labels)):
                mask, scores, logits = sam_predictor.predict(
                    box=box,
                    point_coords=tree_points[:j+1],
                    point_labels=tree_labels[:j+1],
                    mask_input=logits,
                    multimask_output=False)
                if return_iterations:
                    result_mask.append(mask.astype('bool'))

        if not return_iterations:
            if scores > threshold:
                result_mask.append(mask.astype('bool'))

    if len(result_mask) > 0:
        return np.concatenate(result_mask)
    else:
        return np.empty(shape=(0, mask.shape[0], mask.shape[1]))