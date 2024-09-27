import os
import laspy
import rasterio
import numpy as np
import pandas as pd
import supervision as sv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def rasterize_lidar(lidar_folder, rgb_folder, filename, label=False, boxes=None,
                    img_size=(400,400), min_threshold=1):
    '''
    This function is written to work with the images and labels provided in the NEONTreeEvaluation dataset.
    To make it generalizable to other datasets, we would need to include user-specified parameters for image
    size and tree labels, as well as probably some other adjustments.

    params:
        lidar_folder (str): Path to folder containing laz files
        rgb_folder (str): Path to folder containing tiff files
        filename (str): Lidar / tiff filename minus the extension
        label (bool): True if Lidar data already has "label" field with individual
                      tree labels
        boxes (np.array): T x 4 array of [x0,y0,x1,y1] bounding boxes, where T is the
                          total number of boxes (trees). If provided, used to label
                          Lidar points for individual trees.
        img_size (tuple): RGB image size in pixels (H x W)
        min_threshold (int): Number of Lidar points that fall within a given pixel in
                             order to add that pixel as a point prompt. Default is 1.

    returns:
        coord_array: 1 x P x 2 array of X,Y coordinates in pixel space, where P is the
                     number of rasterized points
        label_array: N x P array of 1s and 0s, 1 to indicate the corresponding entry in
                     coord_array is a specific tree, 0 to indicate background or a different
                     tree. If using common label for all trees, N is 1.
    '''

    # Read Lidar file
    las = laspy.read(os.path.join(lidar_folder, f'{filename}.laz'))

    # LiDAR to DataFrame
    points = np.vstack((las.x, las.y, las.z)).transpose()
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])

    # Align Lidar coordinates with tiff indices
    with rasterio.open(os.path.join(rgb_folder, filename+'.tif')) as rast_img:
        left, bottom, right, top = rast_img.bounds
        df = df[df['x'].between(left, right) & df['y'].between(bottom, top)]        
        pixels = [rast_img.index(x,y) for x,y in zip(df['x'], df['y'])]
        # Points on right-most and bottom-most edges of the image are recorded at index 400, should be 399
        df['x_bin'] = [pixel[1] if pixel[1]!=400 else 399 for pixel in pixels]
        df['y_bin'] = [pixel[0] if pixel[0]!=400 else 399 for pixel in pixels]
        df = df.astype({'x_bin':'int', 'y_bin':'int'})

    # If Lidar data has labels, add these to dataframe
    if label:
        df['label'] = las.label
        num_trees = df['label'].max()

    # Otherwise, if boxes provided, use these to label Lidar data
    elif boxes is not None:
        num_trees = len(boxes)
        df['label'] = 0
        for i, box in enumerate(boxes):
            xmin, xmax = box[0], box[2]
            ymin, ymax = box[1], box[3]
            df.loc[df['x_bin'].between(xmin, xmax) & df['y_bin'].between(ymin, ymax) & df['z'].gt(2), 'label'] = i+1

    # Otherwise use a height threshold to label all points above 2 meters as "1" for "tree"
    else:
        df['label'] = 0
        df.loc[df['z']>2, 'label'] = 1
        num_trees = 1

    # Record the number of LiDAR points in each grid space (count) and the highest label (max)
    grid_labels = df.groupby(['x_bin', 'y_bin'])['label'].agg(['max','count']).reset_index()

    # use min_threshold to select valid points
    valid_points = grid_labels[grid_labels['count'] >= min_threshold]

    # 1 x N x 2
    coord_array = valid_points[['x_bin', 'y_bin']].to_numpy()[np.newaxis,:]
    # Create an array for each tree, with ones for that tree and zeros elsewhere, then stack together.
    # Assign based on the highest label (max)
    label_array = np.zeros((num_trees, len(valid_points)), dtype=int)
    for tree_id in range(1, num_trees + 1):
        label_array[tree_id - 1] = (valid_points['max'] == tree_id).astype(int).to_numpy()

    return coord_array, label_array


def kmeans_cluster(coordinates, labels, num_clusters, seed=None):
    '''
    Use KMeans clustering to group (rasterized) LiDAR points collectively labeled as "tree"
    into separate labels for each individual tree. The user must specify the number of
    clusters, which should be based on the number of trees known / estimated to be in the image.

    params:
        coordinates (ndarray): 1 x N x 2 array of X,Y coordinates, where N is the total
                               number of points
        labels (ndarray): 1 x N array of binary labels, 1 for tree and 0 for background
        num_clusters (int): Known or estimated number of trees in the image
        seed (int): Seed for the random state of the KMeans algorithm. Set to an integer
                    for reproducible results. Default is None.

    returns:
        kmeans_coordinates (ndarray): 1 x N x 2 array of X,Y coordinates, shape matches input
        kmeans_labels (ndarray): T x N array of binary labels, where T is the number of
                                 trees / clusters, and each label in N is now specific
                                 to its corresponding tree / cluster in T
    '''
    kmeans_coordinates = []
    kmeans_labels = []

    # Get the indices of tree (positive) and background (negative) points
    pos_indices = labels.nonzero()[1]
    neg_indices = (labels==0).nonzero()[1]

    # Get the corresponding sets of coordinates
    pos_coords = coordinates[0, pos_indices]
    neg_coords = coordinates[0, neg_indices]

    # Group tree points into N clusters and put their labels into an array (with background labels at the end)
    if len(pos_indices) > num_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init='auto')
        cluster_labels = kmeans.fit_predict(pos_coords) + 1
        all_labels = np.concatenate([cluster_labels, np.zeros(len(neg_indices), dtype=int)])
    else:
        raise ValueError(f'{num_clusters} clusters but only {len(pos_indices)} positive points. Reduce num_clusters to be no more than the number of positive points.')

    # Put tree and background coordinates into 1 x N x 2 array to match input
    kmeans_coordinates = np.concatenate([pos_coords, neg_coords], axis=0)[np.newaxis,:]
    
    # Put labels into C x N array, so that each unique tree C has its own array of N binary labels
    kmeans_labels = np.zeros((num_clusters, len(all_labels)), dtype=int)
    for label in range(1, num_clusters + 1):
      kmeans_labels[label - 1] += (all_labels==label)

    return kmeans_coordinates, kmeans_labels


def sample_points(coordinates, labels, pos_samples, neg_samples, 
                  distance_weight=False, neg_sample_spread=20, neg_min_distance=True, seed=None):
    '''
    Sample from rasterized LiDAR points. Can specify the number of positive and negative samples 
    taken. If points are labeled collectively, will sample pos_samples and neg_samples from the
    entire image. If points are labeled individually, will sample pos_samples and neg_samples per
    tree. For individually labeled points, can choose to sample points uniformly or weighted by
    distance from the center of the tree. Weights will prioritize positive and negative points
    near the edges of trees to better delineate them.

    params:
        coordinates (ndarray): 1 x N x 2 array of X,Y coordinates, where N is the total number of points
        labels (ndarray): T x N array of binary labels, where T is the number of labeled trees
                          (1 for collective labels)
        pos_samples (int): The number of positive samples to draw, either in total for collectively
                           labeled points or per tree for individually labeled points
        neg_samples (int): The number of negative samples to draw, either in total for collectively
                           labeled points or per tree for individually labeled points
        distance_weight (bool): If True, give higher weight to points near the edges of trees; may
                                produce undesirable results with collectively labeled points, suggest
                                setting to False in this case
        neg_sample_spread (int): When using distance weights, use this number to adjust how spread out
                                 negative samples are from the tree; higher values indicate higher spread,
                                 while lower values indicate tighter clustering around trees
        neg_min_distance (bool): If True, the closest negative point can be no closer to the center of the
                                 tree than the farthest positive point; only applies with distance weights
        seed (int): Random seed, set to an integer for reproducible results

    returns:
        sample_coordinates: T x (pos_samples + neg_samples) x 2 array of X,Y coordinates; if collectively
                            labeled points then T = 1
        sample_labels: T x (pos_samples + neg_samples) array of binary labels, with 1 for trees and 0 for
                       background; if collectively labeled points then T = 1    
    '''
    rng = np.random.default_rng(seed)
    sample_coordinates = []
    sample_labels = []


    # For each individual tree (or all trees if collectively labeled):
    for tree in labels:

        # Get the indices of the positive and negative points for that tree
        pos_indices = tree.nonzero()[0]
        neg_indices = (tree==0).nonzero()[0]

        # Get the coordinates for the positive and negative points
        pos_coords = coordinates[0,pos_indices]
        neg_coords = coordinates[0,neg_indices]

        # If there are fewer positive points than pos_samples, supplement with additional negative samples
        if len(pos_indices) >= pos_samples:
            num_pos = pos_samples
        else:
            num_pos = len(pos_indices)
        num_neg = pos_samples - num_pos + neg_samples

        # Distance weighting is only possible with > 0 positive points
        if distance_weight is True and num_pos > 0:
            # Find the center pixel of the positive points
            pos_center = pos_coords.mean(axis=0).reshape(1,-1)

            # Find the distances of all positive and negative points from the positive center
            pos_distances = euclidean_distances(pos_coords, pos_center)[:,0]
            neg_distances = euclidean_distances(neg_coords, pos_center)[:,0]

            # If neg_min_distance, only sample negative points that are at least as far from 
            # the tree centroid as the farthest positive point
            if neg_min_distance:
                min_distance = pos_distances.max()
            else:
                min_distance = 0.0
            neg_distances[neg_distances < min_distance] = np.inf

            # Use distances to calculate sampling probabilities
            # (Boundary points between tree vs background or tree vs tree have higher probability of being sampled)
            if num_pos > 1:
                pos_probs = pos_distances / sum(pos_distances)
            else:
                pos_probs = [1.0]

            # Use neg_sample_spread to further weight spread of negative points
            spread = 1 / neg_sample_spread
            neg_exp = np.exp(-spread*neg_distances)
            neg_probs = neg_exp / sum(neg_exp)

            # Select a random subset of positive indices
            pos_indices = rng.choice(pos_indices, num_pos, replace=False, p=pos_probs, shuffle=False)
            neg_indices = rng.choice(neg_indices, num_neg, replace=False, p=neg_probs, shuffle=False)

        else:
            pos_indices = rng.choice(pos_indices, num_pos, replace=False, shuffle=False)
            neg_indices = rng.choice(neg_indices, num_neg, replace=False, shuffle=False)

        # Combine positive and negative indices, use them to sample coordinates for given tree
        indices = np.concatenate([pos_indices, neg_indices])
        sample_coordinates.append(coordinates[0, indices])

        # Create new labels based on how many positive and negative samples were taken
        sample_labels.append(np.concatenate([np.ones(pos_indices.shape), np.zeros(neg_indices.shape)]))

    # Stack coordinates and labels, then return both arrays
    sample_coordinates = np.stack(sample_coordinates)
    sample_labels = np.stack(sample_labels)

    return sample_coordinates, sample_labels


def show_as_mask(img, detections, coordinates, labels, ax=None,
                 show_positive=True, show_negative=False, show_boxes=False, 
                 img_size=(400,400), fig_size=(5,5), neg_opacity=1.0, title=None, 
                 save=False, output_folder='/content'):
    '''
    Display rasterized LiDAR points as mask, with one pixel per one point. Useful for visualizing images
    with large numbers of LiDAR points. Can visualize collectively labeled or individually labeled points.

    params:
        img (array): RGB image as array
        detections (sv.Detections): Annotations for image put into Supervision Detections object
        coordinates (ndarray): 1 x N x 2 array or T x N x 2 array of X,Y coordinates; if coordinates were
                               sampled using "sample_points", then T is the number of labeled trees
        labels (ndarray): T x N array of binary labels, where T is the number of labeled trees (T = 1 for
                          collective labels)
        ax (matplotlib Axes): If embedding this image in a larger matplotlib figure, pass an axis of that
                              figure as "ax". This will return the image as an Axes object and allow it
                              to be embedded in the larger figure.
        show_positive (bool): If True, show the positively labeled points (trees)
        show_negative (bool): If True, show the negatively labeled points (background)
        show_boxes (bool): If True, show the bounding boxes around trees
        img_size (tuple): Image size in pixels (H x W)
        fig_size (tuple): Display size in inches (W x H)
        neg_opacity (float): Opacity of negative points, value between 0 and 1
        title (str): Optional title to show on image. If title is provided as save is True, title will
                     also be the name of the saved file.
        save (bool): If True, save the image as png. If title is not None, will use title for save name,
                     otherwise will save as "lidar_mask.png".
        output_folder (str): Path to the folder to save image
    
    returns:
        None. Displays and optionally saves image with masks overlaid.
    '''
    img = img[:,:,::-1]

    # T = 1 for collective points, otherwise the number of labeled trees
    num_labels = len(labels)

    # Broadcast coordinates to T x N x 2 if not already input as such
    original_shape = coordinates.shape
    coordinates = np.broadcast_to(coordinates, (num_labels, coordinates.shape[1], coordinates.shape[2]))

    if show_positive:
        # Show tree points
        pos_mask = np.full((num_labels, img_size[0], img_size[1]), False)
        for i in range(num_labels):
            mask_index = labels[i].nonzero()[0]
            mask_coords = coordinates[i, mask_index]
            pos_mask[i,mask_coords[:,1],mask_coords[:,0]] = True
        detections.mask = pos_mask
        # Set colors of masks depending on number of labels
        if num_labels == 1:
            pos_annotator = sv.MaskAnnotator(color=sv.Color.BLACK, opacity=1)
        else:
            pos_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=1)
        img = pos_annotator.annotate(scene=img.copy(), detections=detections)

    if show_negative:
        # Show background points
        neg_mask = np.full((original_shape[0], img_size[0], img_size[1]), False)
        if original_shape[0] == 1:
            labels = np.max(labels, axis=0)[np.newaxis]
        for i in range(len(neg_mask)):
            mask_index = (labels[i]==0).nonzero()[0]
            mask_coords = coordinates[i, mask_index]
            neg_mask[i,mask_coords[:,1],mask_coords[:,0]] = True
        detections.mask = neg_mask
        # Background points are always white for mask visualization
        neg_annotator = sv.MaskAnnotator(color=sv.Color.WHITE, opacity=neg_opacity)
        img = neg_annotator.annotate(scene=img.copy(), detections=detections)

    if show_boxes:
        # Show boxes
        box_annotator = sv.BoxAnnotator(thickness=1, color=sv.Color.RED)
        img = box_annotator.annotate(scene=img.copy(), detections=detections)

    # Display figure, optionally title and save
    return_ax = True
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        return_ax = False
    ax.axis('off')
    ax.imshow(img[:,:,::-1])

    if return_ax:
        return ax

    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save:
        if title:
            plt.savefig(os.path.join(output_folder, title+'.png'))
        else:
            plt.savefig(os.path.join(output_folder, 'lidar_mask.png'))
    plt.show()


def show_as_points(img, detections, coordinates, labels, ax=None,
                   show_positive=True, show_negative=False, show_boxes=False, 
                   img_size=(400,400), fig_size=(5,5), marker_size=75, title=None, 
                   save=False, output_folder='/content'):
    '''
    Display rasterized LiDAR points as dots, larger than a pixel. Useful for visualizing images with few
    LiDAR points. Can visualize collectively labeled or individually labeled points, however multiple trees
    with individually labeled points can become visually confusing (especially if showing negative points). 
    Alternatively, can visualize each tree's points in a separate image using the code below:

    for i in range(len(labels)):
        show_as_points(img, coordinates[i:i+1], labels[i:i+1])

    params:
        img (array): RGB image as array
        detections (sv.Detections): Annotations for image put into Supervision Detections object
        coordinates (ndarray): 1 x N x 2 array or T x N x 2 array of X,Y coordinates; if coordinates were
                               sampled using "sample_points", then T is the number of labeled trees
        labels (ndarray): T x N array of binary labels, where T is the number of labeled trees (T = 1 for
                          collective labels)
        ax (matplotlib Axes): If embedding this image in a larger matplotlib figure, pass an axis of that
                              figure as "ax". This will return the image as an Axes object and allow it
                              to be embedded in the larger figure.
        show_positive (bool): If True, show the positively labeled points (trees)
        show_negative (bool): If True, show the negatively labeled points (background)
        show_boxes (bool): If True, show the bounding boxes around trees
        marker_size (int): Size of dots to display for LiDAR points
        img_size (tuple): Image size in pixels (H x W)
        fig_size (tuple): Display size in inches (W x H)
        title (str): Optional title to show on image. If title is provided as save is True, title will
                     also be the name of the saved file.
        save (bool): If True, save the image as png. If title is not None, will use title for save name,
                     otherwise will save as "lidar_mask.png".
        output_folder (str): Path to the folder to save image
    
        returns:
            None. Displays and optionally saves image with points overlaid.
    '''
    # Set colors for individually labeled points
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[:7] + colors[8:]

    if show_boxes:
        # Show boxes  (we use color BLUE because it will be reversed to RED later)
        box_annotator = sv.BoxAnnotator(thickness=1, color=sv.Color.BLUE)
        img = box_annotator.annotate(scene=img.copy(), detections=detections)

    # Create figure, display image
    return_ax = True
    if ax is None:
      fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
      return_ax = False
    ax.axis('off')
    ax.imshow(img)

    # T = 1 for collective points, otherwise the number of labeled trees
    num_labels = len(labels)

    if show_positive:
        # Show tree points
        if num_labels == 1:
            colors = ['black']
        for i in range(num_labels):
            pos_index = labels[i].nonzero()[0]
            pos_coord = coordinates[i, pos_index]
            ax.scatter(pos_coord[:,0], pos_coord[:,1], color=colors[i%len(colors)], marker='.', s=marker_size, edgecolor=colors[i%len(colors)], linewidth=1)

    if show_negative:
        # Show background points
        if num_labels == 1:
            colors = ['white']
        else:
            marker_size += 25
        for i in range(num_labels):
            neg_index = (labels[i]==0).nonzero()[0]
            neg_coord = coordinates[i, neg_index]
            ax.scatter(neg_coord[:,0], neg_coord[:,1], color='white', marker='.', s=marker_size, edgecolor=colors[i%len(colors)], linewidth=1)

    if return_ax:
      return ax

    # Display figure, optionally title and save
    if title:
        plt.title(title)
    plt.tight_layout()
    if save:
        if title:
            plt.savefig(os.path.join(output_folder, title+'.png'))
        else:
            plt.savefig(os.path.join(output_folder, 'lidar_points.png'))
    plt.show()