import os
import numpy as np
import pandas as pd
import laspy
from sklearn.cluster import KMeans


def rasterize_lidar(lidar_folder, filename, individual_labels, min_threshold=1):
    '''
    This function is written to work with the images and labels provided in the NEONTreeEvaluation dataset.
    To make it generalizable to other datasets, we would need to include user-specified parameters for image 
    size and tree labels, as well as probably some other adjustments.

    params:
        lidar_folder (str): Path to folder containing laz files
        filename (str): Lidar filename minus the ".laz" extension
        individual_labels (bool): True to assign each tree its unique label in Lidar,
                                  False to assign all trees a common label of 1
        min_threshold (int): Number of Lidar points that fall within a given pixel in 
                             order to add that pixel as a point prompt. Default is 1.
    
    returns:
        coord_array: N x P x 2 array of pixel coordinates, where N is the number of trees,
                     P is the number of points, and 2 is the x and y coordinates. If using
                     common label for all trees, N is 1.
        label_array: N x P array of 1s and 0s, 1 to indicate the corresponding entry in
                     coord_array is a specific tree, 0 to indicate background or a different
                     tree. If uusing common label for all trees, N is 1.
    '''

    # TODO: Generalize function to non-NEON datasets
    
    # Read Lidar file
    las = laspy.read(os.path.join(lidar_folder, f'{filename}.laz'))

    # LiDAR to DataFrame
    points = np.vstack((las.x, las.y, las.z)).transpose()
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])

    # Add label
    df['label'] = las.label

    # Define grid
    xmin, xmax = df['x'].min(), df['x'].max()
    ymin, ymax = df['y'].min(), df['y'].max()

    x_bins = np.linspace(xmin, xmax, 401)
    y_bins = np.linspace(ymin, ymax, 401)

    # Allocate X和Y
    df['x_bin'] = pd.cut(df['x'], bins=x_bins, labels=False, include_lowest=True)
    df['y_bin'] = 399 - pd.cut(df['y'], bins=y_bins, labels=False, include_lowest=True)

    num_trees = df['label'].max()

    # count the number of LiDAR points in each grid (count) and the highest label (max)
    grid_labels = df.groupby(['x_bin', 'y_bin'])['label'].agg(['max','count']).reset_index()

    # use min_threshold to select valid points
    valid_points = grid_labels[grid_labels['count'] >= min_threshold]

    # N x 2
    coord_array = valid_points[['x_bin', 'y_bin']].to_numpy()[np.newaxis,:]
    if individual_labels:
        # Create an array for each tree, with ones for that tree and zeros elsewhere, then stack together.
        # Assign based on the highest label (max)
        label_array = np.zeros((num_trees, len(valid_points)), dtype=int)
        for tree_id in range(1, num_trees + 1):
            label_array[tree_id - 1] = (valid_points['max'] == tree_id).astype(int).to_numpy()
    else:
        # Assign 1 to a given pixel if any LiDAR point within that pixel has a label higher than 0
        label_array = (valid_points['max']>0).to_numpy(dtype='int')[np.newaxis,:]

    return coord_array, label_array


def kmeans_cluster(coordinates, labels, num_clusters, seed=None):
    rng = np.random.default_rng(seed)
    kmeans_coordinates = []
    kmeans_labels = []

    # Get the indices of the positive and negative points
    pos_indices = labels.nonzero()[1]
    neg_indices = (labels==0).nonzero()[1]

    pos_coords = coordinates[0, pos_indices]
    neg_coords = coordinates[0, neg_indices]

    if len(pos_indices) > num_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init='auto')
        cluster_labels = kmeans.fit_predict(pos_coords) + 1
        all_labels = np.concatenate([cluster_labels, np.zeros(len(neg_indices), dtype=int)])
    else:
        raise ValueError(f'{num_clusters} clusters but only {len(pos_indices)} positive points. Reduce num_clusters to be no more than the number of positive points.')

    kmeans_coordinates = np.concatenate([pos_coords, neg_coords], axis=0)[np.newaxis,:]
    kmeans_labels = np.zeros((num_clusters, len(all_labels)), dtype=int)
    for label in range(1, num_clusters + 1):
      kmeans_labels[label - 1] += (all_labels==label)

    return kmeans_coordinates, kmeans_labels