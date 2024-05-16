# Copyright (c) 2023 William Locke

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import rasterio
import torch
import numpy as np
import xml.etree.ElementTree as ET

class RGBNEONTreeDataset(torch.utils.data.Dataset):
  def __init__(self, image_path, ann_path, check_values=False):
    '''
    params
      image_path (str): Path to top-level training image directory (contains RGB, LiDAR, Hyperspectral, CHM)
      ann_path (str):  Path to annotations

    __init__ stores filenames from RGB folder, which will be used to retrieve relevant files from other
    folders. Because RGB folder has four files not found in Hyperspectral or CHM, we remove any entries
    not found in these other folders. We also manually remove four images we consider unsuitable for
    training / evaluation, either because they have a large number of invalid pixels or they don't have
    good annotations.
    '''
    self.check_values = check_values
    self.image_path = image_path
    self.ann_path = ann_path
    file_names = os.listdir(os.path.join(image_path, 'RGB'))
    problem_files = set(['2019_SJER_4_251000_4103000_image', '2019_TOOL_3_403000_7617000_image', 'TALL_043_2019', 'SJER_062_2018'])
    basenames = [name.split('.')[0] for name in file_names]
    basenames = [name for name in basenames if name not in problem_files]
    basenames = [name for name in basenames if os.path.exists(os.path.join(ann_path,f'{name}.xml'))]
    self.basenames = list(set(basenames))

  def __len__(self):
    return len(self.basenames)

  def __getitem__(self, idx):
    '''
    returns
      annotated_image (dict) with keys:
        rgb_img: HxWxC ndarray of RGB channels
        multi_img: HxWxC ndarray of multi channels
        prompt (if prompt_path specified): Nx4 ndarray of prompt bounding boxes in XYXY format
        annotation: Nx4 ndarray of ground truth bounding boxes in XYXY format

    Currently I hardcode the multi_img channels, but later I may allow the user to specify them in __init__
    '''
    basename = self.basenames[idx]
    rgb_path = os.path.join(self.image_path, 'RGB', f'{basename}.tif')
    ann_path = os.path.join(self.ann_path, f"{basename}.xml")
    annotated_image = {'basename': basename}

    # Open RGB path and two paths used to construct multi image. Save rgb_img in annotated_image.
    with rasterio.open(rgb_path) as img:
      rgb_img = img.read().transpose(1,2,0)
    
    annotated_image['rgb'] = rgb_img

    # Check for any negative values in NIR, Red, Red-Edge, and CHM channels, which are likely invalid pixels.
    if self.check_values:
      assert rgb_img.min() >= 0, f'{basename} RGB values below 0 (min value: {rgb_img.min()}), check source image'
     
    # Set NaN and values less than 0 equal to 0. This allows for processing images with invalid pixel values
    # in some channels (we specifically allow this in 2019_OSBS_5 in the training set).
    rgb_img[rgb_img<0] = 0

    # Extract bounding boxes from annotations, save in annotated_image.
    xyxy = []
    tree = ET.parse(ann_path)
    root = tree.getroot()
    for obj in root.findall('object'):
      name = obj.find('name').text
      if name == 'Tree':
        bbox = obj.find('bndbox')
        xyxy.append([int(bbox[i].text) for i in range(4)])
    annotation = np.array(xyxy)
    annotated_image['annotation'] = annotation

    return annotated_image

  def get_image(self, basename, return_index=False):
    index = self.basenames.index(basename)
    if return_index:
      return index
    else:
      return self.__getitem__(index)