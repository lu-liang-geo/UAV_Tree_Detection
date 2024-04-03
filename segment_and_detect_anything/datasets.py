# Copyright (c) 2023 William Locke

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import rasterio
import torch
import numpy as np
import xml.etree.ElementTree as ET

class NEONTreeDataset(torch.utils.data.Dataset):
  def __init__(self, image_path, ann_path, prompt_path=None, check_values=False):
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
    self.prompt_path = prompt_path
    file_names = os.listdir(os.path.join(image_path, 'RGB'))
    problem_files = set(['2019_SJER_4_251000_4103000_image', '2019_TOOL_3_403000_7617000_image', 'TALL_043_2019', 'SJER_062_2018'])
    basenames = [name.split('.')[0] for name in file_names]
    basenames = [name for name in basenames if name not in problem_files]
    basenames = [name for name in basenames if os.path.exists(os.path.join(image_path, 'Hyperspectral', f'{name}_hyperspectral.tif'))]
    basenames = [name for name in basenames if os.path.exists(os.path.join(ann_path,f'{name}.xml'))]
    basenames = [name for name in basenames if os.path.exists(os.path.join(image_path, 'CHM', f'{name}_CHM.tif'))]
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
    chm_path = os.path.join(self.image_path, 'CHM', f'{basename}_CHM.tif')
    hs_path = os.path.join(self.image_path, 'Hyperspectral', f'{basename}_hyperspectral.tif')
    ann_path = os.path.join(self.ann_path, f"{basename}.xml")
    if self.prompt_path:
      prompt_path = os.path.join(self.prompt_path, 'Boxes', f"{basename}.npy")
    annotated_image = {'basename': basename}

    # Open RGB path and two paths used to construct multi image. Save rgb_img in annotated_image.
    with rasterio.open(rgb_path) as img:
      rgb_img = img.read().transpose(1,2,0)
    with rasterio.open(hs_path) as img:
      hs_img = img.read()
    with rasterio.open(chm_path) as img:
      chm_img = img.read()
    annotated_image['rgb'] = rgb_img

    # Remove blank rows or columns from edges of CHM and Hyperspectral images, based on null value of -9999.0 in CHM.
    if chm_img[0,0,1]==-9999.0:
      chm_img = chm_img[:,1:,:]
      hs_img = hs_img[:,1:,:]
    if chm_img[0,1,0]==-9999.0:
      chm_img = chm_img[:,:,1:]
      hs_img = hs_img[:,:,1:]
    if chm_img[0,-1,-2]==-9999.0:
      chm_img = chm_img[:,:-1,:]
      hs_img = hs_img[:,:-1,:]
    if chm_img[0,-2,-1]==-9999.0:
      chm_img = chm_img[:,:,:-1]
      hs_img = hs_img[:,:,:-1]

    assert not (chm_img==-9999.0).any()

    # Select NIR, Red, and Red-Edge channels based on frequency reference chart
    # https://github.com/weecology/NeonTreeEvaluation/blob/master/neon_aop_bands.csv
    # Note that reference chart starts at 1 while we start at 0, so add 1 to the numbers below to get corresponding chart row.
    nir = 95
    red = 53
    edge = 69

    # Extract NIR, Red, and Red-Edge channels from Hyperspectral Image. Make CHM single channel.
    nir_img = hs_img[nir]
    red_img = hs_img[red]
    edge_img = hs_img[edge]
    chm_img = chm_img[0]

    # Check for any negative values in NIR, Red, Red-Edge, and CHM channels, which are likely invalid pixels.
    if self.check_values:
      assert rgb_img.min() >= 0, f'{basename} RGB values below 0 (min value: {rgb_img.min()}), check source image'
      assert nir_img.min() >= 0, f'{basename} NIR values below 0 (min value: {nir_img.min()}), check source image.'
      assert red_img.min() >= 0, f'{basename} Red values below 0 (min value: {red_img.min()}), check source image.'
      assert edge_img.min() >= 0, f'{basename} Red Edge values below 0 (min value: {edge_img.min()}), check source image.'
      assert chm_img.min() >= 0, f'{basename} Canopy Height values below 0 (min value: {chm_img.min()}), check source image.'

    # Set NaN and values less than 0 equal to 0. This allows for processing images with invalid pixel values
    # in some channels (we specifically allow this in 2019_OSBS_5 in the training set).
    rgb_img[rgb_img<0] = 0
    nir_img[nir_img<0] = 0
    red_img[red_img<0] = 0
    edge_img[edge_img<0] = 0
    chm_img = np.nan_to_num(chm_img, nan=0.0)

    # Create NDVI from NIR and Red channels. NaN values (where NIR and Red are both 0) converted to 0.
    _ndvi_img = (nir_img - red_img) / (nir_img + red_img)
    ndvi_img = np.nan_to_num(_ndvi_img, nan=0)

    # Standardize Red-Edge channel
    edge_img = (edge_img - edge_img.mean()) / edge_img.std()

    # Standardize CHM channel
    chm_img = (chm_img - chm_img.mean()) / chm_img.std()

    # Create multi-channel image of chm, NDVI, and Red-Edge, save in annotated_image.
    multi_img = np.stack([chm_img, ndvi_img, edge_img], axis=-1).astype('float32')
    annotated_image['multi'] = multi_img

    # If Prompt Boxes have already been generated (and self.prompt_path is not None), load prompt boxes
    # and save in annotated_image.
    if self.prompt_path:
      prompt = np.load(prompt_path)
      annotated_image['prompt'] = prompt

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


class NEONTrainDataset(torch.utils.data.Dataset):

  def __init__(self, pickle_path):
    '''
    params
      pickle_path (str): Path to saved pickle file containing cropped training images and annotations
      
    This class accesses the same maps as NEONTreeDataset, but saved in a single pickle file rather than
    multiple files for "rgb", "multi", "chm", etc.
    '''
    with open(pickle_path, 'rb') as f:
      self.images = pickle.load(f)

  def __len__(self):
    return len(self.images['rgb'])

  def __getitem__(self, idx):
    '''
    returns
    annotated_image (dict) with keys:
      rgb_img: HxWxC ndarray of RGB channels
      multi_img: HxWxC ndarray of multi channels
      prompt (if saved in pickle): Nx4 ndarray of prompt bounding boxes in XYXY format
      annotation: Nx4 ndarray of ground truth bounding boxes in XYXY format
    '''
    annotated_img = {key : self.images[key][idx] for key in self.images.keys()}
    return annotated_img


class VectorDataset(torch.utils.data.Dataset):
  '''
  Dataset of preprocessed vectors from SAM and GroundingDINO. Depending on parameters, will include vectors
  of RGB images, Multi images, and Prompt Boxes. It draws filenames from the annotations folder, and assumes
  these basenames are used identically in the image folders and prompt folders.
  '''
  def __init__(self, image_path, prompt_path, ann_path, rgb=True, multi=True, prompt=True):
    self.rgb = rgb
    self.multi = multi
    self.prompt = prompt
    self.image_path = image_path
    self.prompt_path = prompt_path
    self.ann_path = ann_path

    paths = os.listdir(os.path.join(ann_path, 'Labels'))
    self.basenames = [path.split('.')[0] for path in paths]

  def __len__(self):
    return len(self.basenames)

  def __getitem__(self, idx):
    basename = self.basenames[idx]
    vectors = {'basename': basename}
    if self.rgb:
      vectors['rgb'] = torch.load(os.path.join(self.image_path, 'RGB', f'{basename}.pt'))
      embed_size = tuple(vectors['rgb'].shape[-2:])
    if self.multi:
      vectors['multi'] = torch.load(os.path.join(self.image_path, 'Multi', f'{basename}.pt'))
      embed_size = tuple(vectors['multi'].shape[-2:])
    if self.prompt:
      vectors['prompt'] = {'sparse':torch.load(os.path.join(self.prompt_path, 'Sparse Embeddings', f'{basename}.pt')),
                           'dense': torch.load(os.path.join(self.prompt_path, 'Dense Embeddings', f'{basename}.pt')),
                           'position':torch.load(os.path.join(self.prompt_path, 'Positional Embeddings', f'{embed_size}.pt'))}
    
    vectors['annotation'] = {'boxes':torch.load(os.path.join(self.ann_path, 'Boxes', f'{basename}.pt')), 
                             'labels':torch.load(os.path.join(self.ann_path, 'Labels', f'{basename}.pt'))}

    return vectors

  def get_image(self, basename, return_index=False):
    index = self.basenames.index(basename)
    if return_index:
      return index
    else:
      return self.__getitem__(index)
    


class FastDataset(torch.utils.data.Dataset):
  '''
  Same as VectorDataset, but all vectors are saved together in single pickle files, so loading is faster.
  '''
  def __init__(self, pickle_path):
    paths = os.listdir(pickle_path)
    self.path = pickle_path
    self.basenames = [path.split('.')[0] for path in paths]

  def __len__(self):
    return len(self.basenames)
  
  def __getitem__(self, idx):
    basename = self.basenames[idx]
    with open(os.path.join(self.path, f'{basename}.pickle'), 'rb') as f:
      vectors = pickle.load(f)
    return vectors