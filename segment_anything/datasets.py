import os
import rasterio
import torch
import numpy as np
import xml.etree.ElementTree as ET

class NEONTreeDataset(torch.utils.data.Dataset):
  def __init__(self, data_path, ann_path, prompt_path=None, check_values=False):
    '''
    params
      data_path (str): Path to top-level training image directory (contains RGB, LiDAR, Hyperspectral, CHM)
      ann_path (str):  Path to annotations

    __init__ stores filenames from RGB folder, which will be used to retrieve relevant files from other
    folders. Because RGB folder has four files not found in Hyperspectral or CHM, we remove any entries
    not found in these other folders.
    '''
    self.check_values = check_values
    self.data_path = data_path
    self.ann_path = ann_path
    self.prompt_path = prompt_path
    file_names = os.listdir(os.path.join(data_path, 'RGB'))
    problem_files = set(['2019_SJER_4_251000_4103000_image', '2019_TOOL_3_403000_7617000_image', 'TALL_043_2019', 'SJER_062_2018'])
    basenames = [name.split('.')[0] for name in file_names]
    basenames = [name for name in basenames if name not in problem_files]
    basenames = [name for name in basenames if os.path.exists(os.path.join(data_path, 'Hyperspectral', f'{name}_hyperspectral.tif'))]
    basenames = [name for name in basenames if os.path.exists(os.path.join(ann_path,f'{name}.xml'))]
    basenames = [name for name in basenames if os.path.exists(os.path.join(data_path, 'CHM', f'{name}_CHM.tif'))]
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
    rgb_path = os.path.join(self.data_path, 'RGB', f'{basename}.tif')
    chm_path = os.path.join(self.data_path, 'CHM', f'{basename}_CHM.tif')
    hs_path = os.path.join(self.data_path, 'Hyperspectral', f'{basename}_hyperspectral.tif')
    ann_path = os.path.join(self.ann_path, f"{basename}.xml")
    if self.prompt_path:
      prompt_path = os.path.join(self.prompt_path, 'Boxes', f"{basename}.npy")
    annotated_image = dict()

    # Open RGB path and two paths used to construct multi image. Save rgb_img in annotated_image.
    with rasterio.open(rgb_path) as img:
      rgb_img = img.read().transpose(1,2,0)
    with rasterio.open(hs_path) as img:
      hs_img = img.read()
    with rasterio.open(chm_path) as img:
      chm_img = img.read()
    annotated_image['rgb_img'] = rgb_img

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
    annotated_image['multi_img'] = multi_img

    # If Prompt Boxes have already been generated (and self.prompt_path is not None), load prompt boxes
    # and save in annotated_image.
    if self.prompt_path:
      prompt = np.load(os.path.join(prompt_path, f'{basename}.npy'))
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