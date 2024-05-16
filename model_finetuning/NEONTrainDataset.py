import torch
import pickle

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
    
    annotated_img = {key : self.images[key][idx] for key in ["rgb", "multi", "annotation", "basename"]}
    return annotated_img
