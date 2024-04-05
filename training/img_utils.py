import logging
import glob
from types import new_class
import torch
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
import torchvision.transforms.functional as TF
from torch_utils import distributed as dist


def reshape_fields(img, inp_or_tar, crop_size_x, crop_size_y,rnd_x, rnd_y, params, y_roll, train, normalize=True):
    #Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of size ((n_channels*(n_history+1), crop_size_x, crop_size_y)

    if len(np.shape(img)) == 3:
      img = np.expand_dims(img, 0)
      
    if img.shape[3] > 720:
        img = img[:, :, 0:720]         #remove last pixel for era5 data

    #n_history = np.shape(img)[0] - 1   #for era5
    n_history = params.n_history
    
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1] #this will either be N_in_channels or N_out_channels
    channels = params.in_channels if inp_or_tar =='inp' else params.out_channels
    #print('channels', channels)
    
    # dist.print0('normalize', normalize)
    # dist.print0('train', train)

    if normalize and train:
        # So it's loading statistics from an external file??
        mins = np.load(params.min_path)[:, channels]
        maxs = np.load(params.max_path)[:, channels]
        means = np.load(params.global_means_path)[:, channels]
        stds = np.load(params.global_stds_path)[:, channels]
        
    if crop_size_x == None:
        crop_size_x = img_shape_x
    if crop_size_y == None:
        crop_size_y = img_shape_y


    """
    if normalize and train:
        if params.normalization == 'minmax':
          img  -= mins
          img /= (maxs - mins)
        elif params.normalization == 'zscore':
          #print('params.normalization == zscore')
          img -=means
          img /=stds
    """

    if params.roll:
        img = np.roll(img, y_roll, axis = -1)

    # if train and (crop_size_x or crop_size_y):
    #     img = img[:,:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    if (crop_size_x or crop_size_y):
        img = img[:,:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    if inp_or_tar == 'inp':
        img = np.reshape(img, (n_channels*(n_history+1), crop_size_x, crop_size_y))
    elif inp_or_tar == 'tar':
        img = np.reshape(img, (n_channels, crop_size_x, crop_size_y))

    # do min max norm here
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    return torch.as_tensor(img)
          


    

