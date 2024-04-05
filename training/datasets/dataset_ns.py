import numpy as np
import os
from .dataset import Dataset
import torch

from torch.nn import functional as F

class NSDataset(Dataset):
    """
    https://zenodo.org/records/7495555
    
    2D Navier-Stokes data (200 trajectories, 500 time-steps each) at 64 x 64 spatial resolution.

    This dataset is a drop-in replacement for the ERA-CWA dataset which is not yet
    public. It is not intended to be a difficult dataset to train on, rather its
    purpose is to simply provide something which can be trained on as a proof of
    concept.

    We wish to train a conditional diffusion model `p(y_t|u_{t-k}, ..., u_{t+k})`, where
    `y_t` is the original resolution of NS at timestep `t` and `u` is a much smaller
    resolution, i.e. we want to do (function space) super resolution.

    `k` is handled by `WindowedDataset` and we need not worry about that here, we
    simply just need to write the class to give us `y_t` and `u_t`, and we can generate
    `u_t` by downsampling `y_t` and upsampling it back to the same resolution as `y_t`.
    """
    
    def __init__(self, 
                 path, 
                 resolution, 
                 lowres_scale_factor=0.25,   #
                 train=True                  #
        ):
        
        # The dataset contains multiple trajectories, we only want one of them
        # since this dataset is intended to represent a single time series.

        if train:
            which_trajectory = 0
        else:
            which_trajectory = 1

        # u is the actual function we wish to learn, conditioned on low res
        # versions of the function y
        u = np.load(os.path.join(path, "2D_NS_Re40.npy"))[which_trajectory][1:]
        u = torch.from_numpy(u).float().unsqueeze(1)
        self.u = F.interpolate(u, size=resolution, mode='bilinear')
        
        # Downscale by a factor of `lowres_scale_factor` (e.g. 0.25 => 1/4)
        # then upscale back. y is meant to be a low-res version of u.
        y = F.interpolate(self.u, scale_factor=lowres_scale_factor)
        self.y = F.interpolate(y, size=resolution, mode='bilinear')

        self.u, self.y = self.normalise(
            self.u, self.y, gain=1.0,
            check_valid=True
        )
    
        assert self.u.shape == self.u.shape
        self.res = resolution

    @property
    def resolution(self):
        return self.res
        
    @property
    def name(self):
        return "NSDataset"

    @property
    def num_channels(self):
        """How many channels are in u?"""
        return 1

    @property
    def y_dim(self):
        """How many channels are in y?"""
        return 1

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idc):
        return self.u[idc], self.y[idc]