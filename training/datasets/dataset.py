import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):

    norm_params_set: bool = False
    
    @property
    def name(self):
        raise NotImplementedError("")

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def y_dim(self):
        raise NotImplementedError

    def normalise(self, x: torch.Tensor, y: torch.Tensor, gain: float, check_valid: bool = False):
        """"Helper function to return normalised version of x and y."""
        self._set_normalisation_parameters(x, y)
        self.gain = gain
        u_normed = self.norm_u(x)
        y_normed = self.norm_y(y)
        if check_valid:
            assert torch.isclose(self.denorm_u(u_normed), x, atol=1e-4).all() 
            assert torch.isclose(self.denorm_y(y_normed), y, atol=1e-4).all()        
        return u_normed, y_normed

    def _set_normalisation_parameters(self, u, y):
        self.min_U = u.min(dim=0, keepdims=True)[0]
        self.max_U = u.max(dim=0, keepdims=True)[0]

        self.min_y = y.min(dim=0, keepdims=True)[0]
        self.max_y = y.max(dim=0, keepdims=True)[0]

    def norm_u(self, U):
        """Convert U into a format amenable to training"""
        U = (U - self.min_U) / (self.max_U - self.min_U + 1e-6)
        U = ((U - 0.5) / 0.5) * self.gain
        return U

    def denorm_u(self, u_normed):
        """Denormalisation for Design Bench test oracle"""
        # u_normed = u_normed*0.5 + 0.5
        u_normed = (u_normed * 0.5 + (0.5*self.gain)) / self.gain
        return u_normed * (self.max_U - self.min_U + 1e-6) + self.min_U

    def norm_y(self, y):
        """Convert y into a format amenable to training"""
        y = (y - self.min_y) / (self.max_y - self.min_y)
        y = ((y-0.5)/0.5)*self.gain
        return y

    def denorm_y(self, y_normed):
        """Denormalisation for Design Bench test oracle"""
        y_normed = (y_normed * 0.5 + (0.5*self.gain)) / self.gain
        return (y_normed * (self.max_y - self.min_y)) + self.min_y

def window(yy: np.ndarray, t: int, k: int):
    """Use this method to validate windowing logic in WindowedDataset."""
    # k is how many elements to the left/right, so the total window size is 2k + 1
    assert t >= 0, "t must be >= 0"
    assert t <= len(yy)-1, "t must be <= len-1"    
    #assert k % 2 != 0, "k must be odd numbered"
    if t - k < 0:
        last_part = yy[0: t+k+1 ]
        num_missing = ((2*k)+1) - len(last_part)
        padding = np.zeros((num_missing,), dtype=yy.dtype)
        return np.concatenate((padding, last_part ), axis=0)
    elif t+k > len(yy)-1:
        #print("Exceeded")
        first_part = yy[t-k :: ]
        num_missing = ((2*k)+1) - len(first_part)
        padding = np.zeros((num_missing,), dtype=yy.dtype)
        return np.concatenate((first_part, padding ), axis=0)
    else:
        pass 
    return yy[t-k : t+k+1]

class WindowedDataset(Dataset):
    """Wraps a dataset to allow sampling a window of inputs instead."""

    def __init__(self, dataset, window_size):
        self.dataset = dataset
        self.window_size = window_size

    @property
    def num_channels(self):
        return self.dataset.num_channels

    @property
    def y_dim(self):
        return self.dataset.y_dim

    @property
    def resolution(self):
        return self.dataset.resolution
    
    @property
    def label_dim(self):
        return (self.window_size*2 + 1)*self.y_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, t):
        """Returns tensors of shape (nc, h, w) and (window_sz, nc, h, w)"""
        k = self.window_size
        u = self.dataset.__getitem__(t)[0]
        if t - k < 0:
            # If the left side of the window runs off past zero
            # valid_part is indexing dataset[0 : t+k+1]
            valid_part = [ self.dataset.__getitem__(idx)[1] for idx in range(0, t+k+1) ]
            # (n_valid, nc, h, w)
            valid_part = torch.stack(valid_part, dim=0)
            # (n_empty, nc, h, w)
            num_missing = ((2*k)+1) - len(valid_part)
            empty_part = torch.zeros_like(valid_part[0:1]).repeat(num_missing, 1, 1, 1)
            # (window_sz, nc, h, w)
            full = torch.cat((empty_part, valid_part), dim=0)
        elif t+k > len(self)-1:
            # If the right side of the window runs off past length of data
            # valid_part is indexing dataset[ t-k :: ]
            valid_part = [ self.dataset.__getitem__(idx)[1] for idx in range(t-k, len(self)) ]
            # (n_valid, nc, h, w)
            valid_part = torch.stack(valid_part, dim=0)
            # (n_empty, nc, h, w)            
            num_missing = ((2*k)+1) - len(valid_part)
            empty_part = torch.zeros_like(valid_part[0:1]).repeat(num_missing, 1, 1, 1)
            # (window_sz, nc, h, w)
            full = torch.cat((valid_part, empty_part), dim=0)
        else:
            full = [ self.dataset.__getitem__(idx)[1] for idx in range(t-k, t+k+1) ]            
            full = torch.stack(full, dim=0)

        return u, full