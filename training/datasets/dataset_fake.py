import torch
from .dataset import Dataset

class FakeDataset(Dataset):
    
    def __init__(self, path, resolution, train=True):
        self.res = resolution
        
    @property
    def name(self):
        return "FakeDataset"

    @property
    def num_channels(self):
        return 10

    @property
    def y_dim(self):
        return 2

    def __len__(self):
        return 100

    def __getitem__(self, idc):
        x = torch.randn((self.num_channels, self.res, self.res))
        y = torch.randn((self.y_dim, self.res, self.res)) 
        return x, y

if __name__ == '__main__':
    from .dataset import WindowedDataset
    from torch.utils.data import DataLoader
    
    fake_ds = FakeDataset(None, 128, True)
    w_ds = WindowedDataset(fake_ds, window_size=5)
    print(fake_ds)
    print(w_ds)

    loader = DataLoader(w_ds, batch_size=5)
    print([ elem.shape for elem in iter(loader).next() ])