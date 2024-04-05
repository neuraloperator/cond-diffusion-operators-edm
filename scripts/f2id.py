"""Credit to Md Ashiqur Rahman for giving me this code."""

import torch
from torch.nn.functional import interpolate
from scipy import linalg
import numpy as np

def calculated_f2id(features1, features2, resolution=50, mode='linear'):
    '''
    features1, features2: discretized feature functions of real and generated data points.
                         assumed to be 1D of shape (batch, 1, grid_size)
    resolution: Required to put both feature1 and feature2 function to same grid

    '''
    if features1.shape[1]!=resolution:
        features1 = interpolate(features1, size=resolution, mode=mode)
    if features2.shape[1]!=resolution:
        features2 = interpolate(features2, size=resolution, mode=mode)
    
    features1 = features1.reshape(features1.shape[0], -1)
    features2 = features2.reshape(features2.shape[0], -1)

    #print(features1.shape, features2.shape)

    mu1 = torch.mean(features1, dim=0).cpu().detach().numpy()
    mu2 = torch.mean(features2, dim=0).cpu().detach().numpy()
    sigma1 = torch.cov(torch.transpose(features1, 0, -1)).cpu().detach().numpy()
    sigma2 = torch.cov(torch.transpose(features2, 0, -1)).cpu().detach().numpy()
    #print(sigma1.shape, sigma2.shape)
    diff = mu1 - mu2
    
    s = 1/resolution

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3/s):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (s*diff.dot(diff) + s*np.trace(sigma1) +
            s*np.trace(sigma2) - 2 *s* tr_covmean)
    

k1 = torch.randn(1000,1,50)
k2 = torch.randn(1000,1,100)

print(calculated_f2id(k1,k2))