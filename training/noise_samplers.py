import torch
import torch.fft as fft
import numpy as np
import cv2
import math
import os

def get_fixed_coords(Ln1, Ln2):
    xs = torch.linspace(0, 1, steps=Ln1 + 1)[0:-1]
    ys = torch.linspace(0, 1, steps=Ln2 + 1)[0:-1]
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.cat([yy.reshape(-1, 1), xx.reshape(-1, 1)], dim=-1)
    return coords

class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError()

class RBFKernel(NoiseSampler):
    @torch.no_grad()
    def __init__(
        self, n_in, Ln1, Ln2, scale=1, eps=0.01, device=None
    ):
        self.n_in = n_in
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.scale = scale

        # (s^2, 2)
        meshgrid = get_fixed_coords(self.Ln1, self.Ln2).to(device)
        # (s^2, s^2)
        C = torch.exp(-torch.cdist(meshgrid, meshgrid) / (2 * scale**2))
        # Need to add some regularisation or else the sqrt won't exist
        I = torch.eye(C.size(-1)).to(device)

        # Not memory efficient
        #C = C + (eps**2) * I
        I.mul_(eps**2) # inplace multiply by eps**2
        C.add_(I)      # inplace add by I
        del I          # don't need it anymore

        # TODO: can we support f16 in this class to save gpu memory?
        
        self.L = torch.linalg.cholesky(C)

        del C # save memory

    @torch.no_grad()
    def sample(self, N):
        # (N, s^2, s^2) x (N, s^2, 1) -> (N, s^2, 2)
        # We can do this in one big torch.bmm, but I am concerned about memory
        # so let's just do it iteratively.
        # L_padded = self.L.repeat(N, 1, 1)
        # z_mat = torch.randn((N, self.Ln1*self.Ln2, 2)).to(self.device)
        # sample = torch.bmm(L_padded, z_mat)
        samples = torch.zeros((N, self.Ln1 * self.Ln2, self.n_in)).to(self.device)
        for ix in range(N):
            # (s^2, s^2) * (s^2, 2) -> (s^2, 2)
            this_z = torch.randn(self.Ln1 * self.Ln2, self.n_in).to(self.device)
            samples[ix] = torch.matmul(self.L, this_z)

        # reshape into (N, s, s, n_in)
        sample_rshp = samples.reshape(-1, self.Ln1, self.Ln2, self.n_in)

        # reshape into (N, n_in, s, s)
        sample_rshp = sample_rshp.transpose(-1,-2).transpose(-2,-3)
        
        return sample_rshp