import json
import sys
import os
import glob
import pickle
import dnnlib

import torch
from torch_utils import distributed as dist

#from train import run, Arguments
from training import training_loop

from omegaconf import OmegaConf as OC

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

if __name__ == '__main__':

    # Have to do this as it was done in train.py as well.
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    exp_dir = sys.argv[1]
    config = dnnlib.EasyDict(json.loads(
        open(os.path.join(exp_dir, "training_options.json"), "r").read() 
    ))
    # Convert any internal dictionaries into EasyDict as well.
    for key in config.keys():
        if type(config[key]) is dict:
            config[key] = dnnlib.EasyDict(config[key])

    snapshots = sorted(
        glob.glob("{}/network-snapshot.pkl".format(exp_dir))
    )
    if len(snapshots) != 0:
        latest_snapshot = snapshots[-1]
        dist.print0("Found checkpoint: {}".format(latest_snapshot))
        config.resume_pkl = latest_snapshot
        # HACK: we actually have to open the pkl here to
        # get the epoch number.
        with dnnlib.util.open_url(config.resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            config.resume_kimg = pickle.load(f)['cur_nimg'] // 1000
            dist.print0("cur_knimg={}".format(config.resume_kimg))
    
    training_loop.training_loop(**config)