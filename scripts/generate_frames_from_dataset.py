import argparse
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
#from training.dataset_zarr import CwaEraDataset

import dnnlib

def get_figure(u: torch.FloatTensor, y: torch.FloatTensor, idx: int, figsize=(8,4)):
    fig, axs = plt.subplots(1,2, figsize=figsize)
    fig.suptitle("time: {}".format(idx))
    #x = x.transpose(0,1).transpose(1,2)     # convert to TF format
    axs[0].imshow(u)
    axs[0].set_title("u")
    axs[1].imshow(y)
    axs[1].set_title("y")
    return fig

def to_tf_format(x: torch.Tensor):
    assert len(x.shape) == 3
    return x.transpose(0,1).transpose(1,2)

def run(args):

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.path is None:
        print("args.path is `None` so setting to `$DATA_DIR`...")
        if 'DATA_DIR' not in os.environ:
            raise ValueError("DATA_DIR not set, please source env.sh")
        else:
            args.path = os.environ['DATA_DIR']

    dataset_kwargs = json.loads(args.dataset_kwargs)

    dataset_kwargs = dnnlib.EasyDict(
        #class_name='training.dataset_zarr.CwaEraDataset',
        class_name=args.dataset_class,
        path=args.path,
        resolution=args.resolution,
        train=args.split=='train',
        **dataset_kwargs
    )

    ds = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    loader = DataLoader(
        ds,  batch_size=args.batch_size, num_workers=args.num_workers, 
        shuffle=False
    )
    counter = 0
    for b, (xbatch, ybatch) in enumerate(loader):
        # data is in [-1, 1] so rescale first
        xbatch = xbatch*0.5 + 0.5
        ybatch = ybatch*0.5 + 0.5
        for j in range(len(xbatch)):
            fig = get_figure(
                to_tf_format(xbatch[j]), to_tf_format(ybatch[j]), 
                idx=counter
            )
            fig.savefig("{}/{}.png".format(args.outdir, str(counter).zfill(7)))
            plt.close(fig)
            counter += 1
        print("processed: {} frames".format((b+1)*args.batch_size))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--dataset_class", type=str,
                        default="training.datasets.NSDataset")
    parser.add_argument("--dataset_kwargs", type=str,
                        default="{}",
                        help="JSON string for extra args to pass to dataset.")
    parser.add_argument(
        "--path", 
        type=str,
        default=None,
        help="Path to the dataset. Defaults to $DATA_DIR if not set."
    )
    parser.add_argument("--split", type=str, choices=['train', 'test'],
                        default='train')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    run(args)