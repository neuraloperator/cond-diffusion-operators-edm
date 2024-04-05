import argparse
import matplotlib
import matplotlib.pyplot as plt
import torch
import os

from typing import List, Tuple

def get_figure(gen: torch.FloatTensor, 
               u: torch.FloatTensor, 
               y: torch.FloatTensor, 
               idx: int,
               rect: List = [0, 0.03, 1, 0.93],
               figsize: Tuple = (12,4)):
    """
    Args:
      gen: shape (n_seeds, h, w) (batch and channel already indexed into)
      xu: (h, w) (batch and channel already indexed into)
      y: (h, w) (batch and channel already indexed into)
    """
    n_seeds = gen.size(0)
    fig, axs = plt.subplots(n_seeds, 3, figsize=figsize)
    fig.tight_layout(rect=rect)
    axs = axs.flat
    fig.suptitle("t = {}".format(idx))
    #x = x.transpose(0,1).transpose(1,2)     # convert to TF format
    for s in range(n_seeds):
        axs[s*3].imshow(u)
        axs[s*3 + 1].imshow(gen[s])
        axs[s*3 + 2].imshow(y)
        if s == 0:
            axs[s*3].set_title("$\\boldsymbol{u}_t$")
            axs[s*3 + 1].set_title("$\\tilde{\\boldsymbol{u}}_{t}$")
            axs[s*3 + 2].set_title("$\\boldsymbol{y}_t$")
    return fig

def run(args):

    if len(args.figsize) != 2:
        raise ValueError("figsize must be a two-tuple, received: {}".format(args.figsize))

    sample_dict = torch.load(args.samples)
    real = sample_dict['x']
    y = sample_dict['y']
    gen = sample_dict['gen']

    print("real shape   = {} (t, nc, h, w)".format(real.shape))
    print("y shape      = {} (t, ws, nc, h, w)".format(y.shape))
    print("gen shape    = {} (t, n_repeat, nc, h, w)".format(gen.shape))

    if not os.path.exists(args.outdir):
        print("{} does not exist, creating...".format(args.outdir))
        os.makedirs(args.outdir)

    """
    n_seeds = gen.size(1)
    if args.seed > n_seeds-1:
        raise ValueError("Only {} seeds detected in `gen`, yet seed={}".format(
            n_seeds, args.seed
        ))
    """ 
    # index into this seed
    #gen = gen[:, args.seed]

    ch_u, ch_y = args.ch_u, args.ch_y 
    
    counter = 0
    for j in range(len(real)):
        fig = get_figure(
            u=real[j, ch_u], 
            gen=gen[j, :, ch_u], 
            # for y, just index into the midpoint of the window
            y=y[j, (y.size(1)-1)//2, ch_y], 
            idx=counter,
            figsize=args.figsize
        )
        fig.savefig("{}/{}.png".format(args.outdir, str(counter).zfill(7)))
        plt.close(fig)
        counter += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--ch_u", type=int, default=0,
                        help="Which index of the channel axis do we want to viz for x?")
    parser.add_argument("--ch_y", type=int, default=0,
                        help="Which index of the channel axis do we want to viz for y?")
    parser.add_argument("--figsize", nargs="+", type=int, default=[12,8])
    parser.add_argument("--seed", type=int, default=0,
                        help="What seed do we index into for the `gen` tensor?")
    parser.add_argument(
        "--samples", 
        type=str,
        required=True,
        help="Path to the samples.pt file. E.g. <savedir>/<experiment>/<id>/samples/samples.pt"
    )
    args = parser.parse_args()

    run(args)