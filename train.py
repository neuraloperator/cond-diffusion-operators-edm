"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import jstyleson as json
import glob
import click
import torch
import dnnlib
import pickle
from torch_utils import distributed as dist
from training import training_loop

from omegaconf import OmegaConf as OC
import dataclasses
from dataclasses import asdict, dataclass, field
from typing import List, Union, Tuple, Dict

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@dataclass
class Arguments:
    
    # Main options.
    #outdir: str                    # we use `savedir` in out code, see main()

    dataset_class: str                  # the class name of the dataset, e.g. training.dataset_zarr.CwaEraDataset
    resolution: int                     # spatial resolution
    
    dataset_path: Union[str,None] = None            # path to the data location
    dataset_kwargs: dict = field(default_factory=lambda: {})
    window_size: int = 0                # window size for y
    
    #cond: bool = False                 # type=bool, default=False, show_default=True)
    arch: str = 'ddpmpp'                # click.Choice(['ddpmpp', 'ncsnpp', 'adm'])
    precond: str = 'edm'                # click.Choice(['vp', 've', 'edm'])
    
    # Hyperparameters.
    duration: int = 200
    batch: int = 512                    # Total batch size
    batch_gpu: Union[None, int] = None  # limit batch size per gpu

    # Architecture-related.
    cbase: int = 64
    cres: List[int] = field(default_factory=lambda: [1, 2, 4, 4]) # my defaults
    attn: List[int] = field(default_factory=lambda: [16])         # What spatial resolution to perform self-attn
    num_blocks: int = 4                 # Number of residual blocks per resolution
    rank: float = 1.0                   # Rank for factorisation of weight matrices
    fmult: float = 1.0                  # Retain this *100% of max number of Fourier modes

    # Noise-related.
    rbf_scale: float = 0.05             # For GRF noise, how noisy do we want it? (Less is more noisy.)

    # Training / optimiser-related.
    lr: float = 10e-4                   # type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
    eps: float = 1e-8                   # epsilon for ADAM
    ema: float = 0.5                    # type=click.FloatRange(min=0), default=0.5, show_default=True)
    dropout: float = 0.13               # type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
    xflip: bool = False                 # type=bool, default=False, show_default=True)

    # Performance-related.
    fp16: bool = False                  # type=bool, default=False, show_default=True)
    ls: float = 1.0                     # type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
    bench: bool = True                  # type=bool, default=True, show_default=True)
    cache: bool = True                  # type=bool, default=True, show_default=True)
    workers: int = 1                    # type=click.IntRange(min=1), default=1, show_default=True)

    # I/O related.
    tick: int = 50                      # type=click.IntRange(min=1), default=50, show_default=True)
    snap: int = 50                      # type=click.IntRange(min=1), default=50, show_default=True)
    dump: int = 500                     # type=click.IntRange(min=1), default=500, show_default=True)
    seed: int = 0                       # seed

    # CHRIS modification: if it is non-str and `True`, find the latest
    # checkpoint snapshot in the folder and load that in.
    resume: Union[str, bool] = True # by default, we will resume the experiment

    dry_run: bool = False

def main(kwargs, outdir):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """

    # Convert back into a regular dictionary.
    kwargs = dataclasses.asdict(kwargs)
    opts = dnnlib.EasyDict(kwargs)
    
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    dist.print0("window size: {}".format(opts.window_size))

    if opts.dataset_path is None:
        print("args.path is `None` so setting to `$DATA_DIR`...")
        if 'DATA_DIR' not in os.environ:
            raise ValueError("DATA_DIR not set, please source env.sh")
        else:
            opts.dataset_path = os.environ['DATA_DIR']
    
    c.dataset_kwargs = dnnlib.EasyDict(
        #class_name='training.dataset_zarr.CwaEraDataset',
        class_name=opts.dataset_class,
        path=opts.dataset_path,
        resolution=opts.resolution,
        train=True,
        **opts.dataset_kwargs
    )
    c.window_size = opts.window_size
    
    c.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True, 
        num_workers=opts.workers, 
        prefetch_factor=2       # what is this?
    )
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.sampler_kwargs = dnnlib.EasyDict(
        class_name="training.noise_samplers.RBFKernel",
        Ln1=opts.resolution,
        Ln2=opts.resolution,
        scale=opts.rbf_scale
    )
    
    c.optimizer_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam',
        lr=opts.lr, 
        betas=[0.9,0.999], 
        eps=opts.eps
    )

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(
            **c.dataset_kwargs
        )
        # Chris B: omit these check, we assume condtional training
        #c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        #c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        #if opts.cond and not dataset_obj.has_labels:
        #    raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':              # by default we use this
        c.network_kwargs.update(
            model_type='SongUNO',          # was SongUNet previously, now UNO
            embedding_type='positional', 
            encoder_type='standard', 
            decoder_type='standard',
        )
        c.network_kwargs.update(
            channel_mult_noise=1, 
            resample_filter=[1,1], 
            model_channels=128, 
            channel_mult=[2,2,2]
        )
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(
            model_type='SongUNO', 
            embedding_type='fourier', 
            encoder_type='residual', 
            decoder_type='standard'
        )
        c.network_kwargs.update(
            channel_mult_noise=2, 
            resample_filter=[1,3,3,1], 
            model_channels=128, 
            channel_mult=[2,2,2]
        )
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    PRECOND_VALUES = ['vp', 've', 'edm', 'recon']
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    elif opts.precond == 'edm':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'
    elif opts.precond == 'recon':
        # This is only used to train a deterministic autoencoder.
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.ReconLoss'
    else:
        raise ValueError("precond must be one of: {}".format(PRECOND_VALUES))

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.attn is not None:
        c.network_kwargs.attn_resolutions = opts.attn
    c.network_kwargs.update(
        num_blocks=opts.num_blocks,
        rank=opts.rank,
        fmult=opts.fmult,
        dropout=opts.dropout, 
        use_fp16=opts.fp16
    )
    if opts.precond == 'recon':
        # Do not use skip connections if we're training this
        # as an autoencoder.
        c.network_kwargs.update(disable_skip=True)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    # CHRIS B: not sure I need this feature so I'll comment it out
    """
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    """
    if opts.resume is not None:
        if type(opts.resume) is str:
            raise NotImplementedError()
        else:
            # Find all the network snapshot files
            snapshots = sorted(
                glob.glob("{}/network-snapshot.pkl".format(outdir))
            )
            if len(snapshots) != 0:
                latest_snapshot = snapshots[-1]
                dist.print0("Found snapshot: {} ...".format(latest_snapshot))
                c.resume_pkl = latest_snapshot
                # HACK: we actually have to open the pkl here to
                # get the epoch number.
                with dnnlib.util.open_url(c.resume_pkl, verbose=(dist.get_rank() == 0)) as f:
                    c.resume_kimg = pickle.load(f)['cur_nimg'] // 1000

        #c.resume_state_dump = opts.resume
        c.resume_state_dump = None          # keep things simple for now

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    c.run_dir = outdir

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    #dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

import argparse
import logger

def parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('--datadir', type=str, default="")
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument(
        "--override_cfg",
        action="store_true",
        help="If this is set, then if there already exists a config.json "
        + "in the directory defined by savedir, load that instead of args.cfg. "
        + "This should be set so that SLURM does the right thing if the job is restarted.",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    saved_cfg_file = os.path.join(args.savedir, "config.json")
    if os.path.exists(saved_cfg_file) and not args.override_cfg:
        cfg_file = json.loads(open(saved_cfg_file, "r").read())
        logger.debug("Found config in exp dir, loading instead...")
    else:
        cfg_file = json.loads(open(args.cfg, "r").read())
    
    # structured() allows type checking
    conf = OC.structured(Arguments(**cfg_file))

    # Since type checking is already done, convert
    # it back ito a (dot-accessible) dictionary.
    # (OC.to_object() returns back an Arguments object)
    main(OC.to_object(conf), args.savedir)

#----------------------------------------------------------------------------
