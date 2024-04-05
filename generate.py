"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import json
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import misc
from torch.nn.functional import interpolate
from torch_utils import distributed as dist
from torchvision.utils import save_image

from einops import rearrange

from training.datasets.dataset import WindowedDataset

from torchvision.transforms.functional import gaussian_blur

#----------------------------------------------------------------------------
# Deterministic EDM sampler.

def deterministic_edm_sampler(
    net, latents, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        x_hat = x_cur
        t_hat = t_cur

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def deterministic_ablation_sampler(
    net, latents, class_labels=None,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        x_hat = x_cur
        t_hat = t_cur

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

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

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option("--reload_network",           help="If set, do not use network code pickled in checkpoint", is_flag=True)
@click.option("--resolution",              help="Desired resolution of noise (and therefore generated images", type=int, default=None)
@click.option('--outfile',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
# The number of forecasts (x's) we generate per x_t
@click.option('--examples_per_t', metavar='INT',           type=click.IntRange(min=1), default=64, show_default=True)
# The number of timesteps y_t we consider, for t = {1, ..., t_max}.
@click.option('--t_max', help='Number of timesteps (examples) to generate in total', metavar='INT',    type=click.IntRange(min=1), default=2)
# Batch size for generation.
@click.option('--batch_size', help='Batch size for generation', metavar='INT', type=click.IntRange(min=1), default=32)
@click.option('--num_workers', help='Number of workers for data loader', metavar='INT', type=click.IntRange(min=0), default=0)
#@click.option('--noise_kwargs', type=str, default="{}")
@click.option('--rbf_scale',               help="RBF scale",                metavar='INT',                          type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True), default=0.0002)
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, 
         reload_network,
         resolution,
         outfile, 
         subdirs, 
         examples_per_t,
         t_max,
         batch_size, 
         num_workers,
         #noise_kwargs, 
         device=torch.device('cuda'), 
         **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """

    if (t_max*examples_per_t) % batch_size != 0:
        raise ValueError("t_max * examples_per_t must be evenly divisible by batch_size!" + \
            " values are {} * {}, batch_size = {}".format(t_max,examples_per_t,batch_size))
    
    dist.init()

    # Load dataset because we need to be able to sample y's to condition on.
    exp_dir = os.path.dirname(network_pkl)
    config = dnnlib.EasyDict(json.loads(
        open(os.path.join(exp_dir, "training_options.json"), "r").read() 
    ))
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**config.dataset_kwargs) # subclass of training.dataset.Dataset
    dist.print0('Windowing dataset...')
    dataset_obj = WindowedDataset(dataset_obj, window_size=config.window_size)

    # Load network.
    if reload_network:
        # If this is set, do NOT load the network code from the pickle. Reconstruct
        # the network from the actual current code and only load in the weights.
        # This should be set if you've made post-hoc changes to the network code
        # but are loading in weights corresponding to an older version.
        dist.print0('Constructing network...')
        interface_kwargs = dict(
            img_resolution=dataset_obj.resolution, 
            img_channels=dataset_obj.num_channels, 
            label_dim=dataset_obj.label_dim
        )
        net = dnnlib.util.construct_class_by_name(**config.network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
        net.eval().requires_grad_(False).to(device)
        dist.print0(f'Loading network, load weights from inside "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            net_weights = pickle.load(f)['ema'].to(device).state_dict()
        net.load_state_dict(net_weights)
    else:
        dist.print0(f'Loading network from inside "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)

    dist.print0("Sampler kwargs: {}".format(sampler_kwargs))
    
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj, 
        rank=dist.get_rank(), 
        num_replicas=dist.get_world_size(), 
        shuffle=False,
        seed=0 # TODO make it an arg
    )
    data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True, 
        num_workers=num_workers,
        prefetch_factor=2       # what is this?
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj, 
            sampler=dataset_sampler, 
            batch_size=t_max,  # only return `t_max` images
            **data_loader_kwargs
        )
    )

    #noise_kwargs = json.loads(noise_kwargs)
    dist.print0("Loading noise sampler...")
    noise_sampler_kwargs = dnnlib.EasyDict(config.sampler_kwargs)
    noise_sampler_kwargs.n_in = dataset_obj.num_channels
    noise_sampler_kwargs.device = device
    if resolution is not None:
        noise_sampler_kwargs.Ln1 = resolution 
        noise_sampler_kwargs.Ln2 = resolution
    # We can override arguments in the noise_sampler at generation time,
    # for instance if we want to increase the resolution or change the
    # smoothness of the noise.
    """
    if len(noise_kwargs.keys()) > 0:
        for key in noise_kwargs.keys():
            if key in noise_sampler_kwargs:
                noise_sampler_kwargs[key] = noise_kwargs[key]
                dist.print0(f'  noise_sampler: override {key}={noise_kwargs[key]} ...')
            else:
                raise ValueError(f'Unknown key for noise_sampler: "{key}"')
    """
                
    noise_sampler = dnnlib.util.construct_class_by_name(**noise_sampler_kwargs)
    
    # Pick latents and labels.
    #rnd = StackedRandomGenerator(device, np.arange(0, examples_per_t).tolist())

    # shape: (t_max, nc, h, w) and (t_max, w, nc, h, w)
    images_real_, class_labels_ = next(dataset_iterator)
    # t = timestep, ws = window size, nc = num channels
    class_labels = rearrange(class_labels_, 't ws nc h w -> t (ws nc) h w')
    class_labels = rearrange(class_labels, 't N h w -> t 1 N h w').\
        repeat(1, examples_per_t, 1, 1, 1)
    class_labels = rearrange(class_labels, 't rep N h w -> (t rep) N h w')
    #images_real = images_real.view(-1, *tuple(images_real.shape[2:]))    
    
    class_labels = class_labels.to(device)

    # batch_size = the number of conditioning images

    # TODO parallelise this
    buf_samples = []
    N_total = class_labels.size(0)
    n_iters = int(np.ceil(N_total / batch_size))
    for j in range(n_iters):
        dist.print0("Processing batch: {} / {} ...".format(j+1, n_iters))
        this_slice = slice(j*batch_size, (j+1)*batch_size)
        this_class_labels = class_labels[this_slice]
        
        this_latents = noise_sampler.sample(this_class_labels.size(0)).to(device)
        if this_class_labels.size(-1) != this_latents.size(-1):
            # If we're doing super-resolution
            dist.print0(f'  `this_class_label` and `latents` spatial dim mismatch: {this_class_labels.size(-1)} and {this_latents.size(-1)}, upscaling `this_class_label`...')
            this_class_labels = interpolate(
                this_class_labels, 
                (this_latents.size(-2), this_latents.size(-1)),
                mode='bilinear'
            )

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = deterministic_ablation_sampler if have_ablation_kwargs else deterministic_edm_sampler
        samples = sampler_fn(net, this_latents, this_class_labels, **sampler_kwargs)

        print("  samples min={}, max={}".format(samples.min(), samples.max()))

        samples_torch = ((samples*0.5 + 0.5)).cpu()
        buf_samples.append(samples_torch)

    # shape = (t_max*examples_per_t, ch_x, h, w)
    buf_samples = torch.cat(buf_samples, dim=0)
    # shape = (t_max, examples_per_t, ch_x, h, w)
    buf_samples = buf_samples.reshape(
        t_max, examples_per_t, *tuple(buf_samples.shape[1:])
    )
    
    # shape = (t_max, ch_x, h, w)
    images_real_ = (images_real_*0.5 + 0.5)
    # shape = (t_max, ch_y, h, w)
    class_labels_ = (class_labels_*0.5 + 0.5)

    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    dist.print0("Saving to: {}".format(outfile))
    torch.save(
        dict(gen=buf_samples, x=images_real_, y=class_labels_, metadata={}), 
        outfile
    )

    # Done.
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
