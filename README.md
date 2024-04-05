# EDM in function spaces (edm-fs)

This is the public code release of Christopher Beckham's internship at NVIDIA in using neural operators for time series modelling of climate data. This repository is based on [EDM](https://github.com/NVlabs/edm/).

## Setup

This repository is based on the EDM codebase and as such will have a similar set of requirements.  First create a conda environment. We will use the `environment.yml` file that is in the root directory of this repository. This is the same as the corresponding environment file in the original EDM repository from Karras et al located [here](https://github.com/NVlabs/edm).

To create an environment called `edm_fs` we do:

```
conda env create -n edm_fs -f environment.yml
```

Some dependencies may be flexible but PyTorch 1.12 is absolutely crucial (see [this issue](https://github.com/NVlabs/edm/issues/18)).

### Dependencies

We also need to install the `neuraloperators` library. For this repo we need to use my fork of it [here](https://github.com/christopher-beckham/neuraloperator). (Whoever builds on top of this repo may find it useful to see what has changed and try to reconsolidate it with the latest version of the library.)

Clone it, switch to the `dev_refactor` branch and install it with the following steps:

```
git clone git@github.com:christopher-beckham/neuraloperator.git
cd neuraloperator
git checkout dev_refactor
pip install -e .
```

Note that you may also need to install other dependencies that are required by `neuraloperators`. Check their `requirements.txt` file. We need to also install:
 
- jstyleson: `pip install jstyleson`
- torchvision: `pip install torchvision==0.13 --no-deps` (0.13 is compatible with PyTorch `1.12`, and you must specify no-deps so that it doesn't force installing PyTorch 2.0)
- ffmpeg: `conda install -c conda-forge ffmpeg`

### Environment variables

Also cd into `exps` and copy `cp env.sh.bak env.sh` and define the following env variables:

- `$SAVE_DIR`: some location where experiments are to be saved.
- `$DATA_DIR`: this should be kept as is, since it points to the raw CWA/ERA dataset zarr file.

### Dataset

Since the climate dataset used for the internship is not (yet) public, we have to use an open source substitute. For this we use a 2D Navier-Stokes dataset which consists of trajectories in time. Concretely, the neural operator diffusion model is trained to model the following conditional distribution  `p(u_t | y_{t-k}, ..., y_{t+k})` where:
- `u` is the function to model (whose samples are at a fixed discretisation which is defined by `resolution` in `training.datasets.NSDataset`, e.g. `128` for 128px on both spatial dimensions);
- `y` are samples from the same function at a much coarser discretisation, and this is determined by `lowres_scale_factor` in `training.datasets.NSDataset`, i.e. if `lowres_scale_factor=0.0625` then this is a 16x reduction in spatial resolution).
- `k` denotes the size of the context window.

Whatever `$DATA_DIR` is defined to, cd into that directory and download the data:

```
wget https://zenodo.org/records/7495555/files/2D_NS_Re40.npy
```

### Visualising dataset

We can visualise the dataset as a video. For example, if we define `y` to be a 16x reduction of the original resolution then we run the following:

```
python -m scripts.generate_frames_from_dataset \
  --outdir=/tmp/dataset \
  --dataset_kwargs '{"lowres_scale_factor": 0.0625}'
cd /tmp/dataset
ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
  -c:v libx264 -pix_fmt yuv420p out.mp4
```

Here is an example output (in gif format):

![dataset viz](media/dataset.gif)

Here, the `y` variable is actually 8x8 px (i.e. `128*0.0625 = 8`) but has been upsampled back up to 128px with bilinear resampling.

## Running experiments

This code assumes you have a Slurm-based environment and that you are either in an interactive job or will be launching a job. If this is not the case then you can still run the below code but you must ensure that `$SLURM_JOB_ID` is defined. For instance, in a non-Slurm environment you can simply set this to be any other unique identifier.

Experiments are launched by going into `exps` and running `main.sh` with the following arguments:

```
bash main.sh <experiment name> <path to json config> <n gpus>
```

`<experiment name>` means that the experiment will be saved to `$SAVE_DIR/<experiment name>/<slurm id>`. Example json config files are in `exps/json` and you can consult the full set of supported arguments in `train.py`. `<n gpus>` specifies how many GPUs to train on.

For running experiments with `sbatch`, write a wrapper script which calls `main.sh` and specifies any required arguments.

### Generation

To generate a trajectory using a pretrained model, we can use the `generate_samples.sh` script in `exps`. This is a convenience wrapper on top of `generate.py` and it is run with the following arguments:

```
bash generate_samples.sh \
  <experiment name> \
  <n diffusion steps> \
  <n traj> \
  <output file> [<extra args>]
```

Respectively, these arguments correspond to:
- The name of the experiment, _relative_ to `$SAVE_DIR`;
- number of diffusion steps to perform (higher is better quality but takes longer);
- length of the trajectory to generate;
- and output file.

Please see `generate.py` for the full list of supported arguments, and see the next section for an example of how to generate with this script.

## Pretrained models

An example pretrained model can be downloaded [here](https://drive.google.com/file/d/1lpH6WVPqjZU1qNCH_2aWejU834mo6Urj/view?usp=drive_link). Download it to `$SAVE_DIR` and untar it via:

```
cd $SAVE_DIR && tar -xvzf test_ns_ws3_ngf64_v2.tar.gz
```

To generate a sample file from this model for 200 diffusion timesteps, cd into `exps` and run:

```
bash generate_samples.sh \
  test_ns_ws3_ngf64_v2/4148123 \
  200 \
  64 \
  samples200.pt
```

This will spit out a file called `samples200.pt` in the same directory. To generate a 30-fps video from these samples, run the following:

```
bash generate_video_from_samples.sh \
  samples200.pt \
  30 \
  samples200.pt.mp4
```

Example video (in gif format):

![generated viz](media/generated.gif)

with the first column denoting ground truth `u_t` (it is the same across each row), middle column denoting the generated function from diffusion `\tilde{u_t}`, and the third column denoting the low-res function `y_t` (again, same for each row).

### Super-resolution

Since this model was trained with samples from `u` being 64px, we can perform 2x super-resolution by passing in `--resolution=128` like so:

```
bash generate_from_samples.sh \
  test_ns_ws3_ngf64_v2/4148123 \
  200 \
  samples200_128.pt \
  --resolution=128 --batch_size=16
```

Example video (in gif format):

![generated viz](media/generated128.gif)

## Bugs / limitations

This section details some things that should be improved or considered by whomever forks this repository.

### Generation

If you make posthoc changes to the model code (e.g. `training.networks.py`) and then want to generate samples you should also add `--reload_network`, e.g

```
bash generate.sh ... --reload_network
```

This will tell the generation script to instead instantiate the model with its network definition as defined in `networks` and then load the weights from the pickle. By default, EDM's training script pickles not just the model weights but also the network code, and this can be frustrating if one wants to make post-hoc changes to the code which are backward compatible with existing pretrained models.

### Training

Neural operators require significantly more parameters than their finite-d counterparts and this issue is also exacerbated when one is training high-res diffusion models. I suggest future works look at latent consistency models, e.g. performing function space diffusion in the latent space of a pretrained autoencoder. Otherwise, the code should be modified to support `float16` training to alleviate the memory burden.

## Credits

Thanks to my co-authors Kamyar Azzizadenesheli, Nikola Kovachki, Jean Kossaifi, Boris Bonev, and Anima Anandkumar. Special thanks to Tero Karras, Morteza Mardani, Noah Brenowitz, and Miika Aittala.
