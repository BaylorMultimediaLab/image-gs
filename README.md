# Image-GS

Image-GS represents an image with 2D Gaussians that are optimized to reconstruct it. This version adds quantization of the Gaussian
parameters, entropy-coded bitstreams (LZMA or rANS), pruning, and time-budgeted encoding.

## Setup

Requires Python 3.13 and the CUDA 12.8 toolkit (`nvcc`) to compile the CUDA extensions.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# CUDA extensions, compiled for the compute capability of your GPU (e.g., 8.0 for an A100, 8.6 for an RTX 30 series)
export TORCH_CUDA_ARCH_LIST="8.6"
pip install ./gsplat --no-build-isolation
pip install git+https://github.com/rahul-goel/fused-ssim/@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38 --no-build-isolation
```

## Running

```bash
python main.py --input_path uvg/480p/Beauty_0001.png --num_gaussians 2000
```

Every option has a default in `cfgs/default.yaml` and can be overridden on the command line as `--<key> <value>`. Boolean options are
flags that switch them on (they all default to off). Input images are read from `data_root` (default `media/`), and `input_path` is an
image file or a directory with a texture stack. Each run writes its log, config, checkpoints, renders, and bitstream to
`results/<exp_name>/<folder named after the main settings>/`. To render a trained model, rerun with the same options plus
`--eval --render_height <height>`.

## Configurations

| What | Options |
|---|---|
| Rate | `--num_gaussians`; `--prune_ratio` prunes that fraction of them after densification (`--prune_score loo` or `contribution`) |
| Quantization | `--quant_mode step` (fixed step sizes: `--pos_step`, `--scale_step`, `--rot_step`, `--feat_step`), `minmax` (bit depths over each attribute's range: `--pos_bits`, `--scale_bits`, `--rot_bits`, `--feat_bits`), or `none` |
| Bitstream | `--bitstream_codec rans`, `lzma`, or `none`; `--color_transform subtract-green` (codes R-G, G, B-G) or `none` |
| Color space | `--color_space rgb` or `yuv` (with `--y_step`/`--uv_step` or `--y_bits`/`--uv_bits`; requires `--color_transform none`) |
| Schedule | By default, progressive optimization adds Gaussians in rounds and the learning rate decays when PSNR stops improving (up to `--max_steps`). `--max_time <seconds>` optimizes for a fixed time instead, pruning and decaying the learning rate at `--prune_time_ratio` and `--decay_time_ratio` of the budget. `--disable_prog_optim` starts with all Gaussians. |
| Gaussians | `--max_scale <pixels>` caps the Gaussian size; `--init_mode gradient`, `saliency`, or `random` |

Examples:

```bash
# Fixed bit depths, LZMA bitstream
python main.py --quant_mode minmax --pos_bits 16 --scale_bits 8 --rot_bits 8 --feat_bits 10 --bitstream_codec lzma

# Encoding within a 40-second optimization budget
python main.py --disable_prog_optim --max_time 40 --max_steps 1000000

# Unquantized float parameters (a bitstream requires quantization)
python main.py --quant_mode none --bitstream_codec none
```

## Sweeps on a Slurm cluster

`slurm/setup.slurm` builds the environment once, and `slurm/submit_lowres.sh [options]` submits the 480p/720p sweep as a job array with
one run per task, after the setup job. Cluster-specific settings (account, partition, modules) are in the `#SBATCH` headers and
`slurm/env.sh`. `python slurm/lowres_sweep.py --help` lists the sweep options; without `--task`, it runs the whole sweep locally.
