# Image-GS

[Image-GS](https://github.com/NYU-ICL/image-gs) represents an image with 2D Gaussians that are optimized to reconstruct it. This version adds quantization of the Gaussian
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
# fused-ssim at the commit of its pull request #45, which launches the kernels on PyTorch's current stream (needed by cuda_graph)
pip install git+https://github.com/rahul-goel/fused-ssim/@732c243fc6731ac0ffa25de17a0d7219f90b61c6 --no-build-isolation

# C++ rANS coder (rebuild after changing compress/csrc/rans.cpp; compress/rans_codec.py refuses a build from another source)
pip install ./compress/csrc --no-build-isolation
```

## Running

```bash
python main.py --input_path uvg/480p/Beauty_0001.png --qp 32
```

Every option has a default in `cfgs/default.yaml` and can be overridden on the command line as `--<key> <value>`. Boolean options are
flags: `--<key>` switches one on and `--no-<key>` off. Input images are read from `data_root` (default `media/`), and `input_path` is an
image file or a directory with a texture stack. Each run writes its log, config, checkpoints, renders, and bitstream to
`results/<exp_name>/<folder named after the main settings>/`. To render a trained model, rerun with the same options plus
`--eval --render_height <height>`.

## Configurations

| What | Options |
|---|---|
| Rate | `--qp <QP>` (default 32) sets everything below from an HEVC-like quantization parameter: λ pruning with `--qp_lambda_scale` (default 4) times HEVC's λ at that QP, in up to `--prune_times` (4) rounds of at most `--prune_max_ratio` (25%), from a content-agnostic estimate of the number of Gaussians (`--qp_overestimate`, default 0.6 times the most that paid off on the test images), adding `--prune_add_ratio` (25%) after each round but the last, and a feature step that halves every 15 QP; requires `--quant_mode step`, or `none`. Otherwise (`--qp 0`) `--num_gaussians`; `--prune_ratio` prunes that fraction of them after densification (`--prune_score loo` or `contribution`), or `--prune_mode lambda --prune_lambda <λ>` prunes, in up to `--prune_times` rounds, every Gaussian whose removal increases the squared error (8-bit units, summed over pixels) by less than λ times its bits, at most `--prune_max_ratio` of them per round |
| Quantization | `--quant_mode step` (fixed step sizes: `--pos_step`, `--scale_step`, `--rot_step`, `--feat_step`), `minmax` (bit depths over each attribute's range: `--pos_bits`, `--scale_bits`, `--rot_bits`, `--feat_bits`), or `none`. With `step`, `--feat_step_adapt weight` or `area` gives the saved model per-Gaussian feature steps: coarser for Gaussians that contribute little to the image (by the sum of their squared blending weights, or by their area), with strength `--feat_step_gamma`; optimization keeps the uniform step, and decoders derive the steps from the decoded geometry |
| Bitstream | `--bitstream_codec rans`, `lzma`, or `none`; `--color_transform subtract-green` (codes R-G, G, B-G) or `none` |
| Color space | `--color_space rgb` or `yuv` (with `--y_step`/`--uv_step` or `--y_bits`/`--uv_bits`; requires `--color_transform none`) |
| Schedule | By default, all Gaussians start at once and the learning rate decays when PSNR stops improving (up to `--max_steps`); `--no-disable_prog_optim` adds them in rounds instead (progressive optimization). `--max_time <seconds>` optimizes for a fixed time, pruning and decaying the learning rate at `--prune_time_ratio` and `--decay_time_ratio` of the budget, with at least `--prune_min_steps` steps before each pruning round; rounds still due at the decay (after stalls) are skipped. |
| Gaussians | `--max_scale <pixels>` caps the Gaussian size; `--init_mode gradient`, `saliency`, or `random` |
| Real-time | `--realtime` skips what an encoder does not need (saved images, LPIPS/MS-SSIM/FLIP, periodic evaluation unless the adaptive learning rate schedule needs it, bitstream verification), uses fused Adam, and charges `--max_time` by wall-clock time without synchronizing every step, and pauses Python's garbage collection during the optimization. Writes the bitstream and the final checkpoint with PSNR/SSIM. |
| CUDA graph | `--realtime --cuda_graph` replays each optimization step from a CUDA graph, so that a step costs its GPU time instead of the CPU time of launching about 110 kernels, which limits fast GPUs at low resolutions (e.g., 480p on an A100). The graph is captured again after every pruning or densification round (into one shared memory pool). Requires `--quant_mode step` (or `none`). |

Examples:

```bash
# Fixed bit depths, LZMA bitstream (QP rate control requires step quantization)
python main.py --qp 0 --quant_mode minmax --pos_bits 16 --scale_bits 8 --rot_bits 8 --feat_bits 10 --bitstream_codec lzma

# Encoding within a 40-second optimization budget (pruning rounds at 1/6 to 4/6 of the budget)
python main.py --max_time 40 --max_steps 1000000

# Real-time encoding within a 1-second budget
python main.py --realtime --max_time 1 --max_steps 1000000
# The same with CUDA graphs
python main.py --realtime --cuda_graph --max_time 1 --max_steps 1000000

# A fixed number of Gaussians instead of a QP, 10% of them pruned in one round
python main.py --qp 0 --num_gaussians 2000 --prune_times 1 --max_time 40 --max_steps 1000000

# Unquantized float parameters (a bitstream requires quantization)
python main.py --qp 0 --quant_mode none --bitstream_codec none
# The same with the pruning of QP 32 (which estimates the bits of each Gaussian as if quantized with the steps of QP 32)
python main.py --qp 32 --quant_mode none --bitstream_codec none
```
