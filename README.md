# Image-GS

[Image-GS](https://github.com/NYU-ICL/image-gs) represents an image with 2D Gaussians that are optimized to reconstruct it. This version adds quantization of the Gaussian
parameters, entropy-coded bitstreams (rANS), pruning, and time-budgeted encoding.

## Setup

Requires Python 3.13 and the CUDA 12.8 toolkit (`nvcc`) to compile the CUDA extensions.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# CUDA extensions, compiled for the compute capability of your GPU (e.g., 8.0 for an A100, 8.6 for an RTX 30 series)
export TORCH_CUDA_ARCH_LIST="8.6"
pip install ./gsplat --no-build-isolation
pip install git+https://github.com/rahul-goel/fused-ssim/@77aba120314babc98140cd6c7e8c7119fa92cbee --no-build-isolation
# C++ rANS coder
pip install ./compress/csrc --no-build-isolation
```

## Running with the default configuration

```bash
# default with all logging enabled, no fixed time budget
python main.py --input_path uvg/480p/Beauty_0001.png --qp 32
# real-time mode with a fixed time budget
python main.py --input_path uvg/480p/Beauty_0001.png --realtime --cuda_graph --max_time 4 --qp 32
```

By default, most compression-related options are guided by an HEVC-like QP, passed via the command line as `--qp <value>` (default: 32). For time-budgeted experiments, enable real-time mode and cuda graph capture (see [Speed Improvements](#speed-improvements)). The default config is the one that provides the highest BD-rate w.r.t. x265.

## Configurations

Every option has a default in `cfgs/default.yaml` and can be overridden on the command line as `--<key> <value>`. Boolean options are
flags: `--<key>` switches one on and `--no-<key>` off. Input images are read from `data_root` (default `media/`), and `input_path` is an
image file or a directory with a texture stack. Each run writes its log, config, checkpoints, renders, and bitstream to
`results/<exp_name>/<folder named after the main settings>/`. To render a trained model, rerun with the same options plus
`--eval --render_height <height>`.

In the tables below, the column *Set by `--qp`* is *yes* for options that `--qp` derives (a given value is ignored), *qp only* for options
that only take effect with `--qp`, and *no* for options independent of it.

### Rate Control

`--qp` controls the rate like the quantization parameter of HEVC. The QP is first mapped to `qp_offset + qp_slope × QP`, so that the PSNR
roughly matches x265 all-intra at the same QP with 2–4 s encoding budgets. Three settings are derived at the mapped QP:

- **Number of Gaussians before pruning**: a content-agnostic density (8 per 1000 pixels at QP 32, doubling every 5 QPs down) times
  `qp_overestimate`. [Pruning and densification](#pruning-and-densification) decide how many are kept, so this only needs to be in the
  right range.
- **Pruning λ**: `qp_lambda_scale` times HEVC's λ = 0.57 · 2^((QP − 12) / 3).
- **Feature step**: halves every 15 QPs (see [Quantization](#quantization)).

A rate term in the training loss (`rate_weight`, relative to λ) adds the estimated bits of the features and rotations, so that
optimization also trades distortion against rate. Distortion favors luma (see [Color Space](#color-space)). `--qp 0` turns QP rate control off: `num_gaussians` sets the number of Gaussians, and pruning and steps are set explicitly.

| Option | Values | Default | Set by `--qp` | Description |
|---|---|---|---|---|
| `qp` | ≥ 0 | `32` | – | HEVC-like quantization parameter; `0` disables QP rate control |
| `qp_offset` | float | `6.47` | qp only | Offset of the QP mapping (`0` with `qp_slope 1` derives the settings at the QP itself) |
| `qp_slope` | > 0 | `0.767` | qp only | Slope of the QP mapping |
| `qp_overestimate` | > 0 | `0.6` | qp only | Factor on the content-agnostic number of Gaussians before pruning |
| `qp_lambda_scale` | > 0 | `4.0` | qp only | Factor on HEVC's λ for pruning and the rate term |
| `rate_weight` | ≥ 0 | `0.25` | qp only | Weight of the rate term in the loss, relative to λ; `0` disables it; requires `quant_mode step` |
| `num_gaussians` | int > 0 | `1000` | yes | Number of Gaussians without `--qp` |

### Color Space

By default, the Gaussian features are in YUV space. Metrics and saved images are converted back to RGB. The training
losses, the error of λ pruning, and the error map of densification compare the RGB images of Y, `uv_weight` × U, and `uv_weight` × V: a
luma error counts as in RGB, while a chroma error counts less. This favors luma, which VMAF measures; `uv_weight 1` is the RGB loss.

With `--qp`, the Y and U/V steps are those that give the per-channel error of RGB features at the QP's step, times `y_step_ratio` and
`uv_step_ratio`, so that chroma is quantized 2× coarser by default. The rate term weights the U and V bits by `rate_uv_weight`.

| Option | Values | Default | Set by `--qp` | Description |
|---|---|---|---|---|
| `color_space` | `yuv`, `rgb` | `yuv` | no | Color space of the Gaussian features and losses |
| `uv_weight` | > 0 | `0.25` | no | YUV: weight of chroma in the losses, the pruning error, and the error map of densification; `1` matches RGB |
| `y_step_ratio` | > 0 | `1.0` | qp only | YUV: factor on the Y step that gives the error of RGB features |
| `uv_step_ratio` | > 0 | `2.0` | qp only | YUV: factor on the U and V step that gives the error of RGB features |
| `uv_step_gamma` | float | `-1.0` | no | YUV: `feat_step_gamma` of U and V; negative uses `feat_step_gamma` |
| `rate_uv_weight` | float | `-1.0` | qp only | YUV: weight of the U and V bits in the rate term; negative uses `uv_weight` |

### Initialization

The initial number of gaussians is derived from `--qp` or given by `num_gaussians`. Half of them are placed on the strongest small bright or dark blobs of the target image, isotropic at the blob's size. The others are sampled with a density proportional to the local standard deviation. Each of these is
sized at `init_scale_knn` times the RMS distance to its 3 nearest neighbors, so that sparse Gaussians cover the image from the start, and
elongated along the edges of the target by the coherence of its structure tensor (up to `init_aspect`, at the same area). Features start
at the target color at each Gaussian's position. Gaussians added by [densification](#pruning-and-densification) are shaped the same way.

| Option | Values | Default | Set by `--qp` | Description |
|---|---|---|---|---|
| `init_mode` | `local-std`, `gradient`, `random` | `local-std` | no | Position density of the Gaussians that are not on fine blobs: local luma standard deviation, squared image gradient, or uniform |
| `init_std_gamma` | float | `0.8` | no | `local-std`: exponent of the local standard deviation |
| `init_fine_ratio` | [0, 1) | `0.5` | no | Fraction of the Gaussians placed on fine blobs; `0` disables it |
| `init_random_ratio` | [0, 1] | `0.0` | no | Fraction of the other Gaussians placed uniformly at random instead of by `init_mode` |
| `init_scale_knn` | ≥ 0 | `0.4` | no | Initial size as a factor of the RMS distance to the 3 nearest Gaussians; `0` uses `init_scale` |
| `init_scale` | > 0 | `2.25` | no | Initial size in pixels (with `init_scale_knn 0`) |
| `init_aspect` | ≥ 1 | `4.0` | no | Maximum aspect ratio of initial and added Gaussians along edges; `1` keeps them isotropic |

### Pruning and Densification

Optimization alternates between fitting the Gaussians and changing how many there are. The run is split into rounds, `prune_steps`
steps apart, or, with `max_time`, at every `prune_time_ratio` of the budget (1/6, 2/6, 3/6, and 4/6 by default). With `--qp`
(`prune_mode lambda`), each round:

1. **Prunes** by rate–distortion: every Gaussian is scored by how much the squared error increases without it (leave-one-out, 8-bit
   units, summed over pixels), and removed if that increase is below λ times the average bits per Gaussian. A round removes at most
   `prune_max_ratio` of the Gaussians, as leave-one-out scores ignore that neighbors compensate for each other.
2. **Densifies**, except after the last round: adds `prune_add_ratio` times the remaining number of Gaussians at pixels sampled by the
   error of the current rendering, shaped as at [initialization](#initialization). The next round keeps them only if they pay off.

The number of Gaussians thus shrinks or grows toward the number that λ pays for, from the start number that `--qp` derives. Pruning
ends after `prune_times` rounds, or, with `prune_add_ratio 0`, once a round removes none. Optimization then continues with the kept
Gaussians, and with `max_time`, the learning rate decays once at `decay_time_ratio` of the budget.

Without `--qp`, `prune_mode ratio` removes a fixed fraction of `num_gaussians` with the lowest leave-one-out increase of the L1 loss in `prune_times` equal rounds,
without densification.

| Option | Values | Default | Set by `--qp` | Description |
|---|---|---|---|---|
| `prune_mode` | `lambda`, `ratio` | `ratio` | yes (`lambda`) | Rate–distortion pruning with densification, or a fixed fraction |
| `prune_lambda` | > 0 | `0.0` | yes | `lambda`: squared-error increase per bit below which a Gaussian is removed |
| `prune_max_ratio` | (0, 1) | `0.25` | no | `lambda`: maximum fraction of the Gaussians removed per round |
| `prune_add_ratio` | ≥ 0 | `0.25` | no | `lambda`: Gaussians added after each round but the last, as a fraction of the remaining ones; `0` disables densification |
| `prune_times` | int ≥ 1 | `4` | no | Number of rounds (`ratio`), or maximum number of rounds (`lambda`) |
| `prune_steps` | int | `500` | no | Optimization steps before each round (without `max_time`) |
| `prune_time_ratio` | (0, 1] | `0.1667` | no | With `max_time`: fraction of the budget before each round |
| `prune_min_steps` | int ≥ 0 | `20` | no | With `max_time`: minimum steps before each round, so that a stall does not trigger several rounds in a row |
| `prune_ratio` | [0, 1) | `0.1` | no | `ratio`: fraction of `num_gaussians` removed in total |
| `add_color_init` | `target`, `residual` | `target` | no | Features of added Gaussians: the target, or the error of the rendered image at the pixel |

### Quantization

With `quant_mode step`, every parameter is rounded to a multiple of a fixed step: positions in pixels, scales in octaves, rotations in
degrees, and features per channel. Optimization renders the quantized parameters with straight-through gradients, so that the Gaussians
adapt to the grid. `--qp` sets the feature steps (in YUV mode, see [Color Space](#color-space)); the geometry steps are fixed.

Optimization uses uniform steps. The saved model and bitstream switch to per-Gaussian steps, which decoders derive from the geometry they
have already decoded, so they cost no side information:

- **Features**: coarser for Gaussians that contribute little to the image.
- **Rotation**: up to 4× coarser for rounder Gaussians, whose orientation matters less.
- **Position**: up to 8× coarser along the pixel axis nearer an elongated Gaussian's major axis. Positions are coded in two stages: coarse
  positions first, refinement bits after the scales and rotations.

`quant_mode minmax` instead spreads 2^bits levels over the range of each attribute, and `none` keeps float32 parameters (the steps then
only estimate the bits for λ pruning, and no bitstream is written).

| Option | Values | Default | Set by `--qp` | Description |
|---|---|---|---|---|
| `quant_mode` | `step`, `minmax`, `none` | `step` | no | Quantization mode; `--qp` requires `step` or `none` |
| `pos_step` | > 0 | `0.125` | no | Position step in pixels |
| `scale_step` | > 0 | `0.03125` | no | Scale step in octaves |
| `rot_step` | > 0 | `0.703125` | no | Rotation step in degrees; per-Gaussian rotation steps require it to divide 22.5°, per-Gaussian position steps 90° |
| `feat_step` | > 0 | `0.015625` | yes | Feature step (RGB) |
| `y_step` | > 0 | `0.015625` | yes | Y step (YUV) |
| `uv_step` | > 0 | `0.015625` | yes | U and V step (YUV) |
| `feat_step_adapt` | `weight`, `none` | `weight` | no | Per-Gaussian feature steps by the sum of the squared blending weights, or uniform |
| `feat_step_gamma` | float | `0.25` | no | Strength: steps scale as (proxy / its geometric mean)^−γ, snapped to powers of 2^(1/4) in [1/4, 4] |
| `rot_step_adapt` | `aspect`, `none` | `aspect` | no | Per-Gaussian rotation steps by aspect ratio, or uniform |
| `rot_step_aspect` | float | `3.0` | no | Aspect ratio in octaves from which Gaussians keep `rot_step` |
| `rot_step_gamma` | float | `0.5` | no | Exponent on the rotation sensitivity ratio |
| `pos_step_adapt` | `major`, `none` | `major` | no | Per-Gaussian position steps along the major axis with two-stage positions, or uniform |
| `pos_step_gamma` | (0, 2] | `1.0` | no | Exponent on the position sensitivity ratio |
| `pos_bits`, `scale_bits`, `rot_bits`, `feat_bits`, `y_bits`, `uv_bits` | int | `16` | no | `minmax`: bit depth of each attribute (`feat_bits` for RGB, `y_bits` and `uv_bits` for YUV) |

### Entropy Coding

The bitstream stores the Gaussians in Morton (Z-order) order of their positions, so that positions are coded as small gaps between
consecutive Morton codes and Gaussians next to each other in the stream are also close in the image. The other attributes are coded as
residuals of predictions that the decoder repeats from what it has already decoded:

- **Features**: predicted from the nearest earlier Gaussians: Y (or G) from their mean, U from Y and V from U (or R−G and B−G from G) by a
  local linear fit, as in CCLM.
- **Scales**: the sum of the two log-scales predicted from the distances to the nearest Gaussians, as dense regions hold small Gaussians.
- **Rotations**: stored in [−45°, 45°], with the axes swapped where needed (the same ellipse).

rANS codes each stream with a static histogram of hybrid integer tokens (as in JPEG XL), stored in the header. Except in real-time mode,
the encoder decodes the bitstream and checks that it reproduces the quantized parameters.

| Option | Values | Default | Set by `--qp` | Description |
|---|---|---|---|---|
| `bitstream_codec` | `rans`, `none` | `rans` | no | Entropy coder (`none`: no bitstream); requires quantization |
| `feat_prediction` | bool | `true` | no | Code the features as residuals of their prediction |
| `scale_prediction` | bool | `true` | no | `step`: code the scales as residuals of their prediction |
| `color_transform` | `subtract-green`, `none` | `subtract-green` | no | RGB: code R−G, G, B−G; ignored in YUV |

### Speed Improvements

Encoding is bounded by a time budget (`max_time`) rather than a number of steps; pruning rounds and the single learning-rate decay happen
at fractions of the budget. `--realtime` (default off) skips what an encoder does not need (saved images, extra metrics, periodic evaluation, bitstream
verification), uses fused Adam, charges the budget by wall-clock time without synchronizing every step, and pauses Python's garbage
collection during optimization. At low resolutions on fast GPUs, a step is then limited by the CPU launching about 110 kernels;
`--cuda_graph` (default off) replays each step from a CUDA graph instead, captured again after every pruning or densification round.

| Option | Values | Default | Set by `--qp` | Description |
|---|---|---|---|---|
| `max_time` | ≥ 0 | `0.0` | no | Optimization time budget in seconds; `0` disables it and decays the learning rate when PSNR stops improving |
| `max_steps` | int | `10000` | no | Maximum number of steps; set it high with `max_time`, so that the budget decides |
| `decay_time_ratio` | (0, 1] | `0.8333` | no | With `max_time`: fraction of the budget after which the learning rate decays once by `decay_ratio` |
| `realtime` | bool | `false` | no | Real-time encoding |
| `cuda_graph` | bool | `false` | no | Replay each step from a CUDA graph; requires `realtime`, `quant_mode step` or `none` |
| `max_scale` | ≥ 0 | `0.0` | no | Upper bound of the Gaussian scales in pixels (`0` disables it); large Gaussians slow down rasterization |
