import gc
import logging
import math
import os
import struct
import sys
import warnings
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_ssim
from lpips import LPIPS
from pytorch_msssim import MS_SSIM
from torchvision.transforms.functional import gaussian_blur

from compress import decode_bitstream, encode_bitstream, estimate_bits
from gsplat import (
    bin_and_sort_gaussians,
    compute_cumulative_intersects,
    project_gaussians_2d_scale_rot,
    rasterize_gaussians_no_tiles,
    rasterize_gaussians_sum,
)
from gsplat.cuda import nd_rasterize_forward_topk_norm
from utils.color_utils import get_color_transforms, get_yuv_channel_indices, transform_images
from utils.flip import LDRFLIPLoss
from utils.image_utils import (
    compute_image_gradients,
    get_grid,
    get_psnr,
    load_images,
    save_error_maps,
    save_image,
    separate_image_channels,
    visualize_added_gaussians,
    visualize_gaussian_footprint,
    visualize_gaussian_position,
)
from utils.misc_utils import clean_dir, get_latest_ckpt_step, save_cfg, set_random_seed
from utils.quantization_utils import dequantize, quantize_minmax, quantize_step
from utils.saliency_utils import get_smap
from utils.stall_utils import StallTracer

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips")

# Settings derived from the quantization parameter (qp), fitted on the 480p/720p UVG equal-time runs (A100, 40/90 s; experiments/qp/fit.py):
# x265's RD slope in the units of prune_lambda is 0.9-1.1x HEVC's lambda at QP 22-37. The number of Gaussians that paid off doubles every
# ~5 QP (4.7-5.6 per image) and varies 2.5x between images; QP_DENSITY is the most for any image. Refining the feature step paid off
# from QP ~44 (1/32 -> 1/64) and below QP ~29 (1/64 -> 1/128), i.e., the optimal step halves every ~15 QP.
QP_DENSITY = 8.0  # Gaussians per 1000 pixels at QP 32
QP_DENSITY_DOUBLING = 5.0  # QPs over which the number of Gaussians doubles
QP_FEAT_STEP = 1.0 / 64.0  # Feature step at QP 37
QP_FEAT_STEP_HALVING = 15.0  # QPs over which the feature step halves
FEAT_STEP_ADAPT_DIVISIONS = 4  # feat_step_adapt multipliers are powers of 2^(1/4) ...
FEAT_STEP_ADAPT_OCTAVES = 2  # ... between 2^-2 and 2^2


class GaussianSplatting2D(nn.Module):
    isect_capacity_ratio = 2.0  # Tile intersection buffers of CUDA graphs, relative to the number at capture time

    def __init__(self, args):
        super(GaussianSplatting2D, self).__init__()
        self.evaluate = args.eval
        set_random_seed(seed=args.seed)
        self.device = args.device
        self.dtype = torch.float32
        self._init_logging(args)
        self._init_target(args)
        self._init_qp(args)
        self._init_color_space(args)
        self._init_bit_precision(args)
        self._init_gaussians(args)
        self._init_loss(args)
        self._init_optimization(args)
        # Initialization
        if self.evaluate:
            self.ckpt_file = args.ckpt_file
            self._load_model()
        else:
            self._init_pos_scale_feat(args)

    def _init_logging(self, args):
        self.realtime = args.realtime
        self.log_dir = args.log_dir
        self.log_level = args.log_level
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        self.train_dir = os.path.join(self.log_dir, "train")
        self.eval_dir = os.path.join(self.log_dir, "eval")
        self.save_image_format = args.save_image_format
        self.save_plot_format = args.save_plot_format
        self.vis_gaussians = args.vis_gaussians
        self.save_image_steps = args.save_image_steps
        self.save_ckpt_steps = args.save_ckpt_steps
        self.eval_steps = args.eval_steps
        if not self.evaluate:
            clean_dir(path=self.log_dir)
            os.makedirs(self.log_dir, exist_ok=False)
            os.makedirs(self.ckpt_dir, exist_ok=False)
            os.makedirs(self.train_dir, exist_ok=False)
        else:
            os.makedirs(self.eval_dir, exist_ok=True)
        self._gen_logger(args)
        if not self.evaluate:
            save_cfg(path=f"{self.log_dir}/cfg_train.yaml", args=args)

    def _gen_logger(self, args):
        log_fname = "log_train"
        if self.evaluate:
            log_fname = "log_eval"
        log_level = getattr(logging, self.log_level, logging.INFO)
        logging.basicConfig(level=log_level)
        self.worklog = logging.getLogger("Image-GS Logger")
        self.worklog.propagate = False
        datefmt = "%Y/%m/%d %H:%M:%S"
        fileHandler = logging.FileHandler(f"{self.log_dir}/{log_fname}.txt", mode="a", encoding="utf8")
        fileHandler.setFormatter(logging.Formatter(fmt="[{asctime}] {message}", datefmt=datefmt, style="{"))
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logging.Formatter(fmt="\x1b[32m[{asctime}] \x1b[0m{message}", datefmt=datefmt, style="{"))
        self.worklog.handlers = [fileHandler, consoleHandler]
        action = "rendering" if self.evaluate else "optimizing"
        target = f"at QP {args.qp:g}" if args.qp > 0.0 else f"{args.num_gaussians:d} Gaussians"
        self.worklog.info(f"Start {action} {target} for '{args.input_path}'")
        self.worklog.info("***********************************************")

    def _init_target(self, args):
        self.gamma = args.gamma
        self.downsample = args.downsample
        if self.downsample:
            self.downsample_ratio = float(args.downsample_ratio)
        self.block_h, self.block_w = 16, 16  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
        self._load_target_images(path=os.path.join(args.data_root, args.input_path))
        if self.downsample:
            self.gt_images_upsampled = self.gt_images
            self.img_h_upsampled, self.img_w_upsampled = self.img_h, self.img_w
            self.tile_bounds_upsampled = self.tile_bounds
            self._load_target_images(path=os.path.join(args.data_root, args.input_path), downsample_ratio=self.downsample_ratio)
            if not self.evaluate and not self.realtime:
                path = f"{self.log_dir}/gt_upsample-{self.downsample_ratio:.1f}_res-{self.img_h_upsampled:d}x{self.img_w_upsampled:d}"
                self._separate_and_save_images(images=self.gt_images_upsampled, channels=self.input_channels, path=path)
        self.num_pixels = self.img_h * self.img_w
        if not self.evaluate and not self.realtime:
            path = f"{self.log_dir}/gt_res-{self.img_h:d}x{self.img_w:d}"
            self._separate_and_save_images(images=self.gt_images, channels=self.input_channels, path=path)

    def _init_qp(self, args):
        """
        Rate settings from the quantization parameter: lambda pruning with HEVC's lambda at this QP, starting from a content-agnostic
        overestimate of the number of Gaussians, and a feature step that halves every QP_FEAT_STEP_HALVING QPs. The other steps are unchanged.
        """
        if args.qp == 0.0:
            return
        if args.qp < 0.0:
            raise ValueError(f"Quantization parameter must be positive (or 0 to disable it) but got: {args.qp}")
        if args.quant_mode not in ["step", "none"]:
            raise ValueError("The quantization parameter requires quant_mode 'step' or 'none' (qp 0 disables it)")
        if args.qp_overestimate <= 0.0 or args.qp_lambda_scale <= 0.0:
            raise ValueError(f"Overestimation and lambda factors must be positive but got: {args.qp_overestimate}, {args.qp_lambda_scale}")
        density = QP_DENSITY * 2.0**((32.0 - args.qp) / QP_DENSITY_DOUBLING)
        args.num_gaussians = max(1, round(args.qp_overestimate * density * self.img_h * self.img_w / 1000.0))
        args.feat_step = args.y_step = args.uv_step = QP_FEAT_STEP * 2.0**((args.qp - 37.0) / QP_FEAT_STEP_HALVING)
        args.prune_mode = "lambda"
        args.prune_lambda = args.qp_lambda_scale * 0.57 * 2.0**((args.qp - 12.0) / 3.0)
        self.worklog.info(f"QP {args.qp:g}: {args.num_gaussians:d} Gaussians before pruning ({args.qp_overestimate:g} x {density:.2f} per 1000 pixels), "
                          f"feature step 1/{1.0/args.feat_step:.1f}, lambda {args.prune_lambda:.2f} ({args.qp_lambda_scale:g} x HEVC's)")
        if not self.evaluate:
            save_cfg(path=f"{self.log_dir}/cfg_train.yaml", args=args)  # Record the derived settings

    def _load_target_images(self, path, downsample_ratio=None):
        self.gt_images, self.input_channels, self.image_fnames, self.bit_depths = load_images(
            load_path=path, downsample_ratio=downsample_ratio, gamma=self.gamma)
        self.gt_images = torch.from_numpy(self.gt_images).to(dtype=self.dtype, device=self.device)
        self.img_h, self.img_w = self.gt_images.shape[1:]
        self.tile_bounds = ((self.img_w + self.block_w - 1) // self.block_w, (self.img_h + self.block_h - 1) // self.block_h, 1)

    def _separate_and_save_images(self, images, channels, path):
        images_sep = separate_image_channels(images=images, input_channels=channels)
        for idx, image in enumerate(images_sep, 1):
            suffix = "" if len(images_sep) == 1 else f"_{idx:d}"
            save_image(image, f"{path}{suffix}.{self.save_image_format}", gamma=self.gamma)

    def _init_color_space(self, args):
        self.color_space = args.color_space
        if self.color_space not in ["rgb", "yuv"]:
            raise ValueError(f"Unsupported color space '{self.color_space}', valid values include 'rgb' and 'yuv'")
        if self.color_space == "yuv":
            self.rgb_to_yuv, self.yuv_to_rgb = get_color_transforms(self.input_channels, dtype=self.dtype, device=self.device)
            self.y_channels, self.uv_channels, self.other_channels = get_yuv_channel_indices(self.input_channels, device=self.device)
        self.feat_dim = sum(self.input_channels)
        # Optimization target in the color space of the Gaussian features
        self.target_images = self._from_rgb(self.gt_images)

    def _from_rgb(self, images):
        return transform_images(images, self.rgb_to_yuv) if self.color_space == "yuv" else images

    def _to_rgb(self, images):
        return transform_images(images, self.yuv_to_rgb) if self.color_space == "yuv" else images

    def _init_bit_precision(self, args):
        self.quant_mode = args.quant_mode
        if self.quant_mode not in ["none", "minmax", "step"]:
            raise ValueError(f"Unsupported quantization mode '{self.quant_mode}', valid values include 'none', 'minmax', and 'step'")
        self.pos_bits = args.pos_bits
        self.scale_bits = args.scale_bits
        self.rot_bits = args.rot_bits
        self.feat_bits = args.feat_bits
        self.y_bits = args.y_bits
        self.uv_bits = args.uv_bits
        all_steps = [args.pos_step, args.scale_step, args.rot_step, args.feat_step, args.y_step, args.uv_step]
        if self.quant_mode != "minmax" and min(all_steps) <= 0.0:
            raise ValueError(f"Quantization steps must be positive but got: {', '.join(f'{step:g}' for step in all_steps)}")
        self.pos_step = args.pos_step  # Pixels
        self.scale_step = args.scale_step  # Octaves (log2 of the scale in pixels)
        self.rot_step = math.radians(args.rot_step)
        self.feat_step = args.feat_step
        self.y_step = args.y_step
        self.uv_step = args.uv_step
        if self.quant_mode != "minmax":
            # Per-channel steps of the position, scale, rotation, and features as GPU tensors, created once: creating them from Python numbers
            # at every optimization step copies them to the GPU and synchronizes. Without quantization ("none"), they only estimate the bits
            # that step quantization would give (lambda pruning)
            feat_steps = torch.empty(self.feat_dim, dtype=self.dtype, device=self.device)
            for channels, _, step in self._get_feat_groups():
                feat_steps[channels] = step
            self.quant_steps = [self.pos_step / torch.tensor([self.img_w, self.img_h], dtype=self.dtype, device=self.device),
                                torch.full((2,), self.scale_step, dtype=self.dtype, device=self.device),
                                torch.full((1,), self.rot_step, dtype=self.dtype, device=self.device), feat_steps]
        self.feat_step_adapt = args.feat_step_adapt
        if self.feat_step_adapt not in ["none", "area", "weight"]:
            raise ValueError(f"Unsupported feature step adaptation '{self.feat_step_adapt}', valid values include 'none', 'area', and 'weight'")
        if self.feat_step_adapt != "none" and self.quant_mode != "step":
            raise ValueError("feat_step_adapt requires quant_mode 'step'")
        self.feat_step_gamma = float(np.float32(args.feat_step_gamma))  # The value that the bitstream stores
        # The per-Gaussian feature steps apply to the saved model and in evaluation; optimization uses the uniform steps
        self.feat_step_adapt_active = self.feat_step_adapt != "none" and self.evaluate
        self.feat_step_multiplier_cache = None
        self.wrap_rot = args.wrap_rot
        self.bitstream_codec = args.bitstream_codec
        if self.bitstream_codec not in ["none", "lzma", "rans"]:
            raise ValueError(f"Unsupported bitstream codec '{self.bitstream_codec}', valid values include 'none', 'lzma', and 'rans'")
        if self.bitstream_codec != "none" and self.quant_mode == "none":
            raise ValueError("Writing a bitstream requires quantization")
        self.color_transform = args.color_transform
        if self.color_transform not in ["none", "subtract-green"]:
            raise ValueError(f"Unsupported color transform '{self.color_transform}', valid values include 'none' and 'subtract-green'")
        if self.color_transform != "none" and self.color_space != "rgb":
            raise ValueError("Color transforms of the bitstream require color_space 'rgb'")

    def _get_channel_refs(self):
        """
        Reference channels of each attribute for the lossless color transform before entropy coding: subtract-green codes each
        RGB image as R-G, G, B-G, and leaves the other channels unchanged.
        """
        feat_refs = []
        for num_channels in self.input_channels:
            base = len(feat_refs)
            if self.color_transform == "subtract-green" and num_channels == 3:
                feat_refs += [base + 1, -1, base + 1]
            else:
                feat_refs += [-1] * num_channels
        return [None, None, None, feat_refs]

    def _get_feat_groups(self):
        """
        (channel indices, bit precision, step) of each separately quantized group of feature channels.
        In YUV mode, single-channel images are not converted and keep using feat_bits and feat_step.
        """
        if self.color_space != "yuv":
            return [(torch.arange(self.feat_dim, device=self.device), self.feat_bits, self.feat_step)]
        groups = [(self.y_channels, self.y_bits, self.y_step), (self.uv_channels, self.uv_bits, self.uv_step),
                  (self.other_channels, self.feat_bits, self.feat_step)]
        return [group for group in groups if len(group[0]) > 0]

    def _canonicalize(self, scale, rot):
        """
        Equivalent Gaussian parameters to quantize: ellipses are symmetric under rotation by pi,
        and rendering only depends on squared scales (scales are quantized in the log domain in "step" mode, and in the rate estimate of "none").
        """
        if self.wrap_rot:
            rot = torch.remainder(rot, math.pi)
        if self.quant_mode != "minmax":
            scale = scale.abs()
        return scale, rot

    def _get_symbols(self, xy, scale, rot, feat):
        """
        Quantization indices [P, C] of the canonicalized Gaussian parameters (scale in pixels), with the per-channel steps and origins [C]
        that dequantize them. Scales are quantized in pixels in "minmax" mode and in log2 pixels in "step" mode. Without quantization ("none"),
        the "step" indices estimate the rate. With per-Gaussian feature steps (feat_step_adapt), the feature steps are the base steps, which
        _get_feat_step_multipliers scales per Gaussian.
        """
        with torch.no_grad():
            if self.quant_mode != "minmax":
                if not self.feat_step_adapt_active:
                    return [quantize_step(x, steps) for x, steps in zip([xy, torch.log2(scale), rot, feat], self.quant_steps)]
                symbols = [quantize_step(x, steps) for x, steps in zip([xy, torch.log2(scale), rot], self.quant_steps[:3])]
                steps = self.quant_steps[3]
                q_feat = torch.round(feat / (steps * self._get_feat_step_multipliers(symbols)[:, None]))
                return symbols + [(q_feat, steps, torch.zeros_like(steps))]
            symbols = [quantize_minmax(xy, self.pos_bits), quantize_minmax(scale, self.scale_bits), quantize_minmax(rot, self.rot_bits)]
            q_feat, feat_steps, feat_origins = torch.empty_like(feat), feat.new_empty(self.feat_dim), feat.new_empty(self.feat_dim)
            for channels, bits, _ in self._get_feat_groups():
                q_feat[:, channels], feat_steps[channels], feat_origins[channels] = quantize_minmax(feat[:, channels], bits)
        return symbols + [(q_feat, feat_steps, feat_origins)]

    def _dequantize_symbols(self, symbols):
        """
        Gaussian parameters (scale in pixels) from their quantization indices.
        """
        xy, scale, rot, feat = [dequantize(*symbol) for symbol in symbols]
        if self.quant_mode != "minmax":
            if self.feat_step_adapt_active:
                q_feat, steps, origins = symbols[3]
                feat = dequantize(q_feat, steps * self._get_feat_step_multipliers(symbols[:3])[:, None], origins)
            scale = torch.exp2(scale)
        return xy, scale, rot, feat

    def _get_feat_step_multipliers(self, geometry_symbols):
        """
        Per-Gaussian multipliers [P] of the feature steps (feat_step_adapt): (proxy / its geometric mean)^(-feat_step_gamma), snapped to
        powers of 2^(1/FEAT_STEP_ADAPT_DIVISIONS) within 2^(+-FEAT_STEP_ADAPT_OCTAVES). The proxy is the Gaussian's area ("area") or the sum
        of its squared normalized top-K weights over the pixels ("weight"), the sensitivity of the rendered image to its features. Both only
        depend on the quantization indices of the geometry [position, scale, rotation], and neither depends on the order of the Gaussians,
        so a decoder computes the same multipliers. The multipliers of the last geometry are cached.
        """
        indices = [q for q, _, _ in geometry_symbols]
        cache = self.feat_step_multiplier_cache
        if cache is not None and all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(cache[0], indices)):
            return cache[1]
        if self.feat_step_adapt == "area":
            # log2 of the area is the scale step times the sum of the two log2-scale indices, whose mean is exact
            sums = indices[1].sum(dim=1).to(torch.float64)
            log_proxy = (sums - sums.sum() / len(sums)) * geometry_symbols[1][1][0].to(torch.float64)
        else:
            xy, log_scale, rot = [dequantize(*symbol) for symbol in geometry_symbols]
            log_proxy = torch.log2(self._get_feature_weights(xy, torch.exp2(log_scale), rot).clamp(min=2.0**-40))
            log_proxy = log_proxy - log_proxy.mean()
        limit = FEAT_STEP_ADAPT_DIVISIONS * FEAT_STEP_ADAPT_OCTAVES
        divisions = torch.round(-FEAT_STEP_ADAPT_DIVISIONS * self.feat_step_gamma * log_proxy).clamp(-limit, limit)
        multipliers = torch.exp2(divisions / FEAT_STEP_ADAPT_DIVISIONS).to(self.dtype)
        self.feat_step_multiplier_cache = (indices, multipliers)
        return multipliers

    def _init_gaussians(self, args):
        self.num_gaussians = args.num_gaussians
        self.total_num_gaussians = args.num_gaussians
        self.disable_prog_optim = args.disable_prog_optim
        if not self.disable_prog_optim and not self.evaluate:
            self.initial_ratio = args.initial_ratio
            self.add_times = args.add_times
            self.add_steps = args.add_steps
            self.num_gaussians = math.ceil(self.initial_ratio * self.total_num_gaussians)
            self.max_add_num = math.ceil(float(self.total_num_gaussians-self.num_gaussians) / self.add_times)
            min_steps = self.add_steps * self.add_times + args.post_min_steps
            if args.max_steps < min_steps:
                self.worklog.info(f"Max steps ({args.max_steps:d}) is too small for progressive optimization. Resetting to {min_steps:d}")
                args.max_steps = min_steps
        self._init_pruning(args)
        self.topk = args.topk  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
        self.eps = 1e-7 if args.disable_tiles else 1e-4  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
        self.init_scale = args.init_scale
        self.disable_topk_norm = args.disable_topk_norm
        self.disable_inverse_scale = args.disable_inverse_scale
        self.max_scale = args.max_scale
        if self.max_scale < 0.0:
            raise ValueError(f"Maximum scale must be non-negative but got: {self.max_scale}")
        self.disable_color_init = args.disable_color_init
        self.xy = nn.Parameter(torch.rand(self.num_gaussians, 2, dtype=self.dtype, device=self.device), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(self.num_gaussians, 2, dtype=self.dtype, device=self.device), requires_grad=True)
        self.rot = nn.Parameter(torch.zeros(self.num_gaussians, 1, dtype=self.dtype, device=self.device), requires_grad=True)
        self.feat = nn.Parameter(torch.rand(self.num_gaussians, self.feat_dim, dtype=self.dtype, device=self.device), requires_grad=True)
        self.vis_feat = nn.Parameter(torch.rand_like(self.feat), requires_grad=False)  # Only used for Gaussian ID visualization
        self._log_compression_rate()

    def _init_pruning(self, args):
        self.prune_mode = args.prune_mode
        if self.prune_mode not in ["ratio", "lambda"]:
            raise ValueError(f"Unsupported pruning mode '{self.prune_mode}', valid values include 'ratio' and 'lambda'")
        if not 0.0 <= args.prune_ratio < 1.0:
            raise ValueError(f"Pruning ratio must be in [0, 1) but got: {args.prune_ratio}")
        if not 0.0 <= args.relocate_ratio < 1.0:
            raise ValueError(f"Relocation ratio must be in [0, 1) but got: {args.relocate_ratio}")
        # With prune_mode "lambda", the number of pruned Gaussians is decided during optimization and final_num_gaussians is an upper bound
        self.num_prune = round(args.prune_ratio * self.total_num_gaussians) if self.prune_mode == "ratio" else 0
        self.final_num_gaussians = self.total_num_gaussians - self.num_prune
        self.num_pruned = 0
        self.prune_rounds_done = 0
        self.prune_round_step = 0  # Step of the last pruning round
        self.prune_finished = self.prune_mode == "ratio" and self.num_prune == 0
        if args.prune_min_steps < 0:
            raise ValueError(f"Minimum number of steps between pruning rounds must be non-negative but got: {args.prune_min_steps:d}")
        self.prune_min_steps = args.prune_min_steps
        self.relocate_ratio = args.relocate_ratio
        if self.evaluate:
            # Checkpoints only contain the Gaussians that survived pruning
            checkpoint = torch.load(self._get_ckpt_path(args.ckpt_file), weights_only=False)
            self.num_gaussians = self.final_num_gaussians = checkpoint["state_dict"]["xy"].shape[0]
            return
        if self.relocate_ratio > 0.0 and self.disable_prog_optim:
            raise ValueError("Relocation happens at densification steps and requires progressive optimization")
        if self.prune_mode == "lambda":
            if args.prune_lambda <= 0.0:
                raise ValueError(f"Pruning lambda must be positive but got: {args.prune_lambda}")
            if not 0.0 < args.prune_max_ratio < 1.0:
                raise ValueError(f"Maximum pruning ratio per round must be in (0, 1) but got: {args.prune_max_ratio}")
            if args.prune_score != "loo":
                raise ValueError("Pruning mode 'lambda' requires prune_score 'loo'")
            if args.prune_add_ratio < 0.0:
                raise ValueError(f"Ratio of added Gaussians per round must be non-negative but got: {args.prune_add_ratio}")
            self.prune_lambda = args.prune_lambda
            self.prune_max_ratio = args.prune_max_ratio
            self.prune_add_ratio = args.prune_add_ratio
            self.worklog.info(f"Pruning with lambda {self.prune_lambda:g}: up to {args.prune_times:d} rounds of at most {100.0*self.prune_max_ratio:g}% "
                              f"of the Gaussians, sizes logged before pruning assume all {self.total_num_gaussians:d}"
                              + (f", adding {100.0*self.prune_add_ratio:g}% after each round but the last" if self.prune_add_ratio > 0.0 else ""))
        if not self.prune_finished or self.relocate_ratio > 0.0:
            self.prune_score = args.prune_score
            if self.prune_score not in ["contribution", "loo"]:
                raise ValueError(f"Unsupported pruning score '{self.prune_score}', valid values include 'contribution' and 'loo'")
            if self.prune_score == "loo" and (args.disable_tiles or args.disable_topk_norm):
                raise ValueError("Pruning score 'loo' requires tile-based rendering with top-K normalization")
        if not self.prune_finished:
            if args.prune_times < 1:
                raise ValueError(f"Number of pruning rounds must be positive but got: {args.prune_times:d}")
            self.prune_times = args.prune_times
            self.prune_steps = args.prune_steps
            self.max_prune_num = math.ceil(float(self.num_prune) / args.prune_times)
            # Pruning starts once progressive optimization has added all Gaussians
            self.prune_start_step = 0 if self.disable_prog_optim else self.add_steps * self.add_times
            min_steps = self.prune_start_step + self.prune_steps * args.prune_times + args.post_min_steps
            if args.max_steps < min_steps:
                self.worklog.info(f"Max steps ({args.max_steps:d}) is too small for pruning. Resetting to {min_steps:d}")
                args.max_steps = min_steps

    def _is_final_num_gaussians(self):
        """
        Whether progressive optimization and pruning have both finished.
        """
        return self.prune_finished and self.num_gaussians == self.final_num_gaussians

    def _log_compression_rate(self):
        bytes_uncompressed = 0.0
        curr_channel = 0
        for num_channels, bit_depth in zip(self.input_channels, self.bit_depths):
            bytes_uncompressed += float(self.gt_images[curr_channel:curr_channel+num_channels].numel()) * (bit_depth / 8.0)
            curr_channel += num_channels
        bpp_uncompressed = 0.0
        for num_channels, bit_depth in zip(self.input_channels, self.bit_depths):
            bpp_uncompressed += float(num_channels) * bit_depth
        bppc_uncompressed = bpp_uncompressed / self.feat_dim
        self.bpp_uncompressed = bpp_uncompressed
        self.worklog.info(f"Uncompressed: {bytes_uncompressed/1e3:.2f} KB | {bpp_uncompressed:.3f} bpp | {bppc_uncompressed:.3f} bppc")
        if self.quant_mode == "step":
            self.num_bytes = 0.0
            self.worklog.info("Compressed: depends on the parameter ranges with step quantization, logged when saving checkpoints")
            self.worklog.info("***********************************************")
        else:
            self._log_compressed_size(self._get_fixed_length_bits())

    def _get_fixed_length_bits(self):
        """
        Total size in bits of the Gaussian parameters stored with fixed-length codes.
        With step quantization, each channel uses as many bits as its range of quantization indices requires.
        """
        if self.quant_mode != "step":
            feat_bits = sum(len(channels) * bits for channels, bits, _ in self._get_feat_groups())
            return (2*self.pos_bits + 2*self.scale_bits + self.rot_bits + feat_bits) * self.final_num_gaussians
        symbols = self._get_symbols(self.xy, *self._canonicalize(self._get_scale(), self.rot), self.feat)
        q = torch.cat([q for q, _, _ in symbols], dim=1)
        return torch.ceil(torch.log2(q.amax(dim=0) - q.amin(dim=0) + 1.0)).sum().item() * self.num_gaussians

    def _log_compressed_size(self, bits_compressed):
        bytes_compressed = bits_compressed / 8.0
        bpp_compressed = float(bits_compressed) / self.num_pixels
        bppc_compressed = bpp_compressed / self.feat_dim
        self.num_bytes = bytes_compressed
        label = "Gaussians (float32, not quantized)" if self.quant_mode == "none" else "Compressed"
        self.worklog.info(f"{label}: {bytes_compressed/1e3:.2f} KB | {bpp_compressed:.3f} bpp | {bppc_compressed:.3f} bppc")
        self.worklog.info(f"Compression rate: {self.bpp_uncompressed/bpp_compressed:.2f}x | {100.0*bpp_compressed/self.bpp_uncompressed:.2f}%")
        self.worklog.info("***********************************************")

    def _init_loss(self, args):
        self.l1_loss = None
        self.l2_loss = None
        self.ssim_loss = None
        self.l1_loss_ratio = args.l1_loss_ratio
        self.l2_loss_ratio = args.l2_loss_ratio
        self.ssim_loss_ratio = args.ssim_loss_ratio

    def _init_optimization(self, args):
        self.disable_tiles = args.disable_tiles
        self.start_step = 1
        self.max_steps = args.max_steps
        self.pos_lr = args.pos_lr
        self.scale_lr = args.scale_lr
        self.rot_lr = args.rot_lr
        self.feat_lr = args.feat_lr
        # CUDA graph of the optimization step, captured by _graph_step
        self.cuda_graph = args.cuda_graph
        if self.cuda_graph and not (self.realtime and self.quant_mode != "minmax" and not self.disable_tiles):
            raise ValueError("cuda_graph requires realtime, quant_mode 'step' or 'none', and tiles")
        self.graph, self.graph_key, self.graph_capacity, self.prev_step_event = None, None, 0, None
        self.graph_pool, self.graph_stream = None, None  # Shared by all captures, created by the first one
        self.graph_captures, self.graph_capture_time = 0, 0.0
        self.graph_capture_max = (0.0, 0, 0.0)  # Slowest capture: seconds, step, seconds of its eager step
        if args.stall_trace < 0.0:
            raise ValueError(f"Stall threshold must be non-negative but got: {args.stall_trace}")
        self.stall_trace, self.stall_tracer = args.stall_trace, None
        # Peak number of tile intersections since the last capture, on the GPU and copied to pinned memory by every replay
        self.isect_peak = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.isect_peak_host = torch.zeros(1, dtype=torch.int64, pin_memory=self.cuda_graph)
        self.optimizer = self._new_optimizer()
        self.disable_lr_schedule = args.disable_lr_schedule
        if not self.disable_lr_schedule:
            self.decay_ratio = args.decay_ratio
            self.check_decay_steps = args.check_decay_steps
            self.max_decay_times = args.max_decay_times
            self.decay_threshold = args.decay_threshold
        # Time-budgeted optimization: pruning rounds and a single learning rate decay at fixed fractions of the budget
        self.max_time = args.max_time
        self.prune_time_ratio = args.prune_time_ratio
        self.decay_time_ratio = args.decay_time_ratio
        if self.max_time < 0.0:
            raise ValueError(f"Time budget must be non-negative but got: {self.max_time}")
        if self.max_time > 0.0 and not (0.0 < self.prune_time_ratio <= 1.0 and 0.0 < self.decay_time_ratio <= 1.0):
            raise ValueError(f"Time ratios must be in (0, 1] but got: {self.prune_time_ratio}, {self.decay_time_ratio}")

    def _new_optimizer(self):
        """
        Adam over the current Gaussian parameters. Real-time mode uses the fused implementation (one kernel per parameter group instead of
        about ten), which is faster but not bit-identical. For CUDA graphs, the step counts stay on the GPU (capturable) and the learning rates
        are GPU tensors, which the schedules decay in place, so that the captured step uses the current ones.
        """
        if self.cuda_graph:
            groups = [(self.xy, self.pos_lr), (self.scale, self.scale_lr), (self.rot, self.rot_lr), (self.feat, self.feat_lr)]
            return torch.optim.Adam([{'params': param, 'lr': torch.tensor(lr, device=self.device)} for param, lr in groups],
                                    fused=True, capturable=True)
        return torch.optim.Adam([{'params': self.xy, 'lr': self.pos_lr},
                                 {'params': self.scale, 'lr': self.scale_lr},
                                 {'params': self.rot, 'lr': self.rot_lr},
                                 {'params': self.feat, 'lr': self.feat_lr}], fused=True if self.realtime else None)

    def _init_pos_scale_feat(self, args):
        self.init_mode = args.init_mode
        self.init_random_ratio = args.init_random_ratio
        self.pixel_xy = get_grid(h=self.img_h, w=self.img_w).to(dtype=self.dtype, device=self.device).reshape(-1, 2)
        with torch.no_grad():
            # Position
            if self.init_mode == 'gradient':
                self._compute_gmap()
                self.xy.copy_(self._sample_pos(prob=self.image_gradients))
            elif self.init_mode == 'saliency':
                self.smap_filter_size = args.smap_filter_size
                self._compute_smap(path="models")
                self.xy.copy_(self._sample_pos(prob=self.saliency))
            else:
                self.xy.copy_(self.pixel_xy.detach().clone()[self._sample_pixels(self.num_gaussians)])
            # Scale
            self.scale.fill_(self.init_scale if self.disable_inverse_scale else 1.0/self.init_scale)
            # Feature
            if not self.disable_color_init:
                self.feat.copy_(self._get_target_features(positions=self.xy).detach().clone())

    def _sample_pixels(self, num, prob=None):
        """
        Indices of num distinct pixels on the GPU, drawn uniformly or with the probabilities prob [H*W] (successive sampling without
        replacement, as np.random.choice).
        """
        if prob is None:
            return torch.randperm(self.num_pixels, device=self.device)[:num]
        return torch.multinomial(torch.as_tensor(prob, device=self.device), num, replacement=False)

    def _sample_pos(self, prob):
        num_random = round(self.init_random_ratio*self.num_gaussians)
        selected_random = self._sample_pixels(num_random)
        selected_other = self._sample_pixels(self.num_gaussians-num_random, prob=prob)
        return torch.cat([self.pixel_xy.detach().clone()[selected_random], self.pixel_xy.detach().clone()[selected_other]], dim=0)

    def _compute_gmap(self):
        g_norm = torch.hypot(*compute_image_gradients(torch.pow(self.gt_images, 1.0/self.gamma)))
        g_norm = g_norm / g_norm.max()
        if not self.realtime:
            save_image(g_norm, f"{self.log_dir}/gmap_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}")
            self.worklog.info("Image gradient map successfully saved")
            self.worklog.info("***********************************************")
        g_norm = g_norm.reshape(-1).square()
        self.image_gradients = g_norm / g_norm.sum()

    def _compute_smap(self, path):
        smap = get_smap(torch.pow(self.gt_images.detach().clone(), 1.0/self.gamma), path, self.smap_filter_size)
        if not self.realtime:
            save_image(smap, f"{self.log_dir}/smap_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}")
            self.worklog.info("Saliency map successfully saved")
            self.worklog.info("***********************************************")
        self.saliency = (smap / smap.sum()).reshape(-1)

    def _get_target_features(self, positions):
        with torch.no_grad():
            # target_images [1, C, H, W]; positions [1, 1, P, 2]; top-left [-1, -1]; bottom-right [1, 1]
            target_features = F.grid_sample(self.target_images.unsqueeze(0), positions[None, None, ...] * 2.0 - 1.0, align_corners=False)
            target_features = target_features[0, :, 0, :].permute(1, 0)  # [P, C]
        return target_features

    def _get_ckpt_path(self, ckpt_file):
        """
        Path of the given checkpoint file, or of the latest checkpoint if none is given.
        """
        if ckpt_file != "":
            return os.path.join(self.ckpt_dir, ckpt_file)
        latest_step = get_latest_ckpt_step(self.ckpt_dir)
        if latest_step == -1:
            raise FileNotFoundError(f"No checkpoint found in '{self.ckpt_dir}'")
        return os.path.join(self.ckpt_dir, f"ckpt_step-{latest_step:d}.pt")

    def _load_model(self):
        ckpt_path = self._get_ckpt_path(self.ckpt_file)
        checkpoint = torch.load(ckpt_path, weights_only=False)
        ckpt_color_space = checkpoint.get("color_space", "rgb")  # Checkpoints without this field were trained in RGB
        if ckpt_color_space != self.color_space:
            raise ValueError(f"Checkpoint was trained in '{ckpt_color_space}' but color_space is '{self.color_space}'")
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.start_step = checkpoint["step"]+1
        self.worklog.info(f"Checkpoint '{ckpt_path}' successfully loaded")
        self.worklog.info("***********************************************")

    def _save_model(self, final=False):
        if self.quant_mode != "none":
            self.feat_step_adapt_active = self.feat_step_adapt != "none"
            self._quantize()
        if self.quant_mode != "step" or not self.realtime:  # Fixed-length size of the Gaussians (float32 without quantization)
            self._log_compressed_size(self._get_fixed_length_bits())
        psnr, ssim = self._evaluate(log=False, upsample=False)
        if self.realtime:
            self.lpips_final, self.flip_final, self.msssim_final = float("nan"), float("nan"), float("nan")
        else:
            self._evaluate_extra()
        ckpt_data = {"step": self.step,
                     "psnr": psnr,
                     "ssim": ssim,
                     "lpips": self.lpips_final,
                     "flip": self.flip_final,
                     "msssim": self.msssim_final,
                     "bytes": self.num_bytes,
                     "time": self.total_time_accum,
                     "color_space": self.color_space,
                     "state_dict": self.state_dict(),
                     "optim_state_dict": self.optimizer.state_dict()}
        if final and self.bitstream_codec != "none":
            ckpt_data["bitstream_bytes"] = self._write_bitstream(path=f"{self.log_dir}/bitstream.{self.bitstream_codec}")
        save_path = f"{self.ckpt_dir}/ckpt_step-{self.step:d}.pt"
        torch.save(ckpt_data, save_path)
        self.worklog.info(f"Checkpoint 'ckpt_step-{self.step:d}.pt' successfully saved")
        self.worklog.info(
            f"PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | LPIPS: {self.lpips_final:.4f} | FLIP: {self.flip_final:.4f} | MS-SSIM: {self.msssim_final:.4f}")
        self.worklog.info("***********************************************")
        if not final:  # Optimization continues with the uniform feature steps
            self.feat_step_adapt_active = False

    def _quantize(self):
        """
        Replace the Gaussian parameters by their quantized values, as used for rendering.
        """
        with torch.no_grad():
            xy, scale, rot, feat = self._get_params()
            self.xy.copy_(xy)
            self.scale.copy_(scale if self.disable_inverse_scale else 1.0 / scale)
            self.rot.copy_(rot)
            self.feat.copy_(feat)

    def _estimate_bits(self):
        """
        Fast estimate of the coded size in bits of the quantized Gaussians, without the bitstream header.
        """
        with torch.no_grad():
            symbols = self._get_symbols(self.xy, *self._canonicalize(self._get_scale(), self.rot), self.feat)
            indices = [q.cpu().numpy().astype(np.int64) for q, _, _ in symbols]
        return estimate_bits(indices, channel_refs=self._get_channel_refs())

    def _write_bitstream(self, path):
        """
        Encode the quantized Gaussians into a bitstream file, then decode it and check that it reproduces the quantized parameters (except in
        real-time mode). Returns the size of the bitstream in bytes.
        """
        with torch.no_grad():
            symbols = self._get_symbols(self.xy, *self._canonicalize(self._get_scale(), self.rot), self.feat)
            attributes = [(q.cpu().numpy().astype(np.int64), steps.cpu().numpy().astype(np.float64), origins.cpu().numpy().astype(np.float64))
                          for q, steps, origins in symbols]
            meta = struct.pack("<IIBB", self.img_h, self.img_w, ["minmax", "step"].index(self.quant_mode), ["rgb", "yuv"].index(self.color_space))
            if self.feat_step_adapt != "none":  # A decoder computes the per-Gaussian feature steps from the decoded geometry
                meta += struct.pack("<Bf", ["area", "weight"].index(self.feat_step_adapt) + 1, self.feat_step_gamma)
            data, order, attribute_bytes = encode_bitstream(attributes, meta, codec=self.bitstream_codec, channel_refs=self._get_channel_refs())
            with open(path, "wb") as file:
                file.write(data)
            if not self.realtime:
                with open(path, "rb") as file:
                    decoded, _ = decode_bitstream(file.read())
                if not all(np.array_equal(q[order], q_decoded) for (q, _, _), (q_decoded, _, _) in zip(attributes, decoded)):
                    raise RuntimeError(f"Decoding '{path}' does not reproduce the quantization indices")
                decoded_symbols = [[torch.from_numpy(x).to(dtype=self.dtype, device=self.device) for x in attribute] for attribute in decoded]
                images = self._rasterize(*self._dequantize_symbols(decoded_symbols), self.img_h, self.img_w, self.tile_bounds)
                psnr, _ = self._evaluate(log=False, upsample=False)
                decoded_psnr, _ = self._evaluate(log=False, upsample=False, images=self._to_rgb(images))
        num_bits = 8.0 * len(data)
        names = ["Position", "Scale", "Rotation", "Feature"]
        bits_per_gaussian = [f"{name} {8.0*num_bytes/self.num_gaussians:.2f}" for name, num_bytes in zip(names, attribute_bytes)]
        size_info = f"Bitstream ({self.bitstream_codec}): {len(data)/1e3:.2f} KB | {num_bits/self.num_pixels:.3f} bpp"
        bits_info = f"Bits per Gaussian: {num_bits/self.num_gaussians:.2f}"
        if not self.realtime:  # Comparison with fixed-length codes
            fixed_length_bits = self._get_fixed_length_bits()
            size_info += f" | {num_bits/fixed_length_bits*100.0:.2f}% of fixed-length codes"
            bits_info += f" (fixed-length: {fixed_length_bits/self.num_gaussians:.2f})"
        self.worklog.info(f"{size_info} | Saved to '{path}'")
        self.worklog.info(f"{bits_info} | {' | '.join(bits_per_gaussian)} | Header {len(data)-sum(attribute_bytes):.0f} bytes")
        if not self.realtime:
            self.worklog.info(f"Lossless decoding verified | PSNR: {decoded_psnr:.4f} decoded, {psnr:.4f} encoder side")
        self.worklog.info("***********************************************")
        return len(data)

    def render(self, render_height=None):
        img_h, img_w = self.img_h, self.img_w
        if render_height is not None:
            img_h, img_w = render_height, round((float(render_height)/img_h)*img_w)
        tile_bounds = ((img_w + self.block_w - 1) // self.block_w, (img_h + self.block_h - 1) // self.block_h, 1)
        upsample_ratio = float(img_h) / self.img_h
        with torch.no_grad():
            num_prep_runs = 2
            for _ in range(num_prep_runs):
                self.forward(img_h, img_w, tile_bounds, upsample_ratio, benchmark=True)
            images, render_time = self.forward(img_h, img_w, tile_bounds, upsample_ratio)
            images = self._to_rgb(images)
            path = f"{self.eval_dir}/render_upsample-{upsample_ratio:.1f}_res-{img_h:d}x{img_w:d}"
            self._separate_and_save_images(images=images, channels=self.input_channels, path=path)
        self.worklog.info(f"Step: {self.start_step-1:d} | Time: {render_time:.6f} s")
        self.worklog.info(f"Rendering at resolution ({img_h:d}, {img_w:d}) completed")
        self.worklog.info("***********************************************")
        if self.bitstream_codec != "none":
            self._write_bitstream(path=f"{self.eval_dir}/bitstream.{self.bitstream_codec}")

    def benchmark_render_time(self, num_reps, render_height=None):
        img_h, img_w = self.img_h, self.img_w
        if render_height is not None:
            img_h, img_w = render_height, round((float(render_height)/img_h)*img_w)
        tile_bounds = ((img_w + self.block_w - 1) // self.block_w, (img_h + self.block_h - 1) // self.block_h, 1)
        upsample_ratio = float(img_h) / self.img_h
        with torch.no_grad():
            render_time_all = np.zeros(num_reps, dtype=np.float32)
            num_prep_runs = 2
            for _ in range(num_prep_runs):
                self.forward(img_h, img_w, tile_bounds, upsample_ratio, benchmark=True)
            for rid in range(num_reps):
                render_time = self.forward(img_h, img_w, tile_bounds, upsample_ratio, benchmark=True)
                render_time_all[rid] = render_time
        return render_time_all

    def forward(self, img_h, img_w, tile_bounds, upsample_ratio=None, benchmark=False, isect_capacity=None):
        """
        With isect_capacity, the tile intersections go to buffers of that size without reading their number back to the CPU (CUDA graph
        capture), and their peak number is tracked in isect_peak.
        """
        xy, scale, rot, feat = self._get_params(upsample_ratio=upsample_ratio)
        begin = perf_counter()
        tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
        xy, radii, conics, num_tiles_hit = tmp
        if isect_capacity is not None:
            torch.maximum(self.isect_peak, num_tiles_hit.sum(dtype=torch.int64), out=self.isect_peak)
        if not self.disable_tiles:
            enable_topk_norm = not self.disable_topk_norm
            tmp = xy, radii, conics, num_tiles_hit, feat, img_h, img_w, self.block_h, self.block_w, enable_topk_norm, isect_capacity
            out_image = rasterize_gaussians_sum(*tmp)
        else:
            tmp = xy, conics, feat, img_h, img_w
            out_image = rasterize_gaussians_no_tiles(*tmp)
        render_time = perf_counter() - begin
        if benchmark:
            return render_time
        out_image = out_image.view(-1, img_h, img_w, self.feat_dim).permute(0, 3, 1, 2).contiguous()
        return out_image.squeeze(dim=0), render_time

    def _clamp_scale(self):
        """
        Project the Gaussian scales onto at most max_scale pixels. Inverse scales are kept away from zero, which also keeps them from
        changing sign through a huge Gaussian.
        """
        with torch.no_grad():
            if self.disable_inverse_scale:
                self.scale.clamp_(-self.max_scale, self.max_scale)
            else:
                bound = 1.0 / self.max_scale
                self.scale.copy_(torch.where(self.scale < 0.0, self.scale.clamp(max=-bound), self.scale.clamp(min=bound)))

    def _get_scale(self, upsample_ratio=None):
        scale = self.scale
        if not self.disable_inverse_scale:
            scale = 1.0 / scale
        if upsample_ratio is not None:
            scale = upsample_ratio * scale
        return scale

    def _get_params(self, upsample_ratio=None):
        """
        Gaussian parameters used for rendering (scale in pixels), quantized with straight-through gradients if enabled.
        Scales are quantized before upsampling, so that they match the stored values.
        """
        xy, scale, rot, feat = self.xy, self._get_scale(), self.rot, self.feat
        if self.quant_mode != "none":
            scale, rot = self._canonicalize(scale, rot)
            params = (xy, scale, rot, feat)
            q_params = self._dequantize_symbols(self._get_symbols(*params))
            xy, scale, rot, feat = [x + (q_x - x).detach() for x, q_x in zip(params, q_params)]
        if upsample_ratio is not None:
            scale = upsample_ratio * scale
        return xy, scale, rot, feat

    def _visualize_gaussian_id(self, img_h, img_w, tile_bounds, upsample_ratio=None):
        feat = self._get_params()[3]
        return self._render_features(self.vis_feat * feat.norm(dim=-1, keepdim=True), img_h, img_w, tile_bounds, upsample_ratio)

    def _render_features(self, feat, img_h, img_w, tile_bounds, upsample_ratio=None):
        """
        Render arbitrary per-Gaussian features [P, C] with the current Gaussian geometry.
        """
        xy, scale, rot, _ = self._get_params(upsample_ratio=upsample_ratio)
        return self._rasterize(xy, scale, rot, feat, img_h, img_w, tile_bounds)

    def _rasterize(self, xy, scale, rot, feat, img_h, img_w, tile_bounds):
        """
        Render per-Gaussian features [P, C] with the given Gaussian parameters (scale in pixels).
        """
        tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
        xy, radii, conics, num_tiles_hit = tmp
        if not self.disable_tiles:
            enable_topk_norm = not self.disable_topk_norm
            tmp = xy, radii, conics, num_tiles_hit, feat, img_h, img_w, self.block_h, self.block_w, enable_topk_norm
            out_image = rasterize_gaussians_sum(*tmp)
        else:
            tmp = xy, conics, feat, img_h, img_w
            out_image = rasterize_gaussians_no_tiles(*tmp)
        out_image = out_image.view(-1, img_h, img_w, feat.shape[-1]).permute(0, 3, 1, 2).contiguous()
        return out_image.squeeze(dim=0)

    def optimize(self):
        self.psnr_curr, self.ssim_curr = 0.0, 0.0
        self.best_psnr, self.best_ssim = 0.0, 0.0
        self.decay_times, self.no_improvement_steps = 0, 0
        self.render_time_accum, self.total_time_accum = 0.0, 0.0
        self.round_time_accum = 0.0  # Wall-clock time of the densification and pruning rounds
        self.round_time_max = (0.0, 0)  # Slowest round: seconds, step
        self.lpips_final, self.flip_final, self.msssim_final = 1.0, 1.0, 0.0

        self.step = 0
        adaptive_lr_schedule = not self.disable_lr_schedule and self.max_time == 0.0  # Decays based on periodic evaluation
        wall_begin = perf_counter()
        with torch.no_grad():
            if not self.realtime:
                self._log_images(log_final=False, plot_gaussians=self.vis_gaussians)
        gc_enabled = gc.isenabled()
        if self.realtime:
            gc.disable()  # A full garbage collection takes ~0.1 s, so it waits until after the optimization
        if self.stall_trace > 0.0:
            self.stall_tracer = StallTracer(self.stall_trace, os.path.join(self.log_dir, "stalls.txt"))
        for step in range(self.start_step, self.max_steps+1):
            self.step = step
            step_begin = perf_counter()
            if self.cuda_graph:
                self._graph_step()
            else:
                self.optimizer.zero_grad()
                # Rendering
                images, render_time = self.forward(self.img_h, self.img_w, self.tile_bounds)
                self.render_time_accum += render_time
                # Optimization
                begin = perf_counter()
                self._get_total_loss(images)
                self.total_loss.backward()
                self.optimizer.step()
                if self.max_scale > 0.0:
                    self._clamp_scale()
                if not self.realtime:
                    if self.max_time > 0.0:
                        torch.cuda.synchronize()  # Charge each step its GPU time
                    self.total_time_accum += (perf_counter() - begin + render_time)
            # Logging
            terminate = False
            with torch.no_grad():
                if self.step % self.eval_steps == 0 and (adaptive_lr_schedule or not self.realtime):
                    self._evaluate(log=True, upsample=False)
                    if adaptive_lr_schedule and self._is_final_num_gaussians():
                        terminate = self._lr_schedule()
                if self.step % self.save_image_steps == 0 and not self.realtime:
                    self._log_images(log_final=False, plot_gaussians=self.vis_gaussians)
                if self.step % self.save_ckpt_steps == 0 and self._is_final_num_gaussians():
                    self._save_model()
                begin = perf_counter()
                # Pruned Gaussians are not added back
                if not self.disable_prog_optim and self.step % self.add_steps == 0 and self.num_gaussians + self.num_pruned < self.total_num_gaussians:
                    # Relocation: replace the Gaussians with the lowest score by additional error-guided Gaussians
                    relocate_num = round(self.relocate_ratio * self.num_gaussians)
                    if relocate_num > 0:
                        self._remove_lowest_score(relocate_num, action="Relocating", plot_gaussians=self.vis_gaussians)
                    self._add_gaussians(self.max_add_num + relocate_num, plot_gaussians=self.vis_gaussians)
                if not self.prune_finished and self.step > self.prune_start_step and self._is_prune_step():
                    with self._stall_trace("pruning round"):
                        if self.prune_mode == "lambda":
                            self._prune_gaussians_lambda(plot_gaussians=self.vis_gaussians)
                        else:
                            self._prune_gaussians(self.max_prune_num, plot_gaussians=self.vis_gaussians)
                    self.prune_round_step = self.step
                round_time = perf_counter() - begin
                self.round_time_accum += round_time
                self.round_time_max = max(self.round_time_max, (round_time, self.step))
                if self.realtime:
                    # Wall-clock time of the whole step without synchronizing: reading back the number of tile intersections in every forward
                    # pass (or waiting for the previous step of the CUDA graph) keeps the CPU at most one step ahead of the GPU
                    self.total_time_accum += perf_counter() - step_begin
                elif self.max_time > 0.0:
                    self.total_time_accum += perf_counter() - begin  # Densification and pruning are part of the time budget
                if self.max_time > 0.0:
                    terminate = self._time_schedule() or terminate
                if terminate:
                    break
        if gc_enabled:
            gc.enable()
        if self.stall_tracer is not None:
            self.stall_tracer.close()
            self.stall_tracer = None
        torch.cuda.synchronize()
        optimization_time = perf_counter() - wall_begin
        with torch.no_grad():
            if not self.realtime:
                self._log_images(log_final=True, plot_gaussians=self.vis_gaussians)
            self._save_model(final=True)
        torch.cuda.synchronize()
        self.wall_times = {"optimization": optimization_time, "saving": perf_counter() - wall_begin - optimization_time}
        self.worklog.info(f"Optimization completed | Wall-clock time: {optimization_time:.3f} s optimization ({self.step:d} steps), "
                          f"{self.wall_times['saving']:.3f} s final evaluation, checkpoint, and bitstream")
        if self.round_time_accum > 0.0:
            self.worklog.info(f"Densification and pruning rounds: {self.round_time_accum:.3f} s, slowest {self.round_time_max[0]:.3f} s "
                              f"(step {self.round_time_max[1]:d})")
        if self.cuda_graph:
            seconds, step, eager = self.graph_capture_max
            self.worklog.info(f"CUDA graph captures: {self.graph_captures:d} ({self.graph_capture_time:.3f} s), slowest {seconds:.3f} s "
                              f"(step {step:d}; eager step {eager:.3f} s, capture {seconds-eager:.3f} s)")
        self.worklog.info("***********************************************")
        self.worklog.info(f"Mean scale: {self._get_scale().mean().item():.4f} (pixel) | {self.scale.mean().item():.4f} (raw)")
        self.worklog.info("***********************************************")
        return self.psnr_curr, self.ssim_curr

    def _graph_step(self):
        """
        One optimization step replayed from a CUDA graph (cuda_graph). The step is captured again whenever pruning or densification replaced
        the Gaussian parameters or the optimizer, or the tile intersections outgrew their fixed-capacity buffers (dropping the excess in the
        meantime, for a step or two). Then waits for the previous step, so that the CPU stays at most one step ahead of the GPU.
        """
        key = (self.xy, self.scale, self.rot, self.feat, self.optimizer)
        overflow = int(self.isect_peak_host[0]) > self.graph_capacity  # Peak of a recent step (written by the replays)
        if self.graph is None or overflow or any(a is not b for a, b in zip(key, self.graph_key)):
            with self._stall_trace("CUDA graph capture"):
                self._capture_step(key)
            return
        self.graph.replay()
        event = torch.cuda.Event()
        event.record()
        if self.prev_step_event is not None:
            self.prev_step_event.synchronize()
        self.prev_step_event = event

    @contextmanager
    def _stall_trace(self, event):
        """
        With stall_trace, logs the event if it takes longer than that (with its stacks in stalls.txt).
        """
        if self.stall_tracer is None:
            yield
            return
        self.stall_tracer.begin()
        yield
        message = self.stall_tracer.end(event)
        if message is not None:
            self.worklog.info(f"Step: {self.step:d} | {message}")

    def _capture_step(self, key):
        """
        Optimization step as a CUDA graph: an eager step (on a side stream, as before any capture; it also creates the optimizer state of new
        parameters), then the capture of the next one, with buffers for isect_capacity_ratio times the current number of tile intersections.
        All captures share one memory pool, so that each reuses the memory of the previous graph (never replayed again) instead of allocating
        its own, and unlike torch.cuda.graph, a capture does not empty PyTorch's cache of GPU memory (cudaFree, then cudaMalloc again).
        """
        begin = perf_counter()
        if self.graph_pool is None:
            self.graph_pool, self.graph_stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream()
        # The previous graph lives until the new one is captured: freeing the last graph of a pool would release the pool
        previous_graph = self.graph
        self.graph, self.graph_key, self.prev_step_event = None, None, None
        stream = self.graph_stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.optimizer.zero_grad()
            images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds)
            self._get_total_loss(images)
            self.total_loss.backward()
            self.optimizer.step()
            if self.max_scale > 0.0:
                self._clamp_scale()
            with torch.no_grad():
                xy, scale, rot, _ = self._get_params()
                num_tiles_hit = project_gaussians_2d_scale_rot(xy, scale, rot, self.img_h, self.img_w, self.tile_bounds)[3]
                capacity = max(1024, round(self.isect_capacity_ratio * num_tiles_hit.sum().item()))
        torch.cuda.current_stream().wait_stream(stream)
        eager_time = perf_counter() - begin
        self.isect_peak.zero_()
        self.isect_peak_host.zero_()
        self.optimizer.zero_grad(set_to_none=True)  # backward() then allocates the gradients from the memory pool of the graph
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            graph.capture_begin(pool=self.graph_pool)
            images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds, isect_capacity=capacity)
            self._get_total_loss(images)
            self.total_loss.backward()
            self.optimizer.step()
            if self.max_scale > 0.0:
                self._clamp_scale()
            self.isect_peak_host.copy_(self.isect_peak, non_blocking=True)
            graph.capture_end()
        del previous_graph
        self.graph, self.graph_key, self.graph_capacity = graph, key, capacity
        self.graph_captures += 1
        capture_time = perf_counter() - begin
        self.graph_capture_time += capture_time
        self.graph_capture_max = max(self.graph_capture_max, (capture_time, self.step, eager_time))

    def _get_total_loss(self, images):
        self.total_loss = 0
        if self.l1_loss_ratio > 1e-7:
            self.l1_loss = self.l1_loss_ratio * F.l1_loss(images, self.target_images)
            self.total_loss += self.l1_loss
        else:
            self.l1_loss = None
        if self.l2_loss_ratio > 1e-7:
            self.l2_loss = self.l2_loss_ratio * F.mse_loss(images, self.target_images)
            self.total_loss += self.l2_loss
        else:
            self.l2_loss = None
        if self.ssim_loss_ratio > 1e-7:
            self.ssim_loss = self.ssim_loss_ratio * (1 - fused_ssim(images.unsqueeze(0), self.target_images.unsqueeze(0)))
            self.total_loss += self.ssim_loss
        else:
            self.ssim_loss = None

    def _evaluate(self, log=True, upsample=False, images=None):
        """
        Evaluate the rendered RGB images, or the given ones.
        """
        if upsample:  # Do not log performance metrics for upsampled images
            log = False
        if images is None:
            images = self._render_images(upsample=upsample)
        images = torch.pow(torch.clamp(images, 0.0, 1.0), 1.0/self.gamma)
        gt_images = torch.pow(self.gt_images_upsampled if upsample else self.gt_images, 1.0/self.gamma)
        psnr = get_psnr(images, gt_images).item()
        ssim = fused_ssim(images.unsqueeze(0), gt_images.unsqueeze(0)).item()
        if log:
            self.psnr_curr, self.ssim_curr = psnr, ssim
            loss_results = f"Loss: {self.total_loss.item():.4f}"
            loss_results += f", L1: {self.l1_loss.item():.4f}" if self.l1_loss is not None else ""
            loss_results += f", L2: {self.l2_loss.item():.4f}" if self.l2_loss is not None else ""
            loss_results += f", SSIM: {self.ssim_loss.item():.4f}" if self.ssim_loss is not None else ""
            time_results = f"Total: {self.total_time_accum:.2f} s | Render: {self.render_time_accum:.2f} s"
            self.worklog.info(f"Step: {self.step:d} | {time_results} | {loss_results} | PSNR: {self.psnr_curr:.2f} | SSIM: {self.ssim_curr:.4f}")
        return psnr, ssim

    def _evaluate_extra(self):
        images = torch.pow(torch.clamp(self._render_images(upsample=False), 0.0, 1.0), 1.0/self.gamma)[None, ...]
        gt_images = torch.pow(self.gt_images, 1.0/self.gamma)[None, ...]
        msssim_metric = MS_SSIM(data_range=1.0, size_average=True, channel=self.feat_dim).to(device=self.device).eval()
        self.msssim_final = msssim_metric(images, gt_images).item()
        lpips_metric = LPIPS(net='alex').to(device=self.device).eval()
        flip_metric = LDRFLIPLoss().to(device=self.device).eval()
        num_channels = 1 if self.feat_dim < 3 else 3
        self.lpips_final = lpips_metric(images[:, :num_channels], gt_images[:, :num_channels]).item()
        if self.feat_dim >= 3:
            self.flip_final = flip_metric(images[:, :3], gt_images[:, :3]).item()

    def _log_images(self, log_final=False, plot_gaussians=False):
        images = self._render_images(upsample=False)
        if log_final:
            path = f"{self.log_dir}/render_res-{self.img_h:d}x{self.img_w:d}"
            self._separate_and_save_images(images=images, channels=self.input_channels, path=path)
        psnr, ssim = self._evaluate(log=False, upsample=False)
        path = f"{self.train_dir}/render_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
        self._separate_and_save_images(images=images, channels=self.input_channels, path=path)
        if plot_gaussians:
            path = f"{self.train_dir}/flip-error_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
            save_error_maps(path, images, self.gt_images, channels=self.input_channels,
                            gamma=self.gamma, save_image_format=self.save_image_format)
            # path = f"{self.train_dir}/gaussian-footprint_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
            # visualize_gaussian_footprint(path, self.xy, self._get_scale(), self.rot, self.feat, self.img_h,
            #                     self.img_w, self.input_channels, alpha=0.8, gamma=self.gamma, save_image_format=self.save_plot_format)
            path = f"{self.train_dir}/gaussian-position_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
            every_n = max(1, self.total_num_gaussians // 1000)
            size = 1.5 * (self.img_h * self.img_w) / 1e4
            visualize_gaussian_position(path, images, self.xy, self.input_channels, color="#c0b1fc", size=size,
                                        every_n=every_n, alpha=0.9, gamma=self.gamma, save_image_format=self.save_plot_format)
            images = self._visualize_gaussian_id(self.img_h, self.img_w, self.tile_bounds)
            path = f"{self.train_dir}/gaussian-id_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
            self._separate_and_save_images(images=images, channels=self.input_channels, path=path)
        if self.downsample:
            images = self._render_images(upsample=True)
            psnr, ssim = self._evaluate(log=False, upsample=True)
            img_h, img_w = self.img_h_upsampled, self.img_w_upsampled
            path = f"{self.train_dir}/render_upsample-{self.downsample_ratio:.1f}_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{img_h:d}x{img_w:d}"
            self._separate_and_save_images(images=images, channels=self.input_channels, path=path)

    def _render_images(self, upsample=False):
        """
        Render images in RGB, regardless of the color space of the Gaussian features.
        """
        if upsample:
            images, _ = self.forward(self.img_h_upsampled, self.img_w_upsampled, self.tile_bounds_upsampled, upsample_ratio=self.downsample_ratio)
        else:
            images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds)
        return self._to_rgb(images)

    def _is_prune_step(self):
        """
        Whether the next pruning round is due: every prune_steps steps, or with a time budget, every prune_time_ratio of the budget but at
        least prune_min_steps steps after the previous round (or the start), so that a stall (e.g., a slow round) does not set off the next
        rounds at consecutive steps.
        """
        if self.max_time > 0.0:
            return (self.total_time_accum >= (self.prune_rounds_done + 1) * self.prune_time_ratio * self.max_time
                    and self.step - self.prune_round_step >= self.prune_min_steps)
        return (self.step - self.prune_start_step) % self.prune_steps == 0

    def _time_schedule(self):
        """
        Schedule of time-budgeted optimization: a single learning rate decay after decay_time_ratio of the budget, and termination once the
        budget is used up. Pruning rounds that are still due by then are skipped (when stalls delayed them), because the steps after them
        would not recover the image. Returns whether to terminate.
        """
        if not self.prune_finished and self.total_time_accum >= self.decay_time_ratio * self.max_time:
            self.prune_finished = True
            self.final_num_gaussians = self.num_gaussians
            self.worklog.info(f"Step: {self.step:d} | Pruning stopped after {self.prune_rounds_done:d} of {self.prune_times:d} rounds at "
                              f"{self.total_time_accum:.2f} s with {self.num_gaussians:d} Gaussians ({self.num_pruned:d} pruned)")
            self.worklog.info("***********************************************")
        if not self.disable_lr_schedule and self.decay_times == 0 and self.total_time_accum >= self.decay_time_ratio * self.max_time:
            self.decay_times = 1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] /= self.decay_ratio
            self.worklog.info(f"Step: {self.step:d} | Learning rate decayed by {self.decay_ratio:.1f} after {self.total_time_accum:.2f} s")
            self.worklog.info("***********************************************")
        if self.total_time_accum >= self.max_time:
            self.worklog.info(f"Step: {self.step:d} | Time budget of {self.max_time:.2f} s used up ({self.total_time_accum:.2f} s)")
            self.worklog.info("***********************************************")
            return True
        return False

    def _lr_schedule(self):
        if (self.psnr_curr <= self.best_psnr + 100*self.decay_threshold or self.ssim_curr <= self.best_ssim + self.decay_threshold):
            self.no_improvement_steps += self.eval_steps
            if self.no_improvement_steps >= self.check_decay_steps:
                self.no_improvement_steps = 0
                self.decay_times += 1
                if self.decay_times > self.max_decay_times:
                    return True
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= self.decay_ratio
                self.worklog.info(f"Learning rate decayed by {self.decay_ratio:.1f}")
                self.worklog.info("***********************************************")
            return False
        else:
            self.best_psnr = self.psnr_curr
            self.best_ssim = self.ssim_curr
            self.no_improvement_steps = 0
            return False

    def _add_gaussians(self, add_num, plot_gaussians=False):
        add_num = min(add_num, self.total_num_gaussians-self.num_gaussians)
        if add_num <= 0:
            return
        raw_images = self._render_images(upsample=False)
        images = torch.pow(torch.clamp(raw_images, 0.0, 1.0), 1.0/self.gamma)
        gt_images = torch.pow(self.gt_images, 1.0/self.gamma)
        kernel_size = round(np.sqrt(self.img_h * self.img_w) // 400)
        if kernel_size >= 1:
            kernel_size = max(3, kernel_size)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            gt_images = gaussian_blur(img=gt_images, kernel_size=kernel_size)
        diff_map = (gt_images - images).detach().clone()
        error_map = torch.pow(torch.abs(diff_map).mean(dim=0).reshape(-1), 2.0)
        sample_prob = error_map / error_map.sum()
        selected = self._sample_pixels(add_num, prob=sample_prob)
        # New Gaussians
        new_xy = self.pixel_xy.detach().clone()[selected]
        new_scale = torch.ones(add_num, 2, dtype=self.dtype, device=self.device)
        init_scale = self.init_scale
        new_scale.fill_(init_scale if self.disable_inverse_scale else 1.0/init_scale)
        new_rot = torch.zeros(add_num, 1, dtype=self.dtype, device=self.device)
        new_feat = self._from_rgb(diff_map).permute(1, 2, 0).reshape(-1, self.feat_dim)[selected]
        new_vis_feat = torch.rand_like(new_feat)
        # Old Gaussians
        old_xy = self.xy.detach().clone()
        old_scale = self.scale.detach().clone()
        old_rot = self.rot.detach().clone()
        old_feat = self.feat.detach().clone()
        old_vis_feat = self.vis_feat.detach().clone()
        # Update trainable parameters
        self.num_gaussians += add_num
        all_xy = torch.cat([old_xy, new_xy], dim=0)
        all_scale = torch.cat([old_scale, new_scale], dim=0)
        all_rot = torch.cat([old_rot, new_rot], dim=0)
        all_feat = torch.cat([old_feat, new_feat], dim=0)
        all_vis_feat = torch.cat([old_vis_feat, new_vis_feat], dim=0)
        self.xy = nn.Parameter(all_xy, requires_grad=True)
        self.scale = nn.Parameter(all_scale, requires_grad=True)
        self.rot = nn.Parameter(all_rot, requires_grad=True)
        self.feat = nn.Parameter(all_feat, requires_grad=True)
        self.vis_feat = nn.Parameter(all_vis_feat, requires_grad=False)
        # Plot Gaussians
        if plot_gaussians:
            path = f"{self.train_dir}/add-gaussians_step-{self.step:d}_num-{self.num_gaussians:d}_res-{self.img_h:d}x{self.img_w:d}"
            every_n = max(1, self.total_num_gaussians // 2000)
            size = (self.img_h * self.img_w) / 1e4
            visualize_added_gaussians(path, raw_images, old_xy, new_xy, self.input_channels, size=size, every_n=every_n,
                                      alpha=0.8, gamma=self.gamma, save_image_format=self.save_plot_format)
        # Update optimizer
        self.optimizer = self._new_optimizer()
        self.worklog.info(f"Step: {self.step:d} | Adding {add_num:d} Gaussians ({self.num_gaussians-add_num:d} -> {self.num_gaussians:d})")
        self.worklog.info("***********************************************")

    def _get_contribution(self):
        """
        Contribution of each Gaussian to the rendered image, measured as its total blending weight summed over all pixels.
        Rendered images are linear in the features, so these weights are the gradients of the image sum w.r.t. all-one features.
        """
        ones = torch.ones(self.num_gaussians, 1, dtype=self.dtype, device=self.device, requires_grad=True)
        with torch.enable_grad():
            images = self._render_features(ones, self.img_h, self.img_w, self.tile_bounds)
            contribution, = torch.autograd.grad(images.sum(), ones)
        return contribution.squeeze(dim=-1)

    def _iter_topk_weights(self, xy, scale, rot, feat, chunk_size=2**18):
        """
        Top-K Gaussians of each pixel with their blending weights before normalization, in chunks of pixels to bound memory usage at high
        resolutions: yields the first pixel of the chunk, the ids [P, K] (empty entries as 0), valid [P, K], and the weights [P, K] (0 for
        empty entries).
        """
        xy, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(xy, scale, rot, self.img_h, self.img_w, self.tile_bounds)
        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)
        _, _, _, gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(len(xy), num_intersects, xy, radii, cum_tiles_hit, self.tile_bounds)
        _, pixel_topk = nd_rasterize_forward_topk_norm(self.tile_bounds, (self.block_w, self.block_h, 1), (self.img_w, self.img_h, 1),
                                                       gaussian_ids_sorted, tile_bins, xy.contiguous(), conics.contiguous(), feat.contiguous())
        pixel_topk = pixel_topk.view(-1, self.topk).long()  # [H*W, K], -1 for empty entries
        rows, cols = torch.meshgrid(torch.arange(self.img_h, device=self.device), torch.arange(self.img_w, device=self.device), indexing="ij")
        pixels = torch.stack([cols, rows], dim=-1).reshape(-1, 1, 2).to(self.dtype)  # Pixel coordinates used by the rasterizer
        for begin in range(0, len(pixel_topk), chunk_size):
            ids = pixel_topk[begin:begin+chunk_size]
            valid = ids >= 0
            ids = ids.clamp(min=0)
            delta = xy[ids] - pixels[begin:begin+chunk_size]  # [P, K, 2]
            conic = conics[ids]  # [P, K, 3]
            sigma = 0.5 * (conic[..., 0] * delta[..., 0]**2 + conic[..., 2] * delta[..., 1]**2) + conic[..., 1] * delta[..., 0] * delta[..., 1]
            yield begin, ids, valid, torch.exp(-sigma) * valid

    def _get_feature_weights(self, xy, scale, rot):
        """
        Sum over the pixels of each Gaussian's squared normalized top-K weight [P] (float64): the sensitivity of the rendered image to its
        features. The weights are normalized in float64 and summed in fixed point with integer atomics, so the sums do not depend on the
        order of the Gaussians or of the additions.
        """
        fixed_point = torch.zeros(len(xy), dtype=torch.int64, device=self.device)
        no_feat = torch.zeros(len(xy), 1, dtype=self.dtype, device=self.device)  # The top-K lists do not depend on the features
        for _, ids, valid, weights in self._iter_topk_weights(xy, scale, rot, no_feat):
            weights = weights.to(torch.float64)
            normalized = weights / (weights.sum(dim=1, keepdim=True) + self.eps)
            fixed_point.index_add_(0, ids[valid], torch.round(normalized[valid].square() * 2.0**40).to(torch.int64))
        return fixed_point.to(torch.float64) / 2.0**40

    def _get_loo_loss_increase(self, metric="l1", chunk_size=2**18):
        """
        Increase of the loss when removing each Gaussian on its own (leave-one-out), computed from a single rendering pass.
        The loss is the L1 loss (metric "l1") or the squared error summed over pixels and averaged over channels in 8-bit units (metric "sse").
        Removing Gaussian k from the top-K list of a pixel changes its value from S_f / (S_w + eps) to (S_f - w_k f_k) / (S_w - w_k + eps),
        where S_w and S_f are the sums of the top-K weights and weighted features. Ignores the Gaussian that would move into the top-K list.
        """
        xy, scale, rot, feat = self._get_params()
        targets = self.target_images.permute(1, 2, 0).reshape(-1, self.feat_dim)
        loss_increase = torch.zeros(self.num_gaussians, dtype=self.dtype, device=self.device)
        for begin, ids, valid, weights in self._iter_topk_weights(xy, scale, rot, feat, chunk_size):
            weighted_feat = weights.unsqueeze(-1) * feat[ids]  # [P, K, C]
            sum_weights = weights.sum(dim=1, keepdim=True)  # [P, 1]
            sum_feat = weighted_feat.sum(dim=1)  # [P, C]
            target = targets[begin:begin+chunk_size]
            loo_values = (sum_feat.unsqueeze(1) - weighted_feat) / (sum_weights - weights + self.eps).unsqueeze(-1)  # [P, K, C]
            diff, loo_diff = sum_feat / (sum_weights + self.eps) - target, loo_values - target.unsqueeze(1)
            if metric == "sse":
                error, loo_error = diff.square().sum(dim=-1, keepdim=True), loo_diff.square().sum(dim=-1)  # [P, 1], [P, K]
            else:
                error, loo_error = diff.abs().sum(dim=-1, keepdim=True), loo_diff.abs().sum(dim=-1)
            loss_increase.index_add_(0, ids[valid], (loo_error - error)[valid])
        return loss_increase * (255.0**2 / self.feat_dim if metric == "sse" else 1.0 / targets.numel())

    def _prune_gaussians(self, prune_num, plot_gaussians=False):
        prune_num = min(prune_num, self.max_prune_num, self.num_prune-self.num_pruned)
        if prune_num <= 0:
            return
        self._remove_lowest_score(prune_num, action="Pruning", plot_gaussians=plot_gaussians)
        self.num_pruned += prune_num
        self.prune_rounds_done += 1
        self.prune_finished = self.num_pruned == self.num_prune

    def _prune_gaussians_lambda(self, plot_gaussians=False):
        """
        Pruning round of prune_mode "lambda": remove the Gaussians whose removal increases the squared error by less than prune_lambda times
        the bits they take (the average of the current Gaussians), at most prune_max_ratio of them. With prune_add_ratio, each round but the last
        then adds that fraction of Gaussians at high-error pixels, which the next round tests, so that the number of Gaussians can also grow.
        Pruning finishes after prune_times rounds, or once a round finds no Gaussian to remove (without adding).
        """
        self.prune_rounds_done += 1
        sse_increase = self._get_loo_loss_increase(metric="sse")
        bits_per_gaussian = self._estimate_bits() / self.num_gaussians
        threshold = self.prune_lambda * bits_per_gaussian
        num_below = int((sse_increase < threshold).sum().item())
        prune_num = min(num_below, round(self.prune_max_ratio * self.num_gaussians))
        round_info = (f"Round {self.prune_rounds_done:d}: {num_below:d} below lambda x {bits_per_gaussian:.1f} bits per Gaussian = "
                      f"{threshold:.1f} SSE")
        if prune_num > 0:
            pruned = torch.topk(sse_increase, prune_num, largest=False).indices
            prune_info = round_info
            if not self.realtime:  # Removing many Gaussians at once costs more than the sum of their leave-one-out scores, so measure it
                keep = torch.ones(self.num_gaussians, dtype=torch.bool, device=self.device)
                keep[pruned] = False
                psnr_before, _ = self._evaluate(log=False, upsample=False)
                images = self._rasterize(*[param[keep] for param in self._get_params()], self.img_h, self.img_w, self.tile_bounds)
                psnr_after, _ = self._evaluate(log=False, upsample=False, images=self._to_rgb(images))
                prune_info += f" | PSNR before fine-tuning: {psnr_before:.2f} -> {psnr_after:.2f} dB"
            self._remove_gaussians(pruned, "Pruning", prune_info, plot_gaussians=plot_gaussians)
            self.num_pruned += prune_num
            self.final_num_gaussians = self.num_gaussians
        if (prune_num == 0 and self.prune_add_ratio == 0.0) or self.prune_rounds_done == self.prune_times:
            self.prune_finished = True
            self.worklog.info(f"Step: {self.step:d} | {round_info} | Pruning finished with {self.num_gaussians:d} Gaussians "
                              f"({self.num_pruned:d} pruned in {self.prune_rounds_done:d} rounds)")
            self.worklog.info("***********************************************")
        elif self.prune_add_ratio > 0.0:
            add_num = round(self.prune_add_ratio * self.num_gaussians)
            self.total_num_gaussians = self.num_gaussians + add_num  # _add_gaussians adds up to total_num_gaussians
            self._add_gaussians(add_num, plot_gaussians=plot_gaussians)
            self.final_num_gaussians = self.num_gaussians

    def _remove_lowest_score(self, remove_num, action, plot_gaussians=False):
        """
        Remove the Gaussians with the lowest pruning score.
        """
        if self.prune_score == "contribution":
            score = self._get_contribution()
            pruned = torch.topk(score, remove_num, largest=False).indices
            score_info = f"Pruned contribution: {100.0*score[pruned].sum().item()/score.sum().item():.2f}%"
        else:
            score = self._get_loo_loss_increase()
            pruned = torch.topk(score, remove_num, largest=False).indices
            score_info = f"Predicted L1 increase: {score[pruned].sum().item():.5f}"
        self._remove_gaussians(pruned, action, score_info, plot_gaussians=plot_gaussians)

    def _remove_gaussians(self, pruned, action, score_info, plot_gaussians=False):
        """
        Remove the given Gaussians [indices], keeping the optimizer state of the remaining ones.
        """
        remove_num = len(pruned)
        keep = torch.ones(self.num_gaussians, dtype=torch.bool, device=self.device)
        keep[pruned] = False
        # Plot Gaussians (red: kept, green: removed)
        if plot_gaussians:
            path = f"{self.train_dir}/{action.lower()}-gaussians_step-{self.step:d}_num-{self.num_gaussians-remove_num:d}_res-{self.img_h:d}x{self.img_w:d}"
            every_n = max(1, self.total_num_gaussians // 2000)
            size = (self.img_h * self.img_w) / 1e4
            visualize_added_gaussians(path, self._render_images(upsample=False), self.xy[keep], self.xy[pruned], self.input_channels, size=size,
                                      every_n=every_n, alpha=0.8, gamma=self.gamma, save_image_format=self.save_plot_format)
        # Update trainable parameters, keeping the optimizer state of the remaining Gaussians
        for name, param_group in zip(["xy", "scale", "rot", "feat"], self.optimizer.param_groups):
            old_param = param_group["params"][0]
            new_param = nn.Parameter(old_param.detach()[keep], requires_grad=True)
            state = self.optimizer.state.pop(old_param, None)
            if state is not None:
                state["exp_avg"], state["exp_avg_sq"] = state["exp_avg"][keep], state["exp_avg_sq"][keep]
                self.optimizer.state[new_param] = state
            param_group["params"] = [new_param]
            setattr(self, name, new_param)
        self.vis_feat = nn.Parameter(self.vis_feat.detach()[keep], requires_grad=False)
        self.num_gaussians -= remove_num
        self.worklog.info(
            f"Step: {self.step:d} | {action} {remove_num:d} Gaussians ({self.num_gaussians+remove_num:d} -> {self.num_gaussians:d}) | {score_info}")
        self.worklog.info("***********************************************")
