import gc
import logging
import math
import os
import sys
import warnings
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_l1_ssim, fused_ssim
from lpips import LPIPS
from pytorch_msssim import MS_SSIM

from compress import bitstream_order, decode_bitstream, encode_bitstream
from compress.prediction import get_prediction_refs, predict_residuals, reconstruct_indices
from compress.pruning import add_probabilities, contribution, keep_gaussians, loo_loss_increase, select_lambda_pruning
from compress.quantization import (
    FEAT_STEP_ADAPT_DIVISIONS,
    POS_REFINE_BITS,
    ROT_STEP_ADAPT_OCTAVES,
    bitstream_meta,
    canonicalize,
    dequantize,
    dequantize_symbols,
    estimate_symbol_bits,
    feat_step_multipliers,
    feature_weights,
    fixed_length_bits,
    get_channel_refs,
    get_minmax_symbols,
    get_step_symbols,
    pos_step_exponent_table,
    pos_step_exponents,
    rot_step_exponent_table,
    rot_step_exponents,
    symbols_to_numpy,
)
from compress.rate import estimated_bits, rate_loss, rate_model
from compress.scale_prediction import predicted_sums, reconstruct_scales, scale_residuals
from gsplat import (
    compute_cumulative_intersects,
    project_gaussians_2d_scale_rot,
    rasterize_gaussians_no_tiles,
    rasterize_gaussians_sum,
)
from gsplat.cuda import nd_rasterize_forward_topk_norm
from gsplat.utils import sort_intersects_by_tile
from utils.color_utils import UV_STEP_SCALE, Y_STEP_SCALE, get_color_transforms, get_loss_transform, get_yuv_channel_indices, transform_images
from utils.flip import LDRFLIPLoss
from utils.image_utils import (
    compute_image_gradients,
    compute_structure_tensor,
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
from utils.init_utils import STRUCTURE_TENSOR_SIGMA, find_fine_blobs, initial_shapes, local_std_density
from utils.misc_utils import clean_dir, get_latest_ckpt_step, save_cfg, set_random_seed
from utils.saliency_utils import get_smap
from utils.stall_utils import StallTracer

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips")

# Settings derived from the quantization parameter (qp), fitted on the 480p/720p UVG equal-time runs (A100, 40/90 s; experiments/qp/fit.py):
# x265's RD slope in the units of prune_lambda is 0.9-1.1x HEVC's lambda at QP 22-37. The number of Gaussians that paid off doubles every
# ~5 QP (4.7-5.6 per image) and varies 2.5x between images; QP_DENSITY is the most for any image. Refining the feature step paid off
# from QP ~44 (1/32 -> 1/64) and below QP ~29 (1/64 -> 1/128), i.e., the optimal step halves every ~15 QP. The settings are derived at the
# mapped QP qp_offset + qp_slope * qp; the default mapping matches the PSNR of x265 all-intra 4:2:0 (results/lowres/hevc.csv) at the same QP,
# fitted on the real-time runs at QPs 16-40 with 2-4 s A100 budgets (phase qp-refit, experiments/realtime/qp_refit.py).
QP_DENSITY = 8.0  # Gaussians per 1000 pixels at QP 32
QP_DENSITY_DOUBLING = 5.0  # QPs over which the number of Gaussians doubles
QP_FEAT_STEP = 1.0 / 64.0  # Feature step at QP 37
QP_FEAT_STEP_HALVING = 15.0  # QPs over which the feature step halves


def get_mapped_qp(args):
    """
    The QP at which the rate settings are derived from the quantization parameter qp.
    """
    return args.qp_offset + args.qp_slope * args.qp


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
        Rate settings from the quantization parameter, derived at the mapped QP qp_offset + qp_slope * qp: lambda pruning with HEVC's lambda
        at the mapped QP, starting from a content-agnostic overestimate of the number of Gaussians, and a feature step that halves every
        QP_FEAT_STEP_HALVING QPs. In YUV mode, Y, and U and V get the error per channel of RGB features at that step (Y_STEP_SCALE and
        UV_STEP_SCALE times it), times y_step_ratio and uv_step_ratio. The other steps are unchanged.
        """
        if args.y_step_ratio <= 0.0 or args.uv_step_ratio <= 0.0:
            raise ValueError(f"Y and UV step ratios must be positive but got: {args.y_step_ratio:g}, {args.uv_step_ratio:g}")
        if args.qp == 0.0:
            return
        if args.qp < 0.0:
            raise ValueError(f"Quantization parameter must be positive (or 0 to disable it) but got: {args.qp}")
        if args.quant_mode not in ["step", "none"]:
            raise ValueError("The quantization parameter requires quant_mode 'step' or 'none' (qp 0 disables it)")
        if args.qp_overestimate <= 0.0 or args.qp_lambda_scale <= 0.0:
            raise ValueError(f"Overestimation and lambda factors must be positive but got: {args.qp_overestimate}, {args.qp_lambda_scale}")
        if args.qp_slope <= 0.0:
            raise ValueError(f"QP slope must be positive but got: {args.qp_slope}")
        qp = get_mapped_qp(args)
        density = QP_DENSITY * 2.0**((32.0 - qp) / QP_DENSITY_DOUBLING)
        args.num_gaussians = max(1, round(args.qp_overestimate * density * self.img_h * self.img_w / 1000.0))
        args.feat_step = QP_FEAT_STEP * 2.0**((qp - 37.0) / QP_FEAT_STEP_HALVING)
        args.y_step = args.y_step_ratio * Y_STEP_SCALE * args.feat_step
        args.uv_step = args.uv_step_ratio * UV_STEP_SCALE * args.feat_step
        args.prune_mode = "lambda"
        args.prune_lambda = args.qp_lambda_scale * 0.57 * 2.0**((qp - 12.0) / 3.0)
        yuv_steps = f" (Y 1/{1.0/args.y_step:.1f}, UV 1/{1.0/args.uv_step:.1f})" if args.color_space == "yuv" else ""
        self.worklog.info(f"QP {args.qp:g} (settings of QP {qp:.2f}): {args.num_gaussians:d} Gaussians before pruning ({args.qp_overestimate:g} x "
                          f"{density:.2f} per 1000 pixels), feature step 1/{1.0/args.feat_step:.1f}{yuv_steps}, lambda {args.prune_lambda:.2f} "
                          f"({args.qp_lambda_scale:g} x HEVC's)")
        if not self.evaluate:
            save_cfg(path=f"{self.log_dir}/cfg_train.yaml", args=args)  # Record the derived settings

    def _load_target_images(self, path, downsample_ratio=None):
        self.gt_images, self.input_channels, self.image_fnames, self.bit_depths = load_images(
            load_path=path, downsample_ratio=downsample_ratio, gamma=self.gamma)
        # Contiguous, so that the losses read them directly (a channel-first view of the loaded [H, W, C] data costs strided kernels and
        # copies every step)
        self.gt_images = torch.from_numpy(self.gt_images).to(dtype=self.dtype, device=self.device).contiguous()
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
        self.uv_weight = args.uv_weight
        if self.uv_weight <= 0.0:
            raise ValueError(f"UV weight must be positive but got: {self.uv_weight:g}")
        self.feat_dim = sum(self.input_channels)
        self.loss_transform, self.rgb_loss_transform = None, None
        if self.color_space == "yuv":
            self.rgb_to_yuv, self.yuv_to_rgb = get_color_transforms(self.input_channels, dtype=self.dtype, device=self.device)
            self.y_channels, self.uv_channels, self.other_channels = get_yuv_channel_indices(self.input_channels, device=self.device)
            # The losses, the error of lambda pruning, and the error map of added Gaussians compare RGB images of the luma-weighted features
            # (uv_weight 1: RGB images, as in RGB mode), from the features and from RGB images
            self.loss_transform = get_loss_transform(self.input_channels, self.uv_weight, dtype=self.dtype, device=self.device)
            self.rgb_loss_transform = self.loss_transform @ self.rgb_to_yuv
        # Optimization target in the color space of the Gaussian features, and in the space of the losses
        self.target_images = self._from_rgb(self.gt_images)
        self.loss_target_images = self.gt_images if self.loss_transform is None else transform_images(self.target_images, self.loss_transform)

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
        if args.feat_step_adapt not in ["none", "area", "weight"]:
            raise ValueError(f"Unsupported feature step adaptation '{args.feat_step_adapt}', valid values include 'none', 'area', and 'weight'")
        self.feat_step_adapt = args.feat_step_adapt if self.quant_mode == "step" else "none"  # Only the step mode has feature steps
        # The values that the bitstream stores, and the gamma of each feature channel
        self.feat_step_gamma = float(np.float32(args.feat_step_gamma))
        self.uv_step_gamma = float(np.float32(args.uv_step_gamma if args.uv_step_gamma >= 0.0 else args.feat_step_gamma))
        self.feat_step_gammas = torch.full((self.feat_dim,), self.feat_step_gamma, dtype=torch.float64, device=self.device)
        if self.color_space == "yuv":
            self.feat_step_gammas[self.uv_channels] = self.uv_step_gamma
        # The per-Gaussian feature steps apply to the saved model and in evaluation; optimization uses the uniform steps
        self.feat_step_adapt_active = self.feat_step_adapt != "none" and self.evaluate
        self.feat_step_multiplier_cache = None
        self._init_geometry_steps(args)
        self.wrap_rot = args.wrap_rot
        self.canonical_rot = args.canonical_rot
        # The saved model and evaluation also render the canonical rotation; optimization renders the rotation as it is (the same ellipses)
        self.canonical_rot_active = self.canonical_rot and self.evaluate
        self.bitstream_codec = args.bitstream_codec
        if self.bitstream_codec not in ["none", "lzma", "rans"]:
            raise ValueError(f"Unsupported bitstream codec '{self.bitstream_codec}', valid values include 'none', 'lzma', and 'rans'")
        if self.bitstream_codec != "none" and self.quant_mode == "none":
            raise ValueError("Writing a bitstream requires quantization")
        if args.color_transform not in ["none", "subtract-green"]:
            raise ValueError(f"Unsupported color transform '{args.color_transform}', valid values include 'none' and 'subtract-green'")
        self.color_transform = args.color_transform if self.color_space == "rgb" else "none"  # YUV features are coded as they are
        self.feat_prediction = args.feat_prediction
        self.feat_prediction_refs = get_prediction_refs(self.input_channels, self.color_space)

    def _init_geometry_steps(self, args):
        """
        Per-Gaussian rotation and position steps and the scale prediction (step mode only, ignored otherwise): exponent tables of
        compress.quantization on the GPU, and the refinement bits of two-stage positions in the bitstream.
        """
        if args.rot_step_adapt not in ["none", "aspect"]:
            raise ValueError(f"Unsupported rotation step adaptation '{args.rot_step_adapt}', valid values include 'none' and 'aspect'")
        if args.pos_step_adapt not in ["none", "major"]:
            raise ValueError(f"Unsupported position step adaptation '{args.pos_step_adapt}', valid values include 'none' and 'major'")
        step_mode = self.quant_mode == "step"
        self.rot_step_adapt = args.rot_step_adapt if step_mode else "none"
        self.pos_step_adapt = args.pos_step_adapt if step_mode else "none"
        self.scale_prediction = args.scale_prediction and step_mode
        # The values that the bitstream stores
        self.rot_step_aspect, self.rot_step_gamma = float(np.float32(args.rot_step_aspect)), float(np.float32(args.rot_step_gamma))
        self.pos_step_gamma = float(np.float32(args.pos_step_gamma))
        self.rot_step_table, self.pos_step_table, self.rot_quarter = None, None, 0
        if self.rot_step_adapt != "none" or self.pos_step_adapt != "none":
            # Integer rotation indices per quarter turn, so that the nested grids (and canonical rotations) are exact
            self.rot_quarter = round(0.5 * math.pi / self.rot_step)
            min_quarter = 2**ROT_STEP_ADAPT_OCTAVES if self.rot_step_adapt != "none" else 1
            if abs(self.rot_quarter * self.rot_step - 0.5 * math.pi) > 1e-9 or self.rot_quarter % min_quarter:
                raise ValueError(f"Per-Gaussian rotation or position steps require a rot_step that divides 90 degrees (rotation steps: "
                                 f"{90 / min_quarter:g} degrees) but got: {args.rot_step:g}")
        if self.rot_step_adapt != "none":
            table = rot_step_exponent_table(self.scale_step, self.rot_step_aspect, self.rot_step_gamma)
            self.rot_step_table = torch.from_numpy(table).to(self.device)
        if self.pos_step_adapt != "none":
            if not 0.0 < self.pos_step_gamma <= 2.0:
                raise ValueError(f"pos_step_gamma must be in (0, 2] but got: {self.pos_step_gamma:g}")
            self.pos_step_table = torch.from_numpy(pos_step_exponent_table(self.scale_step, self.rot_step, self.pos_step_gamma)).to(self.device)
        self.pos_refine_bits = POS_REFINE_BITS if self.pos_step_adapt != "none" else 0
        # Like the per-Gaussian feature steps, the rotation and position steps apply to the saved model and in evaluation
        self.geometry_step_adapt = self.rot_step_adapt != "none" or self.pos_step_adapt != "none"
        self.geometry_step_adapt_active = self.geometry_step_adapt and self.evaluate

    def _get_rot_step_exponents(self, q_scale):
        """
        Exponents [P] (int64) of the per-Gaussian rotation steps of the log2-scale indices [P, 2] (zeros without rot_step_adapt).
        """
        if self.rot_step_table is None:
            return torch.zeros(len(q_scale), dtype=torch.int64, device=q_scale.device)
        return rot_step_exponents(q_scale, self.rot_step_table)

    def _get_pos_step_exponents(self, q_scale, q_rot):
        """
        Exponents [P, 2] (int64) of the per-Gaussian position steps of the log2-scale indices [P, 2] and the rotation indices [P] in base steps
        (zeros without pos_step_adapt).
        """
        if self.pos_step_table is None:
            return torch.zeros(len(q_scale), 2, dtype=torch.int64, device=q_scale.device)
        return pos_step_exponents(q_scale, q_rot, self.pos_step_table, self.rot_quarter)

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

    def _canonicalize(self, scale, rot, quarter_turns=None):
        """
        Equivalent Gaussian parameters to quantize (compress.quantization.canonicalize), with the quarter turns of canonical_rot by default.
        """
        quarter_turns = self.canonical_rot if quarter_turns is None else quarter_turns
        return canonicalize(scale, rot, quarter_turns, self.wrap_rot, log_scale=self.quant_mode != "minmax")

    def _get_symbols(self, xy, scale, rot, feat):
        """
        Quantization indices [P, C] of the canonicalized Gaussian parameters (scale in pixels), with the per-channel steps and origins [C]
        that dequantize them (compress.quantization). Scales are quantized in pixels in "minmax" mode and in log2 pixels in "step" mode.
        Without quantization ("none"), the "step" indices estimate the rate. With per-Gaussian feature steps (feat_step_adapt), the feature
        steps are the base steps, which _get_feat_step_multipliers scales per Gaussian.
        """
        if self.quant_mode == "minmax":
            feat_groups = [(channels, bits) for channels, bits, _ in self._get_feat_groups()]
            return get_minmax_symbols(xy, scale, rot, feat, [self.pos_bits, self.scale_bits, self.rot_bits], feat_groups)
        multipliers = self._get_feat_step_multipliers if self.feat_step_adapt_active else None
        if not self.geometry_step_adapt_active:
            return get_step_symbols(xy, scale, rot, feat, self.quant_steps, multipliers)
        return get_step_symbols(xy, scale, rot, feat, self.quant_steps, multipliers, self._get_rot_step_exponents, self._get_pos_step_exponents)

    def _dequantize_symbols(self, symbols):
        """
        Gaussian parameters (scale in pixels) from their quantization indices.
        """
        multipliers = self._get_feat_step_multipliers if self.feat_step_adapt_active else None
        return dequantize_symbols(symbols, log_scale=self.quant_mode != "minmax", feat_step_multipliers=multipliers)

    def _get_feat_step_multipliers(self, geometry_symbols):
        """
        Per-Gaussian multipliers [P, C] of the feature steps (compress.quantization.feat_step_multipliers with feat_step_adapt, and
        feat_step_gamma or for U and V uv_step_gamma; "weight" from the rasterizer's top-K weights). The multipliers of the last geometry are
        cached.
        """
        indices = [q for q, _, _ in geometry_symbols]
        cache = self.feat_step_multiplier_cache
        if cache is not None and all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(cache[0], indices)):
            return cache[1]

        def get_feature_weights(xy, scale, rot):
            no_feat = torch.zeros(len(xy), 1, dtype=self.dtype, device=self.device)  # The top-K lists do not depend on the features
            return feature_weights(self._iter_topk_weights(xy, scale, rot, no_feat), len(xy), self.eps, self.device)

        multipliers = feat_step_multipliers(geometry_symbols, self.feat_step_adapt, self.feat_step_gammas, get_feature_weights).to(self.dtype)
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
        self.init_scale_knn = args.init_scale_knn
        self.init_aspect = args.init_aspect
        if self.init_scale_knn < 0.0 or self.init_aspect < 1.0:
            raise ValueError(f"Initial scale factor must be non-negative and aspect at least 1 but got: {self.init_scale_knn}, {self.init_aspect}")
        self.structure_tensor = None  # Of the target [3, H*W], computed when init_aspect first needs it
        self.init_fine_ratio = args.init_fine_ratio
        if not 0.0 <= self.init_fine_ratio < 1.0:
            raise ValueError(f"Fine initialization ratio must be in [0, 1) but got: {self.init_fine_ratio}")
        self.add_color_init = args.add_color_init
        if self.add_color_init not in ["residual", "target"]:
            raise ValueError(f"Unsupported color initialization '{self.add_color_init}', valid values include 'residual' and 'target'")
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
        return fixed_length_bits(self._get_symbols(self.xy, *self._canonicalize(self._get_scale(), self.rot), self.feat))

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
        # Rate term (compress.rate) with qp and quant_mode "step": rate_weight times the pruning lambda, converted from squared error per bit
        # to L1 loss per bit
        if args.rate_weight < 0.0:
            raise ValueError(f"Rate weight must be non-negative but got: {args.rate_weight:g}")
        self.rate_weight = args.rate_weight if args.qp > 0.0 and self.quant_mode == "step" else 0.0
        if self.rate_weight > 0.0 and self.l1_loss_ratio <= 1e-7:
            raise ValueError("rate_weight requires an L1 loss (set rate_weight 0 without one)")
        if self.rate_weight > 0.0 and not self.canonical_rot:
            raise ValueError("rate_weight requires canonical_rot (the rotation rate is that of canonical rotations; set rate_weight 0 without it)")
        self.rate_lambda = args.prune_lambda
        self.rate_color_refs = get_channel_refs(self.input_channels, self.color_transform)[3]
        # Weights of the feature channels' bits: in YUV mode, U and V by rate_uv_weight (negative: uv_weight, so that chroma is traded off
        # against its weight in the losses as in RGB mode)
        self.rate_channel_weights = torch.ones(self.feat_dim, dtype=self.dtype, device=self.device)
        if self.color_space == "yuv":
            self.rate_channel_weights[self.uv_channels] = args.rate_uv_weight if args.rate_uv_weight >= 0.0 else self.uv_weight
        self.rate_model, self.rate_key, self.rate_loss = None, None, None
        self.rate_inv_rot_step = 1.0 / self.quant_steps[2].item() if self.rate_weight > 0.0 else None

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
            # Position: init_fine_ratio of the Gaussians on fine blobs, the others by init_mode
            images = torch.pow(self.gt_images, 1.0/self.gamma)
            fine_pixels, fine_scale = find_fine_blobs(images, round(self.init_fine_ratio * self.num_gaussians))
            num = self.num_gaussians - len(fine_pixels)
            if self.init_mode == 'gradient':
                self._compute_gmap()
                xy = self._sample_pos(prob=self.image_gradients, num=num)
            elif self.init_mode == 'saliency':
                self.smap_filter_size = args.smap_filter_size
                self._compute_smap(path="models")
                xy = self._sample_pos(prob=self.saliency, num=num)
            elif self.init_mode == 'local-std':
                xy = self._sample_pos(prob=local_std_density(images, args.init_std_gamma), num=num)
            else:
                xy = self.pixel_xy.detach().clone()[self._sample_pixels(num)]
            # Scale and rotation: the others' from their spacing among themselves, the fine Gaussians isotropic at their blob's scale
            scale, rot = self._get_init_shapes(xy, xy)
            if len(fine_pixels) > 0:
                xy = torch.cat([xy, self.pixel_xy[fine_pixels]], dim=0)
                fine_scale = fine_scale.unsqueeze(dim=1).expand(-1, 2)
                scale = torch.cat([scale, fine_scale if self.disable_inverse_scale else 1.0 / fine_scale], dim=0)
                rot = torch.cat([rot, torch.zeros(len(fine_pixels), 1, dtype=self.dtype, device=self.device)], dim=0)
            self.xy.copy_(xy)
            self.scale.copy_(scale)
            self.rot.copy_(rot)
            # Feature
            if not self.disable_color_init:
                self.feat.copy_(self._get_target_features(positions=self.xy).detach().clone())

    def _get_init_shapes(self, all_xy, new_xy):
        """
        Initial scale parameters [M, 2] and rotations [M, 1] of the Gaussians at new_xy [M, 2], which are among all Gaussians all_xy [N, 2]
        (utils.init_utils.initial_shapes with init_scale, init_scale_knn, and init_aspect).
        """
        if self.init_aspect > 1.0 and self.structure_tensor is None:
            images = torch.pow(self.gt_images, 1.0/self.gamma)
            self.structure_tensor = compute_structure_tensor(images, STRUCTURE_TENSOR_SIGMA).reshape(3, -1)
        scale, rot = initial_shapes(all_xy, new_xy, self.img_h, self.img_w, self.init_scale, self.init_scale_knn, self.init_aspect,
                                    self.structure_tensor)
        return (scale if self.disable_inverse_scale else 1.0 / scale), rot

    def _sample_pixels(self, num, prob=None):
        """
        Indices of num distinct pixels on the GPU, drawn uniformly or with the probabilities prob [H*W] (successive sampling without
        replacement, as np.random.choice).
        """
        if prob is None:
            return torch.randperm(self.num_pixels, device=self.device)[:num]
        return torch.multinomial(torch.as_tensor(prob, device=self.device), num, replacement=False)

    def _sample_pos(self, prob, num):
        num_random = round(self.init_random_ratio*num)
        selected_random = self._sample_pixels(num_random)
        selected_other = self._sample_pixels(num-num_random, prob=prob)
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
            self.geometry_step_adapt_active = self.geometry_step_adapt
            self.canonical_rot_active = self.canonical_rot and final  # Intermediate saves keep the axes that the optimizer state belongs to
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
        if not final:  # Optimization continues with the uniform feature, rotation, and position steps
            self.feat_step_adapt_active = False
            self.geometry_step_adapt_active = False

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

    def _update_rate_model(self):
        """
        Fixed inputs of the rate term (compress.rate.rate_model: neighbors in bitstream order, Laplace scales per feature channel and per aspect
        class of the rotation) from the current quantized Gaussians, written into the existing buffers while the number of Gaussians is
        unchanged, so that a captured optimization step uses them.
        """
        with torch.no_grad():
            symbols = self._get_symbols(self.xy, *self._canonicalize(self._get_scale(), self.rot), self.feat)
            model = rate_model(symbols[0][0].cpu().numpy().astype(np.int64), self._dequantize_symbols(symbols)[3], dequantize(*symbols[1]),
                               symbols[2][0][:, 0], self.rate_color_refs, self.feat_prediction_refs, self.quant_steps[3],
                               self.rate_channel_weights, self.dtype, self.device, self.pos_refine_bits)
            if self.rate_model is not None and self.rate_model.neighbors.shape == model.neighbors.shape:
                for buffer, value in zip(self.rate_model, model):
                    if isinstance(buffer, torch.Tensor):  # Not the channel references, which never change
                        buffer.copy_(value)
            else:
                self.rate_model = model
            self.rate_key = (self.xy,)  # In a tuple, as a Parameter attribute would be registered as a parameter of the model

    def _estimate_bits(self):
        """
        Fast estimate of the coded size in bits of the quantized Gaussians, without the bitstream header.
        """
        with torch.no_grad():
            symbols = self._get_symbols(self.xy, *self._canonicalize(self._get_scale(), self.rot), self.feat)
            return estimate_symbol_bits(symbols, get_channel_refs(self.input_channels, self.color_transform))

    def _write_bitstream(self, path):
        """
        Encode the quantized Gaussians into a bitstream file, then decode it and check that it reproduces the quantized parameters (except in
        real-time mode). Returns the size of the bitstream in bytes.
        """
        with torch.no_grad():
            symbols = self._get_symbols(self.xy, *self._canonicalize(self._get_scale(), self.rot), self.feat)
            attributes = symbols_to_numpy(symbols)
            meta = bitstream_meta(self.img_h, self.img_w, self.quant_mode, self.color_space, self.feat_prediction, self.feat_step_adapt,
                                  self.feat_step_gamma, self.uv_step_gamma, self._geometry_meta())
            channel_refs = get_channel_refs(self.input_channels, self.color_transform)
            coded = list(attributes)
            q_xy, q_scale, q_rot = attributes[0][0], attributes[1][0], attributes[2][0]
            _, order, _ = bitstream_order(q_xy, self.pos_refine_bits)
            if self.scale_prediction:  # From the positions that a decoder has before the scales (coarse with two-stage positions)
                predicted = predicted_sums(q_xy >> self.pos_refine_bits, self.scale_step)
                coded[1] = (scale_residuals(q_scale, predicted),) + attributes[1][1:]
            refine_exponents = None
            if self.geometry_step_adapt:  # Rotation indices divided by their per-Gaussian steps, and the position refinement exponents
                if not self.geometry_step_adapt_active:
                    raise RuntimeError("A bitstream with per-Gaussian rotation or position steps requires them in the quantization")
                rot_exponents = self._get_rot_step_exponents(symbols[1][0]).cpu().numpy()[:, None]
                if np.any(q_rot & ((1 << rot_exponents) - 1)):
                    raise RuntimeError("Rotation indices must be multiples of their per-Gaussian steps")
                coded[2] = (q_rot >> rot_exponents,) + attributes[2][1:]
                refine_exponents = self._get_pos_step_exponents(symbols[1][0], symbols[2][0][:, 0]).cpu().numpy()
            if self.feat_prediction:
                # Residuals of the features predicted in bitstream order (the order of encode_bitstream), which include the reference
                # subtraction of the color transform
                q_feat, feat_steps, feat_origins = attributes[3]
                residuals = np.empty_like(q_feat)
                residuals[order] = predict_residuals(q_feat[order], feat_steps, self._get_feat_step_exponents(symbols[:3])[order],
                                                     attributes[0][0][order], channel_refs[3], self.feat_prediction_refs)
                coded[3], channel_refs[3] = (residuals, feat_steps, feat_origins), None
            data, order, attribute_bytes = encode_bitstream(coded, meta, codec=self.bitstream_codec, channel_refs=channel_refs,
                                                            refine_bits=self.pos_refine_bits, refine_exponents=refine_exponents)
            with open(path, "wb") as file:
                file.write(data)
            if not self.realtime:
                with open(path, "rb") as file:
                    decoded_symbols = self._decode_bitstream(file.read())
                decoded = [q.cpu().numpy() for q, _, _ in decoded_symbols]
                if not all(np.array_equal(q[order], q_decoded) for (q, _, _), q_decoded in zip(attributes, decoded)):
                    raise RuntimeError(f"Decoding '{path}' does not reproduce the quantization indices")
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

    def _geometry_meta(self):
        """
        The geometry options of the bitstream metadata (compress.quantization.bitstream_meta), None if all are off.
        """
        if not (self.scale_prediction or self.geometry_step_adapt):
            return None
        rot_step = (self.rot_step_aspect, self.rot_step_gamma) if self.rot_step_adapt != "none" else None
        return self.scale_prediction, rot_step, self.pos_step_gamma if self.pos_step_adapt != "none" else None

    def _reconstruct_geometry(self, attributes):
        """
        Reconstruction of the geometry in decode_bitstream (bitstream order, numpy): the log2-scale indices from their prediction
        (scale_prediction, from the positions decoded so far: the coarse ones of two-stage positions), the rotation indices from their values
        divided by the per-Gaussian steps, and the refinement exponents of two-stage positions.
        """
        (q_xy, *_), (q_scale, *scale_params), (q_rot, *rot_params) = attributes[:3]
        if self.scale_prediction:
            q_scale = reconstruct_scales(q_scale, predicted_sums(q_xy >> self.pos_refine_bits, self.scale_step))
        refine_exponents = None
        if self.geometry_step_adapt:
            scale_tensor = torch.from_numpy(q_scale).to(self.device)
            q_rot = q_rot << self._get_rot_step_exponents(scale_tensor).cpu().numpy()[:, None]
            refine_exponents = self._get_pos_step_exponents(scale_tensor, torch.from_numpy(q_rot[:, 0]).to(self.device)).cpu().numpy()
        return [attributes[0], (q_scale, *scale_params), (q_rot, *rot_params)] + list(attributes[3:]), refine_exponents

    def _decode_bitstream(self, data):
        """
        Quantization symbols (GPU tensors) of a bitstream of _write_bitstream, with the Gaussians in bitstream order.
        """
        decoded, _ = decode_bitstream(data, reconstruct=self._reconstruct_geometry)
        symbols = [[torch.from_numpy(x).to(dtype=self.dtype, device=self.device) for x in attribute] for attribute in decoded]
        if self.feat_prediction:
            residuals, feat_steps, _ = decoded[3]
            color_refs = get_channel_refs(self.input_channels, self.color_transform)[3]
            indices = reconstruct_indices(residuals, feat_steps, self._get_feat_step_exponents(symbols[:3]), decoded[0][0], color_refs,
                                          self.feat_prediction_refs)
            symbols[3][0] = torch.from_numpy(indices).to(dtype=self.dtype, device=self.device)
        return symbols

    def _get_feat_step_exponents(self, geometry_symbols):
        """
        Exponents d [P, C] (int64) of the per-Gaussian feature step multipliers 2^(d/4) (zero without feat_step_adapt) of the geometry symbols.
        """
        if not self.feat_step_adapt_active:
            return np.zeros((len(geometry_symbols[0][0]), self.feat_dim), dtype=np.int64)
        multipliers = self._get_feat_step_multipliers(geometry_symbols).to(torch.float64)
        return torch.round(FEAT_STEP_ADAPT_DIVISIONS * torch.log2(multipliers)).to(torch.int64).cpu().numpy()

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

    def forward(self, img_h, img_w, tile_bounds, upsample_ratio=None, benchmark=False, isect_capacity=None, weighted=False):
        """
        With isect_capacity, the tile intersections go to buffers of that size without reading their number back to the CPU (CUDA graph
        capture), and their peak number is tracked in isect_peak. With weighted, the image is in the space of the training losses (YUV mode:
        loss_transform): rendering is linear in the features, so transforming the features transforms the image.
        """
        xy, scale, rot, feat = self._get_params(upsample_ratio=upsample_ratio)
        if weighted and self.loss_transform is not None:
            feat = feat @ self.loss_transform.T
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
            scale, rot = self._canonicalize(scale, rot, quarter_turns=self.canonical_rot_active)
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
            if self.rate_weight > 0.0 and (self.rate_key is None or self.rate_key[0] is not self.xy):  # New Gaussians, before a capture
                self._update_rate_model()
            if self.cuda_graph:
                self._graph_step()
            else:
                self.optimizer.zero_grad()
                # Rendering
                images, render_time = self.forward(self.img_h, self.img_w, self.tile_bounds, weighted=True)
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
        if self.rate_weight > 0.0:
            with torch.no_grad():
                self._update_rate_model()
                feat_bits, rot_bits = estimated_bits(self.feat, self.rot, self.rate_model, self.rate_inv_rot_step)
            self.worklog.info(f"Rate term (weight {self.rate_weight:g}): estimated bits per Gaussian of the features "
                              f"{feat_bits.item()/self.num_gaussians:.2f}, rotation by aspect class {rot_bits.item()/self.num_gaussians:.2f}")
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
            images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds, weighted=True)
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
            images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds, isect_capacity=capacity, weighted=True)
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
        """
        Training losses of images rendered in the space of the losses (forward with weighted) against the target in that space.
        """
        target_images = self.loss_target_images
        if self.l1_loss_ratio > 1e-7 and self.ssim_loss_ratio > 1e-7 and self.l2_loss_ratio <= 1e-7:
            # The default losses from one forward and one backward kernel of fused-ssim; the gradient equals that of F.l1_loss plus
            # fused_ssim up to float rounding
            l1, ssim = fused_l1_ssim(images.unsqueeze(0), target_images.unsqueeze(0), padding="same")
            self.l1_loss = self.l1_loss_ratio * l1
            self.ssim_loss = self.ssim_loss_ratio * (1 - ssim)
            self.l2_loss = None
            self.total_loss = self.l1_loss + self.ssim_loss
        else:
            self.total_loss = 0
            if self.l1_loss_ratio > 1e-7:
                self.l1_loss = self.l1_loss_ratio * F.l1_loss(images, target_images)
                self.total_loss += self.l1_loss
            else:
                self.l1_loss = None
            if self.l2_loss_ratio > 1e-7:
                self.l2_loss = self.l2_loss_ratio * F.mse_loss(images, target_images)
                self.total_loss += self.l2_loss
            else:
                self.l2_loss = None
            if self.ssim_loss_ratio > 1e-7:
                self.ssim_loss = self.ssim_loss_ratio * (1 - fused_ssim(images.unsqueeze(0), target_images.unsqueeze(0)))
                self.total_loss += self.ssim_loss
            else:
                self.ssim_loss = None
        if self.rate_weight > 0.0:
            # Lambda is the increase of the squared error (8-bit units, summed over pixels) per bit; near the current mean absolute error,
            # a squared error increase dSSE changes the L1 loss by dSSE / (2 255^2 pixels L1)
            self.rate_loss = rate_loss(self.feat, self.rot, self.l1_loss, self.rate_model, self.rate_inv_rot_step, self.l1_loss_ratio,
                                       self.rate_weight * self.rate_lambda / (2.0 * 255.0**2 * self.num_pixels))
            self.total_loss = self.total_loss + self.rate_loss

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
            if self.rate_weight > 0.0:
                self._update_rate_model()  # The Gaussians have moved since the last pruning round
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
        sample_prob, diff_map = add_probabilities(images, torch.pow(self.gt_images, 1.0/self.gamma), self.rgb_loss_transform)
        selected = self._sample_pixels(add_num, prob=sample_prob)
        # New Gaussians
        new_xy = self.pixel_xy.detach().clone()[selected]
        new_scale, new_rot = self._get_init_shapes(torch.cat([self.xy.detach(), new_xy], dim=0), new_xy)
        colors = self.target_images if self.add_color_init == "target" else self._from_rgb(diff_map)
        new_feat = colors.permute(1, 2, 0).reshape(-1, self.feat_dim)[selected]
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

    def _get_pixel_topk(self, xy, scale, rot, feat):
        """
        Projected positions and conics of the Gaussians, and the top-K Gaussians of each pixel [H, W, K] as the rasterizer selects them (-1
        for empty entries).
        """
        xy, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(xy, scale, rot, self.img_h, self.img_w, self.tile_bounds)
        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)
        gaussian_ids_sorted, tile_bins = sort_intersects_by_tile(len(xy), num_intersects, xy, radii, cum_tiles_hit, self.tile_bounds)
        xy, conics = xy.contiguous(), conics.contiguous()
        _, pixel_topk = nd_rasterize_forward_topk_norm(self.tile_bounds, (self.block_w, self.block_h, 1), (self.img_w, self.img_h, 1),
                                                       gaussian_ids_sorted, tile_bins, xy, conics, feat.contiguous())
        return xy, conics, pixel_topk

    def _iter_topk_weights(self, xy, scale, rot, feat, chunk_size=2**18):
        """
        Top-K Gaussians of each pixel with their blending weights before normalization, in chunks of pixels to bound memory usage at high
        resolutions: yields the first pixel of the chunk, the ids [P, K] (empty entries as 0), valid [P, K], and the weights [P, K] (0 for
        empty entries).
        """
        xy, conics, pixel_topk = self._get_pixel_topk(xy, scale, rot, feat)
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

    def _get_loo_loss_increase(self, metric="l1"):
        """
        Increase of the loss when removing each Gaussian on its own (compress.pruning.loo_loss_increase), in YUV mode in the space of the
        losses (loss_transform), so that with uv_weight 1 the loss is that of RGB images and with uv_weight -> 0 the squared error is that of
        the luma (Y in every channel).
        """
        xy, scale, rot, feat = self._get_params()
        xy, conics, pixel_topk = self._get_pixel_topk(xy, scale, rot, feat)
        return loo_loss_increase(xy, conics, feat, pixel_topk, self.target_images, self.eps, metric, self.loss_transform)

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
        pruned, num_below, threshold = select_lambda_pruning(sse_increase, bits_per_gaussian, self.prune_lambda, self.prune_max_ratio)
        prune_num = len(pruned)
        round_info = (f"Round {self.prune_rounds_done:d}: {num_below:d} below lambda x {bits_per_gaussian:.1f} bits per Gaussian = "
                      f"{threshold:.1f} SSE")
        if prune_num > 0:
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
            score = contribution(lambda feat: self._render_features(feat, self.img_h, self.img_w, self.tile_bounds), self.num_gaussians,
                                 self.dtype, self.device)
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
        for name, param in zip(["xy", "scale", "rot", "feat"], keep_gaussians(self.optimizer, keep)):
            setattr(self, name, param)
        self.vis_feat = nn.Parameter(self.vis_feat.detach()[keep], requires_grad=False)
        self.num_gaussians -= remove_num
        self.worklog.info(
            f"Step: {self.step:d} | {action} {remove_num:d} Gaussians ({self.num_gaussians+remove_num:d} -> {self.num_gaussians:d}) | {score_info}")
        self.worklog.info("***********************************************")
