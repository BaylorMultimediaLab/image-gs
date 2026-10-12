import logging
import math
import os
import struct
import sys
import warnings
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_ssim
from lpips import LPIPS
from pytorch_msssim import MS_SSIM
from torchvision.transforms.functional import gaussian_blur

from compress import decode_bitstream, encode_bitstream
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

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips")


class GaussianSplatting2D(nn.Module):
    def __init__(self, args):
        super(GaussianSplatting2D, self).__init__()
        self.evaluate = args.eval
        set_random_seed(seed=args.seed)
        self.device = args.device
        self.dtype = torch.float32
        self._init_logging(args)
        self._init_target(args)
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
        self.worklog.info(f"Start {action} {args.num_gaussians:d} Gaussians for '{args.input_path}'")
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
            if not self.evaluate:
                path = f"{self.log_dir}/gt_upsample-{self.downsample_ratio:.1f}_res-{self.img_h_upsampled:d}x{self.img_w_upsampled:d}"
                self._separate_and_save_images(images=self.gt_images_upsampled, channels=self.input_channels, path=path)
        self.num_pixels = self.img_h * self.img_w
        if not self.evaluate:
            path = f"{self.log_dir}/gt_res-{self.img_h:d}x{self.img_w:d}"
            self._separate_and_save_images(images=self.gt_images, channels=self.input_channels, path=path)

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
        if self.quant_mode == "step" and min(all_steps) <= 0.0:
            raise ValueError(f"Quantization steps must be positive but got: {', '.join(f'{step:g}' for step in all_steps)}")
        self.pos_step = args.pos_step  # Pixels
        self.scale_step = args.scale_step  # Octaves (log2 of the scale in pixels)
        self.rot_step = math.radians(args.rot_step)
        self.feat_step = args.feat_step
        self.y_step = args.y_step
        self.uv_step = args.uv_step
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
        and rendering only depends on squared scales (scales are quantized in the log domain in "step" mode).
        """
        if self.wrap_rot:
            rot = torch.remainder(rot, math.pi)
        if self.quant_mode == "step":
            scale = scale.abs()
        return scale, rot

    def _get_symbols(self, xy, scale, rot, feat):
        """
        Quantization indices [P, C] of the canonicalized Gaussian parameters (scale in pixels), with the per-channel steps and origins [C]
        that dequantize them. Scales are quantized in pixels in "minmax" mode and in log2 pixels in "step" mode.
        """
        with torch.no_grad():
            if self.quant_mode == "step":
                pos_step = self.pos_step / torch.tensor([self.img_w, self.img_h], dtype=self.dtype, device=self.device)
                symbols = [quantize_step(xy, pos_step), quantize_step(torch.log2(scale), self.scale_step), quantize_step(rot, self.rot_step)]
            else:
                symbols = [quantize_minmax(xy, self.pos_bits), quantize_minmax(scale, self.scale_bits), quantize_minmax(rot, self.rot_bits)]
            q_feat, feat_steps, feat_origins = torch.empty_like(feat), feat.new_empty(self.feat_dim), feat.new_empty(self.feat_dim)
            for channels, bits, step in self._get_feat_groups():
                q, steps, origins = quantize_step(feat[:, channels], step) if self.quant_mode == "step" else quantize_minmax(feat[:, channels], bits)
                q_feat[:, channels], feat_steps[channels], feat_origins[channels] = q, steps, origins
        return symbols + [(q_feat, feat_steps, feat_origins)]

    def _dequantize_symbols(self, symbols):
        """
        Gaussian parameters (scale in pixels) from their quantization indices.
        """
        xy, scale, rot, feat = [dequantize(*symbol) for symbol in symbols]
        if self.quant_mode == "step":
            scale = torch.exp2(scale)
        return xy, scale, rot, feat

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
        self.feat_dim = sum(self.input_channels)
        self.feat = nn.Parameter(torch.rand(self.num_gaussians, self.feat_dim, dtype=self.dtype, device=self.device), requires_grad=True)
        self.vis_feat = nn.Parameter(torch.rand_like(self.feat), requires_grad=False)  # Only used for Gaussian ID visualization
        self._log_compression_rate()

    def _init_pruning(self, args):
        if not 0.0 <= args.prune_ratio < 1.0:
            raise ValueError(f"Pruning ratio must be in [0, 1) but got: {args.prune_ratio}")
        if not 0.0 <= args.relocate_ratio < 1.0:
            raise ValueError(f"Relocation ratio must be in [0, 1) but got: {args.relocate_ratio}")
        self.num_prune = round(args.prune_ratio * self.total_num_gaussians)
        self.final_num_gaussians = self.total_num_gaussians - self.num_prune
        self.num_pruned = 0
        self.relocate_ratio = args.relocate_ratio
        if self.evaluate:
            self.num_gaussians = self.final_num_gaussians  # Checkpoints only contain the Gaussians that survived pruning
            return
        if self.relocate_ratio > 0.0 and self.disable_prog_optim:
            raise ValueError("Relocation happens at densification steps and requires progressive optimization")
        if self.num_prune > 0 or self.relocate_ratio > 0.0:
            self.prune_score = args.prune_score
            if self.prune_score not in ["contribution", "loo"]:
                raise ValueError(f"Unsupported pruning score '{self.prune_score}', valid values include 'contribution' and 'loo'")
            if self.prune_score == "loo" and (args.disable_tiles or args.disable_topk_norm):
                raise ValueError("Pruning score 'loo' requires tile-based rendering with top-K normalization")
        if self.num_prune > 0:
            if args.prune_times < 1:
                raise ValueError(f"Number of pruning rounds must be positive but got: {args.prune_times:d}")
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
        return self.num_pruned == self.num_prune and self.num_gaussians == self.final_num_gaussians

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
        self.worklog.info(f"Compressed: {bytes_compressed/1e3:.2f} KB | {bpp_compressed:.3f} bpp | {bppc_compressed:.3f} bppc")
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
        self.optimizer = torch.optim.Adam([{'params': self.xy, 'lr': self.pos_lr},
                                           {'params': self.scale, 'lr': self.scale_lr},
                                           {'params': self.rot, 'lr': self.rot_lr},
                                           {'params': self.feat, 'lr': self.feat_lr}])
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
                selected = np.random.choice(self.num_pixels, self.num_gaussians, replace=False, p=None)
                self.xy.copy_(self.pixel_xy.detach().clone()[selected])
            # Scale
            self.scale.fill_(self.init_scale if self.disable_inverse_scale else 1.0/self.init_scale)
            # Feature
            if not self.disable_color_init:
                self.feat.copy_(self._get_target_features(positions=self.xy).detach().clone())

    def _sample_pos(self, prob):
        num_random = round(self.init_random_ratio*self.num_gaussians)
        selected_random = np.random.choice(self.num_pixels, num_random, replace=False, p=None)
        selected_other = np.random.choice(self.num_pixels, self.num_gaussians-num_random, replace=False, p=prob)
        return torch.cat([self.pixel_xy.detach().clone()[selected_random], self.pixel_xy.detach().clone()[selected_other]], dim=0)

    def _compute_gmap(self):
        gy, gx = compute_image_gradients(np.power(self.gt_images.detach().cpu().clone().numpy(), 1.0/self.gamma))
        g_norm = np.hypot(gy, gx).astype(np.float32)
        g_norm = g_norm / g_norm.max()
        save_image(g_norm, f"{self.log_dir}/gmap_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}")
        g_norm = np.power(g_norm.reshape(-1), 2.0)
        self.image_gradients = g_norm / g_norm.sum()
        self.worklog.info("Image gradient map successfully saved")
        self.worklog.info("***********************************************")

    def _compute_smap(self, path):
        smap = get_smap(torch.pow(self.gt_images.detach().clone(), 1.0/self.gamma), path, self.smap_filter_size)
        save_image(smap, f"{self.log_dir}/smap_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}")
        self.saliency = (smap / smap.sum()).reshape(-1)
        self.worklog.info("Saliency map successfully saved")
        self.worklog.info("***********************************************")

    def _get_target_features(self, positions):
        with torch.no_grad():
            # target_images [1, C, H, W]; positions [1, 1, P, 2]; top-left [-1, -1]; bottom-right [1, 1]
            target_features = F.grid_sample(self.target_images.unsqueeze(0), positions[None, None, ...] * 2.0 - 1.0, align_corners=False)
            target_features = target_features[0, :, 0, :].permute(1, 0)  # [P, C]
        return target_features

    def _load_model(self):
        if self.ckpt_file != "":
            ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_file)
        else:
            latest_step = get_latest_ckpt_step(self.ckpt_dir)
            if latest_step == -1:
                raise FileNotFoundError(f"No checkpoint found in '{self.ckpt_dir}'")
            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_step-{latest_step:d}.pt")
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
            self._quantize()
            if self.quant_mode == "step":
                self._log_compressed_size(self._get_fixed_length_bits())
        psnr, ssim = self._evaluate(log=False, upsample=False)
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

    def _write_bitstream(self, path):
        """
        Encode the quantized Gaussians into a bitstream file, then decode it and check that it reproduces the quantized parameters.
        Returns the size of the bitstream in bytes.
        """
        with torch.no_grad():
            symbols = self._get_symbols(self.xy, *self._canonicalize(self._get_scale(), self.rot), self.feat)
            attributes = [(q.cpu().numpy().astype(np.int64), steps.cpu().numpy().astype(np.float64), origins.cpu().numpy().astype(np.float64))
                          for q, steps, origins in symbols]
            meta = struct.pack("<IIBB", self.img_h, self.img_w, ["minmax", "step"].index(self.quant_mode), ["rgb", "yuv"].index(self.color_space))
            data, order, attribute_bytes = encode_bitstream(attributes, meta, codec=self.bitstream_codec, channel_refs=self._get_channel_refs())
            with open(path, "wb") as file:
                file.write(data)
            with open(path, "rb") as file:
                decoded, _ = decode_bitstream(file.read())
            if not all(np.array_equal(q[order], q_decoded) for (q, _, _), (q_decoded, _, _) in zip(attributes, decoded)):
                raise RuntimeError(f"Decoding '{path}' does not reproduce the quantization indices")
            decoded_symbols = [[torch.from_numpy(x).to(dtype=self.dtype, device=self.device) for x in attribute] for attribute in decoded]
            images = self._rasterize(*self._dequantize_symbols(decoded_symbols), self.img_h, self.img_w, self.tile_bounds)
            psnr, _ = self._evaluate(log=False, upsample=False)
            decoded_psnr, _ = self._evaluate(log=False, upsample=False, images=self._to_rgb(images))
        num_bits = 8.0 * len(data)
        fixed_length_bits = self._get_fixed_length_bits()
        names = ["Position", "Scale", "Rotation", "Feature"]
        bits_per_gaussian = [f"{name} {8.0*num_bytes/self.num_gaussians:.2f}" for name, num_bytes in zip(names, attribute_bytes)]
        self.worklog.info(f"Bitstream ({self.bitstream_codec}): {len(data)/1e3:.2f} KB | {num_bits/self.num_pixels:.3f} bpp | "
                          f"{num_bits/fixed_length_bits*100.0:.2f}% of fixed-length codes | Saved to '{path}'")
        self.worklog.info(f"Bits per Gaussian: {num_bits/self.num_gaussians:.2f} (fixed-length: {fixed_length_bits/self.num_gaussians:.2f}) | "
                          f"{' | '.join(bits_per_gaussian)} | Header {len(data)-sum(attribute_bytes):.0f} bytes")
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

    def forward(self, img_h, img_w, tile_bounds, upsample_ratio=None, benchmark=False):
        xy, scale, rot, feat = self._get_params(upsample_ratio=upsample_ratio)
        begin = perf_counter()
        tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
        xy, radii, conics, num_tiles_hit = tmp
        if not self.disable_tiles:
            enable_topk_norm = not self.disable_topk_norm
            tmp = xy, radii, conics, num_tiles_hit, feat, img_h, img_w, self.block_h, self.block_w, enable_topk_norm
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
        self.lpips_final, self.flip_final, self.msssim_final = 1.0, 1.0, 0.0

        self.step = 0
        with torch.no_grad():
            self._log_images(log_final=False, plot_gaussians=self.vis_gaussians)
        for step in range(self.start_step, self.max_steps+1):
            self.step = step
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
            if self.max_time > 0.0:
                torch.cuda.synchronize()  # Charge each step its GPU time
            self.total_time_accum += (perf_counter() - begin + render_time)
            # Logging
            terminate = False
            with torch.no_grad():
                if self.step % self.eval_steps == 0:
                    self._evaluate(log=True, upsample=False)
                    if not self.disable_lr_schedule and self.max_time == 0.0 and self._is_final_num_gaussians():
                        terminate = self._lr_schedule()
                if self.step % self.save_image_steps == 0:
                    self._log_images(log_final=False, plot_gaussians=self.vis_gaussians)
                if self.step % self.save_ckpt_steps == 0 and self._is_final_num_gaussians():
                    self._save_model()
                begin = perf_counter()
                # Pruned Gaussians are not added back
                if not self.disable_prog_optim and self.step % self.add_steps == 0 and self.num_gaussians + self.num_pruned < self.total_num_gaussians:
                    # Relocation: replace the Gaussians with the lowest score by additional error-guided Gaussians
                    relocate_num = round(self.relocate_ratio * self.num_gaussians)
                    if relocate_num > 0:
                        self._remove_gaussians(relocate_num, action="Relocating", plot_gaussians=self.vis_gaussians)
                    self._add_gaussians(self.max_add_num + relocate_num, plot_gaussians=self.vis_gaussians)
                if self.num_pruned < self.num_prune and self.step > self.prune_start_step and self._is_prune_step():
                    self._prune_gaussians(self.max_prune_num, plot_gaussians=self.vis_gaussians)
                if self.max_time > 0.0:
                    self.total_time_accum += perf_counter() - begin  # Densification and pruning are part of the time budget
                    terminate = self._time_schedule() or terminate
                if terminate:
                    break
        with torch.no_grad():
            self._log_images(log_final=True, plot_gaussians=self.vis_gaussians)
            self._save_model(final=True)
        self.worklog.info("Optimization completed")
        self.worklog.info("***********************************************")
        self.worklog.info(f"Mean scale: {self._get_scale().mean().item():.4f} (pixel) | {self.scale.mean().item():.4f} (raw)")
        self.worklog.info("***********************************************")
        return self.psnr_curr, self.ssim_curr

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
        Whether the next pruning round is due: every prune_steps steps, or with a time budget, every prune_time_ratio of the budget.
        """
        if self.max_time > 0.0:
            rounds_done = math.ceil(self.num_pruned / self.max_prune_num)
            return self.total_time_accum >= (rounds_done + 1) * self.prune_time_ratio * self.max_time
        return (self.step - self.prune_start_step) % self.prune_steps == 0

    def _time_schedule(self):
        """
        Schedule of time-budgeted optimization: a single learning rate decay after decay_time_ratio of the budget, and termination once the
        budget is used up. Returns whether to terminate.
        """
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
        sample_prob = (error_map / error_map.sum()).cpu().numpy()
        selected = np.random.choice(self.num_pixels, add_num, replace=False, p=sample_prob)
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
        self.optimizer = torch.optim.Adam([{'params': self.xy, 'lr': self.pos_lr},
                                           {'params': self.scale, 'lr': self.scale_lr},
                                           {'params': self.rot, 'lr': self.rot_lr},
                                           {'params': self.feat, 'lr': self.feat_lr}])
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

    def _get_loo_loss_increase(self, chunk_size=2**18):
        """
        Increase of the L1 loss when removing each Gaussian on its own (leave-one-out), computed from a single rendering pass.
        Removing Gaussian k from the top-K list of a pixel changes its value from S_f / (S_w + eps) to (S_f - w_k f_k) / (S_w - w_k + eps),
        where S_w and S_f are the sums of the top-K weights and weighted features. Ignores the Gaussian that would move into the top-K list.
        """
        xy, scale, rot, feat = self._get_params()
        xy, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(xy, scale, rot, self.img_h, self.img_w, self.tile_bounds)
        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)
        _, _, _, gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(
            self.num_gaussians, num_intersects, xy, radii, cum_tiles_hit, self.tile_bounds)
        _, pixel_topk = nd_rasterize_forward_topk_norm(self.tile_bounds, (self.block_w, self.block_h, 1), (self.img_w, self.img_h, 1),
                                                       gaussian_ids_sorted, tile_bins, xy.contiguous(), conics.contiguous(), feat.contiguous())
        pixel_topk = pixel_topk.view(-1, self.topk).long()  # [H*W, K], -1 for empty entries
        rows, cols = torch.meshgrid(torch.arange(self.img_h, device=self.device), torch.arange(self.img_w, device=self.device), indexing="ij")
        pixels = torch.stack([cols, rows], dim=-1).reshape(-1, 1, 2).to(self.dtype)  # Pixel coordinates used by the rasterizer
        targets = self.target_images.permute(1, 2, 0).reshape(-1, self.feat_dim)
        loss_increase = torch.zeros(self.num_gaussians, dtype=self.dtype, device=self.device)
        # Process pixels in chunks to bound memory usage at high resolutions
        for begin in range(0, len(pixel_topk), chunk_size):
            ids = pixel_topk[begin:begin+chunk_size]
            valid = ids >= 0
            ids = ids.clamp(min=0)
            delta = xy[ids] - pixels[begin:begin+chunk_size]  # [P, K, 2]
            conic = conics[ids]  # [P, K, 3]
            sigma = 0.5 * (conic[..., 0] * delta[..., 0]**2 + conic[..., 2] * delta[..., 1]**2) + conic[..., 1] * delta[..., 0] * delta[..., 1]
            weights = torch.exp(-sigma) * valid  # [P, K]
            weighted_feat = weights.unsqueeze(-1) * feat[ids]  # [P, K, C]
            sum_weights = weights.sum(dim=1, keepdim=True)  # [P, 1]
            sum_feat = weighted_feat.sum(dim=1)  # [P, C]
            target = targets[begin:begin+chunk_size]
            error = (sum_feat / (sum_weights + self.eps) - target).abs().sum(dim=-1, keepdim=True)  # [P, 1]
            loo_values = (sum_feat.unsqueeze(1) - weighted_feat) / (sum_weights - weights + self.eps).unsqueeze(-1)  # [P, K, C]
            loo_error = (loo_values - target.unsqueeze(1)).abs().sum(dim=-1)  # [P, K]
            loss_increase.index_add_(0, ids[valid], (loo_error - error)[valid])
        return loss_increase / targets.numel()

    def _prune_gaussians(self, prune_num, plot_gaussians=False):
        prune_num = min(prune_num, self.max_prune_num, self.num_prune-self.num_pruned)
        if prune_num <= 0:
            return
        self._remove_gaussians(prune_num, action="Pruning", plot_gaussians=plot_gaussians)
        self.num_pruned += prune_num

    def _remove_gaussians(self, remove_num, action, plot_gaussians=False):
        """
        Remove the Gaussians with the lowest pruning score.
        """
        score = self._get_contribution() if self.prune_score == "contribution" else self._get_loo_loss_increase()
        pruned = torch.topk(score, remove_num, largest=False).indices
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
        if self.prune_score == "contribution":
            score_info = f"Pruned contribution: {100.0*score[pruned].sum().item()/score.sum().item():.2f}%"
        else:
            score_info = f"Predicted L1 increase: {score[pruned].sum().item():.5f}"
        self.worklog.info(
            f"Step: {self.step:d} | {action} {remove_num:d} Gaussians ({self.num_gaussians+remove_num:d} -> {self.num_gaussians:d}) | {score_info}")
        self.worklog.info("***********************************************")
