"""
Differentiable rate estimates of the Gaussian parameters for a rate term in the training loss: code lengths in bits under discretized
Laplace distributions of the features as residuals of their prediction (as compress.prediction codes them) and of the canonical rotations
around zero with one scale per aspect-ratio class (the context a coder would take from the decoded scales). The neighbors, the aspect
classes, and the distribution scales come from the current parameters and are held fixed between updates (rate_model). The training loss
takes the rate from fused CUDA kernels (rate_loss, gsplat/cuda/csrc/rate.cu), which compute rate_bits in three kernels, so that the term
costs little in a captured optimization step.
"""
import math
from typing import NamedTuple

import numpy as np
import torch

from compress import bitstream_order
from compress.prediction import NUM_NEIGHBORS, causal_neighbors
from gsplat.cuda import rate_loss_backward, rate_loss_forward

# Aspect-ratio classes of the rotation: boundaries of |log2 s0 - log2 s1| in octaves (aspect ratios 1.19, 1.41, 2, 4)
ASPECT_BOUNDARIES = (0.25, 0.5, 1.0, 2.0)
MIN_SCALE = 0.3  # Lower bound of the distribution scales in quantization steps
MIN_CCLM_NEIGHBORS = 3  # As compress/csrc/prediction.cpp
MAX_SLOPE = 4.0


class RateModel(NamedTuple):
    """
    Fixed inputs of the rate estimates (rate_model), for P Gaussians with C feature channels and K neighbors each.
    """
    neighbors: torch.Tensor  # [P, K] int32, missing ones clamped to 0
    weights: torch.Tensor  # [P, K]: 1 / the number of neighbors, 0 for missing ones
    slope: torch.Tensor  # [P, C]: CCLM slopes, 0 without a prediction reference
    inv_steps: torch.Tensor  # [C]: inverse quantization steps
    inv_feat: torch.Tensor  # [C]: 1 / (scale ln 2) of the feature channels
    feat_coeffs: torch.Tensor  # [C]: inv_feat times the channel weights of the rate term
    inv_rot: torch.Tensor  # [P, 1]: 1 / (scale ln 2) of each Gaussian's rotation by its aspect class
    offsets: torch.Tensor  # [P + 1] int32: the Gaussians that predict from Gaussian j are sources[offsets[j]:offsets[j + 1]]
    sources: torch.Tensor  # [E] int32
    source_weights: torch.Tensor  # [E]: their weights of Gaussian j
    operands: tuple  # channel_operands
    color_refs: tuple  # Per channel, the channel subtracted from it before the prediction (-1: none)
    pred_channels: tuple  # Per channel, the reference channel of its CCLM term (itself without one)


def laplace_bits(x, scale):
    """
    Code length in bits of x (in quantization steps, continuous) under a Laplace distribution of the given scale (in steps) discretized to
    unit bins around the integers: -log2(F(x + 1/2) - F(x - 1/2)). Smooth, with a zero gradient at x = 0.
    """
    a = x.abs()
    inner = 1.0 - 0.5 * (torch.exp(-(0.5 + a) / scale) + torch.exp(-(0.5 - a).clamp(min=0.0) / scale))  # |x| < 1/2
    outer = -(a - 0.5) / scale + torch.log(0.5 * (1.0 - torch.exp(-1.0 / scale)))  # |x| >= 1/2, natural log of the probability
    return -torch.where(a < 0.5, torch.log(inner.clamp(min=1e-12)), outer) / math.log(2.0)


def bitstream_neighbors(q_xy, refine_bits=0):
    """
    The neighbors that compress.prediction predicts each Gaussian from (the NUM_NEIGHBORS nearest earlier ones in bitstream order, with
    refine_bits of two-stage positions), for the integer positions q_xy [P, 2] (numpy int64): indices [P, NUM_NEIGHBORS] (int64) into the
    Gaussians in their given order, -1 for none.
    """
    _, order, _ = bitstream_order(q_xy, refine_bits)
    sorted_neighbors = causal_neighbors(q_xy[order])
    neighbors = np.full((len(q_xy), NUM_NEIGHBORS), -1, dtype=np.int64)
    neighbors[order] = np.where(sorted_neighbors >= 0, order[sorted_neighbors.clip(min=0)], -1)
    return neighbors


def channel_operands(color_refs, pred_refs, device):
    """
    Reference channels as index tensors, created once: per channel, its color reference (compress.common.subtract_channels) and its
    prediction reference (compress.prediction), itself without one; None for the color references when there are none (YUV).
    """
    own = torch.arange(len(color_refs), device=device)
    color, pred = torch.tensor(color_refs, device=device), torch.tensor(pred_refs, device=device)
    color_idx = (torch.where(color >= 0, color, own), (color >= 0)) if (color >= 0).any() else None
    return color_idx, torch.where(pred >= 0, pred, own)


def coded_values(feat, color_idx):
    """
    Feature values [P, C] after the reference subtraction of channel_operands (none: the features).
    """
    return feat if color_idx is None else feat - feat[:, color_idx[0]] * color_idx[1]


def feature_residuals(feat, neighbors, weights, slope, operands, inv_steps):
    """
    Residuals [P, C] in quantization steps (inv_steps [C], the inverse steps) of the features [P, C] after reference subtraction and
    prediction as compress.prediction: the mean of the neighbors [P, K] (clamped to valid indices) with weights [P, K] (1 / their number, 0 for
    missing ones), plus the CCLM slope [P, C] (0 without a prediction reference) times the Gaussian's reference value minus the reference mean.
    """
    color_idx, pred_idx = operands
    values = coded_values(feat, color_idx)
    nv = values.index_select(0, neighbors.view(-1)).view(*neighbors.shape, -1)  # [P, K, C]
    mean = torch.bmm(weights[:, None, :], nv)[:, 0]
    return (values - mean - slope * (values[:, pred_idx] - mean[:, pred_idx])) * inv_steps


def canonical_rotation(rot, inv_rot_step):
    """
    Canonical rotations [P, 1] in quantization steps (as compress.quantization.canonicalize with quarter turns: within [-pi/4, pi/4]).
    """
    return (rot - torch.round(rot.detach() * (2.0 / math.pi)) * (0.5 * math.pi)) * inv_rot_step


def smooth_laplace_bits(x, inv_scale_ln2):
    """
    Code length in bits of x (in quantization steps) under a Laplace distribution with 1 / (scale ln 2) = inv_scale_ln2, without its constant
    log2(2 scale) and smoothed within half a step of zero (where the discretized distribution is flat): (sqrt(x^2 + 1/4) - 1/2) / (scale ln 2).
    For the gradients of a rate term: the same slope as laplace_bits away from zero.
    """
    return (torch.sqrt(x.square() + 0.25) - 0.5) * inv_scale_ln2


def predicting_sources(neighbors, dtype):
    """
    For each Gaussian, the Gaussians that predict from it, from the neighbors [P, K] (-1 for none): offsets [P + 1] and sources [E] (int32, in
    increasing order per Gaussian) and their weights of it [E] (1 / their number of neighbors), as RateModel.
    """
    num = len(neighbors)
    valid = neighbors >= 0
    rows = torch.arange(num, device=neighbors.device)[:, None].expand_as(neighbors)[valid]  # Row-major, so in increasing order
    targets = neighbors[valid]
    sources = rows[torch.argsort(targets, stable=True)]
    offsets = torch.zeros(num + 1, dtype=torch.int64, device=neighbors.device)
    offsets[1:] = torch.cumsum(torch.bincount(targets, minlength=num), dim=0)
    return offsets.to(torch.int32), sources.to(torch.int32), 1.0 / valid.sum(dim=1)[sources].to(dtype)


def rate_model(q_xy, feat, log2_scale, q_rot, color_refs, pred_refs, steps, channel_weights, dtype, device, refine_bits=0):
    """
    The fixed inputs of the rate estimates (RateModel) from the current quantized Gaussians: integer positions q_xy [P, 2] (numpy int64),
    dequantized features [P, C] (the uniform steps [C]), canonical log2 scales [P, 2], and rotation indices [P], with the weights of the feature
    channels' bits [C] and the refinement bits of two-stage positions (the bitstream order). Each distribution scale is the maximum-likelihood
    one (mean absolute residual index, at least MIN_SCALE).
    """
    with torch.no_grad():
        neighbors = torch.from_numpy(bitstream_neighbors(q_xy, refine_bits)).to(device)
        offsets, sources, source_weights = predicting_sources(neighbors, dtype)
        valid = (neighbors >= 0).to(dtype)
        count = valid.sum(dim=1, keepdim=True)
        weights = valid / count.clamp(min=1.0)
        neighbors = neighbors.clamp(min=0)
        operands = channel_operands(color_refs, pred_refs, device)
        values = coded_values(feat, operands[0])
        nv = values[neighbors] * valid[..., None]
        pred_idx = operands[1]
        mean = (nv * weights[..., None]).sum(dim=1)
        cov = (nv * nv[..., pred_idx] * weights[..., None]).sum(dim=1) - mean * mean[:, pred_idx]
        var = (nv[..., pred_idx].square() * weights[..., None]).sum(dim=1) - mean[:, pred_idx].square()
        has_ref = (torch.tensor(pred_refs, device=device) >= 0).to(dtype)
        slope = (cov / (var + steps.square())).clamp(-MAX_SLOPE, MAX_SLOPE) * has_ref * (count >= MIN_CCLM_NEIGHBORS)
        inv_steps = 1.0 / steps
        residuals = torch.round(feature_residuals(feat, neighbors, weights, slope, operands, inv_steps))
        inv_feat = 1.0 / (residuals.abs().mean(dim=0).clamp(min=MIN_SCALE) * math.log(2.0))
        boundaries = torch.tensor(ASPECT_BOUNDARIES, dtype=log2_scale.dtype, device=device)
        classes = torch.bucketize((log2_scale[:, 0] - log2_scale[:, 1]).abs(), boundaries)
        sums = torch.zeros(len(ASPECT_BOUNDARIES) + 1, dtype=dtype, device=device).index_add_(0, classes, q_rot.abs().to(dtype))
        counts = torch.bincount(classes, minlength=len(ASPECT_BOUNDARIES) + 1).to(dtype)
        class_scales = torch.where(counts > 0, sums / counts.clamp(min=1.0), q_rot.abs().to(dtype).mean()).clamp(min=MIN_SCALE)
        return RateModel(neighbors.to(torch.int32), weights, slope, inv_steps, inv_feat, inv_feat * channel_weights,
                         1.0 / (class_scales[classes][:, None] * math.log(2.0)), offsets, sources, source_weights, operands,
                         tuple(int(ref) for ref in color_refs), tuple(c if ref < 0 else int(ref) for c, ref in enumerate(pred_refs)))


def rate_bits(feat, rot, model, inv_rot_step):
    """
    Differentiable estimate of the coded size in bits (smooth_laplace_bits, without constants) of the features [P, C] (residuals of their
    prediction, each channel's bits times its channel weight) and of the rotations [P, 1] (canonical, by aspect class), with the fixed inputs of
    rate_model, in PyTorch operations: the definition of what the kernels of rate_loss compute. Returns both totals.
    """
    residuals = feature_residuals(feat, model.neighbors, model.weights, model.slope, model.operands, model.inv_steps)
    feat_bits = smooth_laplace_bits(residuals, model.feat_coeffs).sum()
    return feat_bits, smooth_laplace_bits(canonical_rotation(rot, inv_rot_step), model.inv_rot).sum()


class _RateLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, rot, l1_loss, model, inv_rot_step, l1_ratio, factor):
        loss, loss_weight, own_grad, rot_grad = rate_loss_forward(
            feat.contiguous(), rot.contiguous(), l1_loss.contiguous(), model.neighbors, model.weights, model.slope, list(model.color_refs),
            list(model.pred_channels), model.inv_steps, model.feat_coeffs, model.inv_rot, inv_rot_step, l1_ratio, factor)
        ctx.save_for_backward(loss_weight, own_grad, rot_grad)
        ctx.model, ctx.rot_shape = model, rot.shape
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        loss_weight, own_grad, rot_grad = ctx.saved_tensors
        model = ctx.model
        grad_feat, grad_rot = rate_loss_backward(grad_loss.contiguous(), loss_weight, own_grad, rot_grad, model.offsets, model.sources,
                                                 model.source_weights)
        return grad_feat, grad_rot.view(ctx.rot_shape), None, None, None, None, None


def rate_loss(feat, rot, l1_loss, model, inv_rot_step, l1_ratio, factor):
    """
    The rate term factor / max(l1_loss / l1_ratio, 1e-4) times the sum of rate_bits, with the L1 loss (a 0-dim tensor) treated as a constant,
    in fused CUDA kernels (float32): a 0-dim tensor, differentiable with respect to the features and rotations.
    """
    return _RateLoss.apply(feat, rot, l1_loss.detach(), model, inv_rot_step, l1_ratio, factor)


def estimated_bits(feat, rot, model, inv_rot_step):
    """
    Code lengths in bits (laplace_bits) of the features and rotations under the distributions of rate_model, for logging: both totals.
    """
    with torch.no_grad():
        residuals = feature_residuals(feat, model.neighbors, model.weights, model.slope, model.operands, model.inv_steps)
        feat_bits = laplace_bits(residuals, 1.0 / (model.inv_feat * math.log(2.0))).sum()
        rot_bits = laplace_bits(canonical_rotation(rot, inv_rot_step), 1.0 / (model.inv_rot * math.log(2.0))).sum()
    return feat_bits, rot_bits
