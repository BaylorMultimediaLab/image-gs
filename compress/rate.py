"""
Differentiable rate estimates of the Gaussian parameters for a rate term in the training loss: code lengths in bits under discretized
Laplace distributions of the features as residuals of their prediction (as compress.prediction codes them) and of the canonical rotations
around zero with one scale per aspect-ratio class (the context a coder would take from the decoded scales). The neighbors, the aspect
classes, and the distribution scales come from the current parameters and are held fixed between updates (rate_model), so that the estimate
fits into a captured optimization step with few kernels.
"""
import math

import numpy as np
import torch

from compress.common import morton_order
from compress.prediction import NUM_NEIGHBORS, causal_neighbors

# Aspect-ratio classes of the rotation: boundaries of |log2 s0 - log2 s1| in octaves (aspect ratios 1.19, 1.41, 2, 4)
ASPECT_BOUNDARIES = (0.25, 0.5, 1.0, 2.0)
MIN_SCALE = 0.3  # Lower bound of the distribution scales in quantization steps
MIN_CCLM_NEIGHBORS = 3  # As compress/csrc/prediction.cpp
MAX_SLOPE = 4.0


def laplace_bits(x, scale):
    """
    Code length in bits of x (in quantization steps, continuous) under a Laplace distribution of the given scale (in steps) discretized to
    unit bins around the integers: -log2(F(x + 1/2) - F(x - 1/2)). Smooth, with a zero gradient at x = 0.
    """
    a = x.abs()
    inner = 1.0 - 0.5 * (torch.exp(-(0.5 + a) / scale) + torch.exp(-(0.5 - a).clamp(min=0.0) / scale))  # |x| < 1/2
    outer = -(a - 0.5) / scale + torch.log(0.5 * (1.0 - torch.exp(-1.0 / scale)))  # |x| >= 1/2, natural log of the probability
    return -torch.where(a < 0.5, torch.log(inner.clamp(min=1e-12)), outer) / math.log(2.0)


def bitstream_neighbors(q_xy):
    """
    The neighbors that compress.prediction predicts each Gaussian from (the NUM_NEIGHBORS nearest earlier ones in bitstream order), for the
    integer positions q_xy [P, 2] (numpy int64): indices [P, NUM_NEIGHBORS] (int64) into the Gaussians in their given order, -1 for none.
    """
    _, order, _ = morton_order(q_xy)
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
    For the gradients of a rate term: the same slope as laplace_bits away from zero, in a few kernels.
    """
    return (torch.sqrt(x.square() + 0.25) - 0.5) * inv_scale_ln2


def rate_model(q_xy, feat, log2_scale, q_rot, color_refs, pred_refs, steps, dtype, device):
    """
    The fixed inputs of rate_bits from the current quantized Gaussians: integer positions q_xy [P, 2] (numpy int64), dequantized features
    [P, C] (the uniform steps [C]), canonical log2 scales [P, 2], and rotation indices [P]. Returns (neighbors [P, K] clamped, their weights
    [P, K], the CCLM slopes [P, C], channel_operands, the inverse steps [C], 1 / (scale ln 2) of the feature channels [C], and of each
    Gaussian's rotation [P, 1] by its aspect class), each scale the maximum-likelihood one (mean absolute residual index, at least MIN_SCALE).
    """
    with torch.no_grad():
        neighbors = torch.from_numpy(bitstream_neighbors(q_xy)).to(device)
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
        feat_scales = residuals.abs().mean(dim=0).clamp(min=MIN_SCALE)
        boundaries = torch.tensor(ASPECT_BOUNDARIES, dtype=log2_scale.dtype, device=device)
        classes = torch.bucketize((log2_scale[:, 0] - log2_scale[:, 1]).abs(), boundaries)
        sums = torch.zeros(len(ASPECT_BOUNDARIES) + 1, dtype=dtype, device=device).index_add_(0, classes, q_rot.abs().to(dtype))
        counts = torch.bincount(classes, minlength=len(ASPECT_BOUNDARIES) + 1).to(dtype)
        class_scales = torch.where(counts > 0, sums / counts.clamp(min=1.0), q_rot.abs().to(dtype).mean()).clamp(min=MIN_SCALE)
        return (neighbors, weights, slope, operands, inv_steps, 1.0 / (feat_scales * math.log(2.0)),
                1.0 / (class_scales[classes][:, None] * math.log(2.0)))


def rate_bits(feat, rot, model, inv_rot_step, channel_weights):
    """
    Differentiable estimate of the coded size in bits (smooth_laplace_bits, without constants) of the features [P, C] (residuals of their
    prediction, each channel's bits times channel_weights [C]) and of the rotations [P, 1] (canonical, by aspect class), with the fixed inputs
    of rate_model. Returns both totals.
    """
    neighbors, weights, slope, operands, inv_steps, inv_feat, inv_rot = model
    residuals = feature_residuals(feat, neighbors, weights, slope, operands, inv_steps)
    feat_bits = (smooth_laplace_bits(residuals, inv_feat * channel_weights)).sum()
    return feat_bits, smooth_laplace_bits(canonical_rotation(rot, inv_rot_step), inv_rot).sum()


def estimated_bits(feat, rot, model, inv_rot_step):
    """
    Code lengths in bits (laplace_bits) of the features and rotations under the distributions of rate_model, for logging: both totals.
    """
    neighbors, weights, slope, operands, inv_steps, inv_feat, inv_rot = model
    with torch.no_grad():
        residuals = feature_residuals(feat, neighbors, weights, slope, operands, inv_steps)
        feat_bits = laplace_bits(residuals, 1.0 / (inv_feat * math.log(2.0))).sum()
        rot_bits = laplace_bits(canonical_rotation(rot, inv_rot_step), 1.0 / (inv_rot * math.log(2.0))).sum()
    return feat_bits, rot_bits
