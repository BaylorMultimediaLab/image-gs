"""
Quantization of the Gaussian parameters: uniform quantizers, canonical parameters, quantization indices ("symbols") of the position, scale,
rotation, and features with their dequantization, per-Gaussian feature, rotation, and position steps, and the inputs of the bitstream and its
size estimate. Each attribute's symbols are (indices [P, C], steps [C], origins [C]). Scales are in pixels; "step" quantization (and its
rate estimate without quantization) codes them in log2 pixels. Per-Gaussian rotation and position steps are powers of 2 times the base
steps, so their grids are nested in the base grid: their indices stay in base steps (multiples of 2^e), and only the bitstream codes them
divided by 2^e.
"""
import math
import struct

import numpy as np
import torch

from compress import estimate_bits

FEAT_STEP_ADAPT_DIVISIONS = 4  # Per-Gaussian feature step multipliers are powers of 2^(1/4) ...
FEAT_STEP_ADAPT_OCTAVES = 2  # ... between 2^-2 and 2^2
ROT_STEP_ADAPT_OCTAVES = 2  # Per-Gaussian rotation steps are rot_step times 2^e, 0 <= e <= 2
POS_REFINE_BITS = 3  # Two-stage positions: coarse steps of 2^3 position steps, and per-axis position steps of pos_step times 2^e, 0 <= e <= 3
EXPONENT_TABLE_OCTAVES = 16  # Rows of the step exponent tables: differences of the log2 scales up to 16 octaves (larger ones take the last)


def quantize_minmax(x: torch.Tensor, num_bits: int = 16):
    """
    Uniform quantization with 2^num_bits levels over the range of x.
    Returns the quantization indices, and the per-channel steps and origins that dequantize them (shared by all channels).
    """
    qmin, qmax = 0, 2**num_bits - 1
    min_val, max_val = x.min().item(), x.max().item()
    step = max((max_val - min_val) / (qmax - qmin), 1e-8)
    q_x = torch.round((x - min_val) / step).clamp(qmin, qmax)
    return q_x, x.new_full((x.shape[-1],), step), x.new_full((x.shape[-1],), min_val)


def quantize_step(x: torch.Tensor, step):
    """
    Uniform quantization on a fixed grid anchored at zero, with a scalar or per-channel step.
    Returns the quantization indices, and the per-channel steps and origins that dequantize them.
    """
    steps = torch.as_tensor(step, dtype=x.dtype, device=x.device).expand(x.shape[-1])
    return torch.round(x / steps), steps, torch.zeros_like(steps)


def dequantize(q_x: torch.Tensor, steps: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
    return q_x * steps + origins


def ste_quantize(x: torch.Tensor, num_bits: int = 16) -> torch.Tensor:
    """
    Bit precision control of Gaussian parameters using a straight-through estimator.
    Reference: https://arxiv.org/abs/1308.3432
    """
    # Quantize in forward pass (non-differentiable)
    dq_x = dequantize(*quantize_minmax(x, num_bits))
    # Restore gradients in backward pass
    dq_x = x + (dq_x - x).detach()
    return dq_x


def canonicalize(scale, rot, quarter_turns, wrap_rot, log_scale):
    """
    Equivalent Gaussian parameters to quantize: ellipses are symmetric under rotation by pi (wrap_rot maps the rotation into [0, pi)), and
    under rotation by pi/2 with their axes swapped (quarter_turns keeps the rotation within [-pi/4, pi/4]), and rendering only depends on
    squared scales (absolute values for quantization in the log domain, log_scale). Differentiable, so that straight-through gradients reach
    the swapped axes.
    """
    if wrap_rot:
        rot = torch.remainder(rot, math.pi)
    if quarter_turns:
        turns = torch.round(rot.detach() / (0.5 * math.pi))
        rot = rot - turns * (0.5 * math.pi)
        scale = torch.where(torch.remainder(turns, 2.0) == 1.0, scale.flip(-1), scale)
    if log_scale:
        scale = scale.abs()
    return scale, rot


def rot_step_exponent_table(scale_step, aspect, gamma):
    """
    Exponents e [D] (int64) of the per-Gaussian rotation steps rot_step * 2^e by the absolute difference d of a Gaussian's two log2-scale
    indices (scale_step in octaves): the rotation sensitivity of a Gaussian footprint is proportional to (r - 1/r)^2 for its axis ratio
    r = 2^(d scale_step), so a Gaussian rounder than the reference axis ratio 2^aspect takes a step coarser by
    floor(gamma * log2 of the ratio of the sensitivities), up to 2^ROT_STEP_ADAPT_OCTAVES. Computed in double precision with Python's math
    module (rows up to EXPONENT_TABLE_OCTAVES).
    """
    reference = (2.0**aspect - 2.0**-aspect) ** 2
    exponents = []
    for d in range(int(EXPONENT_TABLE_OCTAVES / scale_step) + 1):
        ratio = 2.0 ** (d * scale_step)
        sensitivity = (ratio - 1.0 / ratio) ** 2
        exponent = ROT_STEP_ADAPT_OCTAVES if sensitivity == 0.0 else math.floor(gamma * math.log2(reference / sensitivity))
        exponents.append(min(max(exponent, 0), ROT_STEP_ADAPT_OCTAVES))
    return np.array(exponents, dtype=np.int64)


def pos_step_exponent_table(scale_step, rot_step, gamma):
    """
    Exponents e [D, Q + 1] (int64) of the per-Gaussian position step pos_step * 2^e along the pixel axis nearer a Gaussian's major axis, by
    the absolute difference d of its two log2-scale indices (scale_step in octaves) and its absolute canonical rotation index a <= Q (rot_step
    in radians, Q = half the rotation indices per quarter turn): the position sensitivity of a Gaussian footprint along a direction at angle
    theta to its major axis, relative to the one along its minor axis, is cos^2(theta) / r^2 + sin^2(theta) for its axis ratio
    r = 2^(d scale_step), so the step grows by floor(gamma / 2 * -log2 of it) at theta = a rot_step, up to 2^POS_REFINE_BITS. Computed in
    double precision with Python's math module (rows up to EXPONENT_TABLE_OCTAVES).
    """
    max_index = round(0.5 * math.pi / rot_step) // 2
    exponents = []
    for d in range(int(EXPONENT_TABLE_OCTAVES / scale_step) + 1):
        inv_ratio2 = 2.0 ** (-2.0 * d * scale_step)
        row = []
        for a in range(max_index + 1):
            relative = math.cos(a * rot_step) ** 2 * inv_ratio2 + math.sin(a * rot_step) ** 2
            row.append(min(max(math.floor(-0.5 * gamma * math.log2(relative)), 0), POS_REFINE_BITS))
        exponents.append(row)
    return np.array(exponents, dtype=np.int64)


def rot_step_exponents(q_scale, table):
    """
    Exponents [P] (int64) of the per-Gaussian rotation steps from the log2-scale indices [P, 2] and rot_step_exponent_table (a tensor).
    """
    d = (q_scale[:, 0] - q_scale[:, 1]).abs().long().clamp(max=len(table) - 1)
    return table[d]


def pos_step_exponents(q_scale, q_rot, table, quarter):
    """
    Exponents [P, 2] (int64) of the per-Gaussian position steps of the x and y axes from the log2-scale indices [P, 2], the rotation indices
    [P] in base steps (quarter of them per quarter turn), and pos_step_exponent_table (a tensor): the table's exponent on the pixel axis
    nearer the major axis, 0 on the other. Rotations outside [-pi/4, pi/4] are canonicalized first (a quarter turn swaps the axes), so that a
    Gaussian gets the same exponents in either form; at exactly pi/4, where the two forms disagree on the nearer axis, the exponent is 0
    (for gamma <= 2).
    """
    a = q_rot.long()
    turns = torch.round(a.double() / quarter).long()
    canonical = (a - turns * quarter).abs().clamp(max=table.shape[1] - 1)
    swapped = torch.remainder(turns, 2) == 1
    s0, s1 = torch.where(swapped, q_scale[:, 1], q_scale[:, 0]), torch.where(swapped, q_scale[:, 0], q_scale[:, 1])
    d = (s0 - s1).abs().long().clamp(max=table.shape[0] - 1)
    exponents = table[d, canonical]
    near_x = s0 >= s1  # Canonical axis 0 is within 45 degrees of the x axis
    return torch.stack([torch.where(near_x, exponents, 0), torch.where(near_x, 0, exponents)], dim=1)


def quantize_nested(x, step, exponents):
    """
    Indices in base steps of x on per-Gaussian grids of step * 2^exponents (exponents broadcast against x): multiples of 2^exponents.
    """
    multipliers = torch.exp2(exponents.to(x.dtype))
    return torch.round(x / (step * multipliers)) * multipliers


def get_step_symbols(xy, scale, rot, feat, steps, feat_step_multipliers=None, rot_exponents=None, pos_exponents=None):
    """
    Symbols of "step" quantization of the canonicalized Gaussian parameters (scale in pixels, quantized in log2 pixels), with the per-channel
    steps of position, log2 scale, rotation, and features. feat_step_multipliers(geometry_symbols) gives per-Gaussian multipliers [P, C] of
    the feature steps, which the returned feature steps (the base steps) do not include. rot_exponents(q_scale) and
    pos_exponents(q_scale, q_rot) give per-Gaussian exponents of the rotation [P] and position [P, 2] steps, whose indices stay in base steps.
    """
    with torch.no_grad():
        q_scale = quantize_step(torch.log2(scale), steps[1])
        if rot_exponents is None:
            q_rot = quantize_step(rot, steps[2])
        else:
            q_rot = (quantize_nested(rot, steps[2], rot_exponents(q_scale[0])[:, None]), steps[2], torch.zeros_like(steps[2]))
        if pos_exponents is None:
            q_xy = quantize_step(xy, steps[0])
        else:
            q_xy = (quantize_nested(xy, steps[0], pos_exponents(q_scale[0], q_rot[0][:, 0])), steps[0], torch.zeros_like(steps[0]))
        symbols = [q_xy, q_scale, q_rot]
        if feat_step_multipliers is None:
            return symbols + [quantize_step(feat, steps[3])]
        q_feat = torch.round(feat / (steps[3] * feat_step_multipliers(symbols)))
        return symbols + [(q_feat, steps[3], torch.zeros_like(steps[3]))]


def get_minmax_symbols(xy, scale, rot, feat, geometry_bits, feat_groups):
    """
    Symbols of "minmax" quantization of the canonicalized Gaussian parameters (scale in pixels): 2^bits levels over the range of the position,
    scale, and rotation (geometry_bits), and of each group of feature channels [(channel indices, bits)].
    """
    with torch.no_grad():
        symbols = [quantize_minmax(x, bits) for x, bits in zip([xy, scale, rot], geometry_bits)]
        q_feat, feat_steps, feat_origins = torch.empty_like(feat), feat.new_empty(feat.shape[1]), feat.new_empty(feat.shape[1])
        for channels, bits in feat_groups:
            q_feat[:, channels], feat_steps[channels], feat_origins[channels] = quantize_minmax(feat[:, channels], bits)
    return symbols + [(q_feat, feat_steps, feat_origins)]


def dequantize_symbols(symbols, log_scale, feat_step_multipliers=None):
    """
    Gaussian parameters (scale in pixels) from their symbols: scales in log2 pixels with log_scale ("step" quantization), feature steps scaled
    per Gaussian by feat_step_multipliers(geometry_symbols) if given.
    """
    xy, scale, rot, feat = [dequantize(*symbol) for symbol in symbols]
    if log_scale:
        if feat_step_multipliers is not None:
            q_feat, steps, origins = symbols[3]
            feat = dequantize(q_feat, steps * feat_step_multipliers(symbols[:3]), origins)
        scale = torch.exp2(scale)
    return xy, scale, rot, feat


def feat_step_multipliers(geometry_symbols, mode, gammas, get_feature_weights=None):
    """
    Per-Gaussian multipliers [P, C] (float64) of the feature steps: (proxy / its geometric mean)^(-gamma) with the gamma of each channel
    (gammas [C], float64), snapped to powers of 2^(1/FEAT_STEP_ADAPT_DIVISIONS) within 2^(+-FEAT_STEP_ADAPT_OCTAVES). The proxy is the
    Gaussian's area (mode "area") or get_feature_weights(xy, scale, rot) of the dequantized geometry (mode "weight", see feature_weights),
    the sensitivity of the rendered image to its features. Both only depend on the symbols of the geometry [position, log2 scale,
    rotation], and neither depends on the order of the Gaussians, so a decoder computes the same multipliers.
    """
    if mode == "area":
        # log2 of the area is the scale step times the sum of the two log2-scale indices, whose mean is exact
        sums = geometry_symbols[1][0].sum(dim=1).to(torch.float64)
        log_proxy = (sums - sums.sum() / len(sums)) * geometry_symbols[1][1][0].to(torch.float64)
    else:
        xy, log_scale, rot = [dequantize(*symbol) for symbol in geometry_symbols]
        log_proxy = torch.log2(get_feature_weights(xy, torch.exp2(log_scale), rot).clamp(min=2.0**-40))
        log_proxy = log_proxy - log_proxy.mean()
    limit = FEAT_STEP_ADAPT_DIVISIONS * FEAT_STEP_ADAPT_OCTAVES
    divisions = torch.round((-FEAT_STEP_ADAPT_DIVISIONS * gammas)[None, :] * log_proxy[:, None]).clamp(-limit, limit)
    return torch.exp2(divisions / FEAT_STEP_ADAPT_DIVISIONS)


def feature_weights(topk_weights, num_gaussians, eps, device):
    """
    Sum over the pixels of each Gaussian's squared normalized top-K weight [num_gaussians] (float64), from the top-K lists of chunks of pixels
    (topk_weights: (first pixel, ids [P, K], valid [P, K], weights before normalization [P, K]) per chunk; eps of the normalization). The
    weights are normalized in float64 and summed in fixed point with integer atomics, so the sums do not depend on the order of the Gaussians
    or of the additions.
    """
    fixed_point = torch.zeros(num_gaussians, dtype=torch.int64, device=device)
    for _, ids, valid, weights in topk_weights:
        weights = weights.to(torch.float64)
        normalized = weights / (weights.sum(dim=1, keepdim=True) + eps)
        fixed_point.index_add_(0, ids[valid], torch.round(normalized[valid].square() * 2.0**40).to(torch.int64))
    return fixed_point.to(torch.float64) / 2.0**40


def fixed_length_bits(symbols):
    """
    Total size in bits of fixed-length codes of the symbols, each channel with as many bits as its range of indices requires.
    """
    q = torch.cat([q for q, _, _ in symbols], dim=1)
    return torch.ceil(torch.log2(q.amax(dim=0) - q.amin(dim=0) + 1.0)).sum().item() * len(q)


def get_channel_refs(input_channels, color_transform):
    """
    Reference channels of each attribute for the lossless color transform before entropy coding: subtract-green codes each RGB image
    as R-G, G, B-G, and leaves the other channels unchanged.
    """
    feat_refs = []
    for num_channels in input_channels:
        base = len(feat_refs)
        if color_transform == "subtract-green" and num_channels == 3:
            feat_refs += [base + 1, -1, base + 1]
        else:
            feat_refs += [-1] * num_channels
    return [None, None, None, feat_refs]


def estimate_symbol_bits(symbols, channel_refs):
    """
    Fast estimate of the coded size in bits of the symbols, without the bitstream header (compress.estimate_bits).
    """
    return estimate_bits([q.cpu().numpy().astype(np.int64) for q, _, _ in symbols], channel_refs=channel_refs)


def symbols_to_numpy(symbols):
    """
    The symbols as numpy arrays (indices int64, steps and origins float64), as compress.encode_bitstream takes them.
    """
    return [(q.cpu().numpy().astype(np.int64), steps.cpu().numpy().astype(np.float64), origins.cpu().numpy().astype(np.float64))
            for q, steps, origins in symbols]


def bitstream_meta(img_h, img_w, quant_mode, color_space, feat_prediction, feat_step_adapt, feat_step_gamma, uv_step_gamma, geometry=None):
    """
    Model metadata of the bitstream: image size, quantization mode, color space, whether the features are coded as residuals of their
    prediction (compress.prediction with get_prediction_refs), and the per-Gaussian feature steps (feat_step_adapt "none", "area", or
    "weight" with feat_step_gamma, and in YUV mode uv_step_gamma for U and V), which a decoder computes from the decoded geometry. With
    geometry options (scale_prediction, rot_step_adapt "aspect" with rot_step_aspect and rot_step_gamma, pos_step_adapt "major" with
    pos_step_gamma; all float32 values), a byte of flags and their parameters follow.
    """
    meta = struct.pack("<IIBBB", img_h, img_w, ["minmax", "step"].index(quant_mode), ["rgb", "yuv"].index(color_space), int(feat_prediction))
    if feat_step_adapt != "none":
        meta += struct.pack("<Bf", ["area", "weight"].index(feat_step_adapt) + 1, feat_step_gamma)
        if color_space == "yuv":
            meta += struct.pack("<f", uv_step_gamma)
    if geometry is not None:
        scale_prediction, rot_step, pos_step = geometry
        meta += struct.pack("<B", int(scale_prediction) | int(rot_step is not None) << 1 | int(pos_step is not None) << 2)
        if rot_step is not None:
            meta += struct.pack("<ff", *rot_step)
        if pos_step is not None:
            meta += struct.pack("<f", pos_step)
    return meta
