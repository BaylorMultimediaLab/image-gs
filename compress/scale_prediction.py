"""
Lossless prediction of the log2-scale indices from the positions before entropy coding: Gaussians in dense regions are small, those in
sparse regions large. The sum of a Gaussian's two log2-scale indices (log2 of its area over the scale step) is predicted by a fixed linear
function of log2(4 d^2 + 1) for the distances d to its 1st, 2nd, 4th, and 8th nearest Gaussian (compress/csrc/prediction.cpp) and coded as
the residual of the rounded prediction; the difference of the two indices, which the positions do not predict, is coded halved (it has the
parity of the sum). The positions are those a decoder has before the scales (the coarse ones of two-stage positions, as integers), and only
the slopes matter: a constant offset moves every residual alike, which the entropy coder's median shift absorbs, so the slopes need no
normalization by the density or the units. Fixed-point logarithms, integer sums, and one correctly rounded division make the prediction the
same in every encoder and decoder.
"""
import functools

import numpy as np

from compress.prediction import _prediction

LOG_FRAC_BITS = 16  # Fractional bits of the fixed-point logarithms
LOG_TABLE_SIZE = 1 << 16  # Smaller arguments of the logarithm come from a table
NEIGHBOR_RANKS = (1, 2, 4, 8)
# Slopes in octaves of the area per unit of log2(4 d^2 + 1), with 16 fractional bits, fitted on 96 bitstreams of 3 UVG frames (480p and
# 720p, QP 22-37, 0.5-4 s budgets) with distances in 1-pixel units (leave-one-image-out fits save the same)
SLOPES = np.array([2623, 1494, 8609, 46086], dtype=np.int64)
SLOPE_FRAC_BITS = 16


def _squaring_log2(values):
    """
    fixed_log2 by repeated squaring of the mantissa in integer arithmetic.
    """
    values = np.asarray(values, dtype=np.uint64)
    exponents = np.frexp(values.astype(np.float64))[1].astype(np.int64) - 1
    mantissa = np.where(exponents <= 30, values << np.clip(30 - exponents, 0, None).astype(np.uint64),
                        values >> np.clip(exponents - 30, 0, None).astype(np.uint64))  # In [2^30, 2^31): 1 <= x < 2 with 30 fraction bits
    fraction = np.zeros(values.shape, dtype=np.int64)
    for _ in range(LOG_FRAC_BITS):
        mantissa = (mantissa * mantissa) >> np.uint64(30)
        carry = mantissa >= np.uint64(1 << 31)
        fraction = (fraction << 1) | carry
        mantissa = np.where(carry, mantissa >> np.uint64(1), mantissa)
    return (exponents << LOG_FRAC_BITS) | fraction


@functools.cache
def _log_table():
    return _squaring_log2(np.arange(1, LOG_TABLE_SIZE))


def fixed_log2(values):
    """
    log2 of positive integers [int64, below 2^53] with LOG_FRAC_BITS fractional bits (int64), by repeated squaring of the mantissa in
    integer arithmetic (truncated: below the exact value by less than one unit of the last bit for values below 2^31); values below
    LOG_TABLE_SIZE from a table of the same.
    """
    values = np.asarray(values, dtype=np.int64)
    table = _log_table()
    small = values < LOG_TABLE_SIZE
    out = table[np.where(small, values, 1) - 1]
    if not small.all():
        out[~small] = _squaring_log2(values[~small])
    return out


def predicted_sums(positions, scale_step):
    """
    Predicted sums [P] (int64) of the two log2-scale indices (scale_step in octaves) of Gaussians at the integer positions [P, 2], in any
    order. With fewer than 8 other Gaussians, a missing distance is the largest one found (0 without any).
    """
    dist2 = _prediction.nearest_distances(np.ascontiguousarray(positions, dtype=np.int64))
    dist2 = np.maximum.accumulate(np.maximum(dist2, 0), axis=1)
    logs = fixed_log2(4 * dist2[:, [rank - 1 for rank in NEIGHBOR_RANKS]] + 1)
    octaves = logs @ SLOPES  # LOG_FRAC_BITS + SLOPE_FRAC_BITS fractional bits, below 2^53
    return np.rint(octaves.astype(np.float64) / (2.0 ** (LOG_FRAC_BITS + SLOPE_FRAC_BITS) * scale_step)).astype(np.int64)


def scale_residuals(q_scale, predicted):
    """
    Values [P, 2] to code instead of the log2-scale indices [P, 2]: the residual of their sum, and their halved difference.
    """
    sums = q_scale[:, 0] + q_scale[:, 1]
    return np.stack([sums - predicted, (q_scale[:, 0] - q_scale[:, 1] - (sums & 1)) // 2], axis=1)


def reconstruct_scales(coded, predicted):
    """
    Log2-scale indices [P, 2] from the values of scale_residuals with the same predictions.
    """
    sums = coded[:, 0] + predicted
    diffs = 2 * coded[:, 1] + (sums & 1)
    return np.stack([(sums + diffs) // 2, (sums - diffs) // 2], axis=1)
