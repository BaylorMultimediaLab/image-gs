"""
Lossless prediction of quantized feature indices before entropy coding, in bitstream (Morton) order. Each channel is predicted from the
NUM_NEIGHBORS nearest earlier Gaussians (by squared distance, then by index): a channel without a reference from the mean of their values,
a channel with a reference channel CCLM-style (as VVC's cross-component linear model, with the parameters from decoded neighbors instead of
signaled): the least-squares line between the channel and its reference over the neighbors, evaluated at the Gaussian's own reference
value. Optional reference subtraction (e.g., subtract-green, compress.common.subtract_channels) comes first, so that R-G and B-G are
predicted from G.

Predictions use the dequantized values, as features of Gaussians with different per-Gaussian steps (feat_step_adapt multipliers 2^(d/4)) are
comparable only there: fixed-point values with FRAC_BITS fractional bits of the smallest channel step. The neighbor search and the
prediction are implemented in C++ (compress/csrc/prediction.cpp, module igs_prediction) with integer sums and a fixed sequence of IEEE double
operations, so that encoders and decoders compute the same predicted indices; the residual is the index minus the predicted value's index
at the Gaussian's own step (rounded half to even).
"""
import numpy as np

from compress.common import add_channels, load_extension, subtract_channels

_prediction = load_extension("igs_prediction", "prediction.cpp")

NUM_NEIGHBORS = _prediction.NUM_NEIGHBORS
FRAC_BITS = 10
QUARTER_POWERS = np.array([1.0, 1.189207115002721, 1.4142135623730951, 1.681792830507429])  # 2^(i/4), i = 0..3


def get_prediction_refs(input_channels, color_space):
    """
    Prediction references of the stacked feature channels (predict_residuals): in every 3-channel image, Y from the neighbors, U from Y, and
    V from U (YUV), or G from the neighbors, and R (or R-G) and B (or B-G) from G (RGB); other channels from the neighbors.
    """
    refs = []
    for num_channels in input_channels:
        base = len(refs)
        if num_channels == 3:
            refs += [-1, base, base + 1] if color_space == "yuv" else [base + 1, -1, base + 1]
        else:
            refs += [-1] * num_channels
    return refs


def fixed_point_steps(steps, exponents):
    """
    Fixed-point value [P, C] (int64) of one index step of each Gaussian and channel: 2^FRAC_BITS times the channel step (steps [C]) over the
    smallest one, times the Gaussian's multiplier 2^(d/4) (exponents [P, C] int64).
    """
    ratios = steps / steps.min()
    return np.rint(np.ldexp(ratios[None, :] * QUARTER_POWERS[exponents % 4], FRAC_BITS + exponents // 4)).astype(np.int64)


def causal_neighbors(positions):
    """
    The NUM_NEIGHBORS nearest earlier Gaussians of each Gaussian (integer positions [P, 2] in bitstream order) that predict_residuals predicts
    from: indices [P, NUM_NEIGHBORS] (int64), -1 for none.
    """
    return _prediction.causal_neighbors(np.ascontiguousarray(positions, dtype=np.int64))


def predict_residuals(indices, steps, exponents, positions, color_refs, pred_refs):
    """
    Residual indices [P, C] to code instead of the indices [P, C] of Gaussians in bitstream order (integer positions [P, 2]), with the
    channel steps [C], per-Gaussian step exponents [P, C], reference subtraction color_refs [C] (-1: none), and prediction references
    pred_refs [C] (-1: the mean of the neighbors, else the reference channel, which must not depend on the channel).
    """
    return _prediction.predict_residuals(subtract_channels(indices, color_refs), fixed_point_steps(steps, exponents), pred_refs, positions)


def reconstruct_indices(residuals, steps, exponents, positions, color_refs, pred_refs):
    """
    Indices [P, C] from the residuals [P, C] of predict_residuals with the same arguments.
    """
    return add_channels(_prediction.reconstruct_indices(residuals, fixed_point_steps(steps, exponents), pred_refs, positions), color_refs)
