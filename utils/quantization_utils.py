import torch


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
