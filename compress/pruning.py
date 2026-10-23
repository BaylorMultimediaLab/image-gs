"""
Pruning and adding Gaussians: leave-one-out and contribution scores, the Gaussians of a lambda (rate-distortion) pruning round, keeping a
subset of the Gaussians with their optimizer state, and where to add Gaussians.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur

from gsplat.cuda import topk_norm_loo_loss_increase


def loo_loss_increase(xy, conics, feat, pixel_topk, target, eps, metric="l1", channel_weights=None):
    """
    Increase of the loss when removing each Gaussian on its own (leave-one-out), from the projected positions [N, 2], conics [N, 3], features
    [N, C], and top-K lists [H, W, K] of the rasterizer, against the target [C, H, W], in a single kernel (topk_norm_loo_loss_increase). The
    loss is the L1 loss (metric "l1") or the squared error summed over pixels and averaged over channels in 8-bit units (metric "sse").
    Removing Gaussian k from the top-K list of a pixel changes its value from S_f / (S_w + eps) to (S_f - w_k f_k) / (S_w - w_k + eps), where
    S_w and S_f are the sums of the top-K weights and weighted features. Ignores the Gaussian that would move into the top-K list. With
    channel_weights [C], the channels are scaled by them, normalized to a mean square of 1, so that with equal weights the loss is that of
    the unscaled channels.
    """
    num_channels, img_h, img_w = target.shape
    scaled_target = target
    if channel_weights is not None:
        weights = channel_weights * torch.sqrt(num_channels / channel_weights.square().sum())
        feat, scaled_target = feat * weights, target * weights[:, None, None]
    increase = topk_norm_loo_loss_increase(img_h, img_w, xy, conics, feat.contiguous(), pixel_topk, scaled_target.contiguous(), metric == "sse",
                                           eps)
    return increase * (255.0**2 / num_channels if metric == "sse" else 1.0 / target.numel())


def contribution(render_features, num_gaussians, dtype, device):
    """
    Contribution of each Gaussian to the rendered image, measured as its total blending weight summed over all pixels. Rendered images are
    linear in the features, so these weights are the gradients of the image sum w.r.t. all-one features (render_features(feat [N, 1])).
    """
    ones = torch.ones(num_gaussians, 1, dtype=dtype, device=device, requires_grad=True)
    with torch.enable_grad():
        images = render_features(ones)
        weights, = torch.autograd.grad(images.sum(), ones)
    return weights.squeeze(dim=-1)


def select_lambda_pruning(sse_increase, bits_per_gaussian, prune_lambda, max_ratio):
    """
    The Gaussians of a lambda pruning round: those whose removal increases the squared error (sse_increase [N]) by less than prune_lambda
    times the bits per Gaussian, at most max_ratio of them (the lowest increases). Returns their indices, the number of Gaussians below the
    threshold, and the threshold.
    """
    threshold = prune_lambda * bits_per_gaussian
    num_below = int((sse_increase < threshold).sum().item())
    num = min(num_below, round(max_ratio * len(sse_increase)))
    pruned = torch.topk(sse_increase, num, largest=False).indices if num > 0 else torch.zeros(0, dtype=torch.long, device=sse_increase.device)
    return pruned, num_below, threshold


def keep_gaussians(optimizer, keep):
    """
    The Gaussians where keep [N] is true, as new parameters, one per parameter group of the Adam optimizer (one parameter of N Gaussians
    each): the optimizer takes them in place of the old ones, with the state (exp_avg, exp_avg_sq) of the kept Gaussians. Returns the new
    parameters in the order of the groups.
    """
    params = []
    for param_group in optimizer.param_groups:
        old_param = param_group["params"][0]
        new_param = nn.Parameter(old_param.detach()[keep], requires_grad=True)
        state = optimizer.state.pop(old_param, None)
        if state is not None:
            state["exp_avg"], state["exp_avg_sq"] = state["exp_avg"][keep], state["exp_avg_sq"][keep]
            optimizer.state[new_param] = state
        param_group["params"] = [new_param]
        params.append(new_param)
    return params


def add_probabilities(images, gt_images, channel_weights=None, from_rgb=None):
    """
    Sampling probabilities [H*W] of the pixels where Gaussians are added, and the error [C, H, W] of the rendered images [C, H, W] against
    the target gt_images [C, H, W] (both RGB, gamma-corrected), the target smoothed with a Gaussian whose kernel grows with the resolution:
    the squared mean absolute error over the channels, in the color space of the features (from_rgb) with each channel scaled by
    channel_weights if given.
    """
    img_h, img_w = gt_images.shape[1:]
    kernel_size = round(np.sqrt(img_h * img_w) // 400)
    if kernel_size >= 1:
        kernel_size = max(3, kernel_size)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        gt_images = gaussian_blur(img=gt_images, kernel_size=kernel_size)
    diff_map = (gt_images - images).detach().clone()
    if channel_weights is None:
        error_map = torch.pow(torch.abs(diff_map).mean(dim=0).reshape(-1), 2.0)
    else:
        error_map = torch.pow((torch.abs(from_rgb(diff_map)) * channel_weights[:, None, None]).mean(dim=0).reshape(-1), 2.0)
    return error_map / error_map.sum(), diff_map
