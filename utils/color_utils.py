import math

import torch

# Full-range BT.601 (JPEG) RGB -> YUV (YCbCr) without the chroma offset, so the transform stays linear and commutes with
# Gaussian rasterization (a weighted sum of features). U and V are centered at 0 with range [-0.5, 0.5]. The coefficients are those of the
# limited-range BT.601 conversion that HEVC's input and the luma metrics use, which only scales and offsets Y, U, V.
RGB_TO_YUV = [[0.299, 0.587, 0.114],
              [-0.168736, -0.331264, 0.5],
              [0.5, -0.418688, -0.081312]]
# RMS error of Y, and of U and V (their root mean square), when R, G, B are quantized independently with step 1: YUV features quantized with
# these factors times a step have the error per channel of RGB features quantized with that step
Y_STEP_SCALE = math.sqrt(sum(c * c for c in RGB_TO_YUV[0]))
UV_STEP_SCALE = math.sqrt(sum(c * c for row in RGB_TO_YUV[1:] for c in row) / 2.0)


def get_color_transforms(input_channels, dtype, device):
    """
    Block-diagonal RGB -> YUV and YUV -> RGB matrices over all stacked channels.
    Every 3-channel image is converted; 1-channel images are left unchanged.
    """
    rgb_to_yuv = torch.tensor(RGB_TO_YUV, dtype=torch.float64)
    blocks = [rgb_to_yuv if num_channels == 3 else torch.eye(num_channels, dtype=torch.float64) for num_channels in input_channels]
    forward = torch.block_diag(*blocks)
    inverse = torch.linalg.inv(forward)
    return forward.to(dtype=dtype, device=device), inverse.to(dtype=dtype, device=device)


def get_loss_transform(input_channels, uv_weight, dtype, device):
    """
    Block-diagonal map of YUV features to the space of the losses: the RGB image of Y, uv_weight U, uv_weight V, i.e., uv_weight RGB +
    (1 - uv_weight) times the gray image of the luma. With uv_weight 1 it is the RGB image, and for any uv_weight a luma error counts as in
    RGB. 1-channel images are left unchanged.
    """
    forward, inverse = get_color_transforms(input_channels, dtype=torch.float64, device="cpu")
    weights = torch.ones(len(forward), dtype=torch.float64)
    _, uv_channels, _ = get_yuv_channel_indices(input_channels, device="cpu")
    weights[uv_channels] = uv_weight
    return (inverse * weights[None, :]).to(dtype=dtype, device=device)


def get_yuv_channel_indices(input_channels, device):
    """
    Indices of the Y channels, the UV channels, and the remaining (single-channel) channels in the stacked features.
    """
    y_channels, uv_channels, other_channels = [], [], []
    curr_channel = 0
    for num_channels in input_channels:
        if num_channels == 3:
            y_channels.append(curr_channel)
            uv_channels.extend([curr_channel+1, curr_channel+2])
        else:
            other_channels.extend(range(curr_channel, curr_channel+num_channels))
        curr_channel += num_channels
    return [torch.tensor(channels, dtype=torch.long, device=device) for channels in (y_channels, uv_channels, other_channels)]


def transform_images(images, transform):
    """
    Apply a channel transform to images of shape [C, H, W].
    """
    return torch.einsum("dc,chw->dhw", transform, images)

