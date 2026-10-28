"""
Initialization of the Gaussians: shapes from their spacing and the edges of the target, Gaussians on fine blobs, and a density of positions
from the local contrast of the target.
Positions are in [0, 1] (x, y) unless stated otherwise, scales in pixels. The first scale axis points along (cos rot, -sin rot) in pixels (x, y).
"""
import math

import torch
import torch.nn.functional as F

from utils.color_utils import RGB_TO_YUV
from utils.image_utils import find_blobs

STRUCTURE_TENSOR_SIGMA = 2.0  # Smoothing of the structure tensor that initial_shapes takes, in pixels
FINE_BLOB_SCALES = (0.7, 1.0, 1.4, 2.0, 2.8, 4.0)  # Scales (pixels) of the blob detection of find_fine_blobs
FINE_BLOB_MAX_SCALE = 1.4  # Largest scale (pixels) of the blobs that find_fine_blobs returns
LOCAL_STD_WINDOW = 9  # Window (pixels) of the local standard deviation of local_std_density
LOCAL_STD_FLOOR = 1.0 / 255.0  # Smallest local standard deviation that local_std_density uses (flat areas, luma in [0, 1])


def estimate_knn_distance(points, queries, img_h, img_w, k=3):
    """
    Estimated RMS distance [M] of each query [M, 2] to its k nearest points [N, 2] other than itself (in pixels; the queries are among the
    points), in O(pixels) instead of O(M N): from the number n of points in the 3x3 cells around the query at the finest level of a
    pyramid of square cells (1, 2, 4, ... pixels wide) where n > k, as in a Poisson process of density rho = n / area, where the squared
    distance to the i-th nearest point is i / (pi rho) on average. Within about 25% of the exact distance (sampled positions, 480p).
    """
    levels = math.ceil(math.log2(max(img_h, img_w)))
    size = 1 << levels
    column = points[:, 0].long().clamp(0, img_w-1)
    row = points[:, 1].long().clamp(0, img_h-1)
    counts = torch.zeros(size * size, dtype=points.dtype, device=points.device)
    counts = counts.index_add_(0, row * size + column, torch.ones_like(points[:, 0])).view(1, 1, size, size)
    column = queries[:, 0].long().clamp(0, img_w-1)
    row = queries[:, 1].long().clamp(0, img_h-1)
    window = torch.ones(1, 1, 3, 3, dtype=points.dtype, device=points.device)
    distance = torch.full_like(queries[:, 0], float("nan"))
    for level in range(levels + 1):
        if level > 0:
            counts = 4.0 * F.avg_pool2d(counts, 2)
        n = F.conv2d(counts, window, padding=1)[0, 0, row >> level, column >> level]
        estimate = torch.sqrt(0.5 * (k + 1) * 9.0 * 4.0**level / (math.pi * n))
        distance = torch.where(torch.isnan(distance) & (n > k), estimate, distance)
    return distance


def initial_shapes(all_xy, new_xy, img_h, img_w, init_scale, init_scale_knn, init_aspect, structure_tensor=None):
    """
    Scales [M, 2] (pixels) and rotations [M, 1] of the Gaussians at new_xy [M, 2], which are among all Gaussians all_xy [N, 2]: init_scale
    pixels, or init_scale_knn times the RMS distance to the 3 nearest other Gaussians, so that the Gaussians cover the image wherever they are
    sparse. With init_aspect > 1, they are elongated along the edges of the target (perpendicular to the dominant gradient direction of its
    structure tensor (Jxx, Jxy, Jyy) [3, H*W]) by an aspect ratio of up to init_aspect, by the coherence of the structure tensor, at the same
    area.
    """
    num = len(new_xy)
    if init_scale_knn > 0.0 and len(all_xy) > 3:
        size = torch.tensor([img_w, img_h], dtype=all_xy.dtype, device=all_xy.device)
        distance = estimate_knn_distance(all_xy * size, new_xy * size, img_h, img_w)
        scale = (init_scale_knn * distance).clamp(0.5, 0.25 * min(img_h, img_w))
    else:
        scale = torch.full((num,), init_scale, dtype=all_xy.dtype, device=all_xy.device)
    scale, rot = torch.stack([scale, scale], dim=1), torch.zeros(num, 1, dtype=all_xy.dtype, device=all_xy.device)
    if init_aspect > 1.0:
        column = (new_xy[:, 0] * img_w).long().clamp(0, img_w-1)
        row = (new_xy[:, 1] * img_h).long().clamp(0, img_h-1)
        jxx, jxy, jyy = structure_tensor[:, row * img_w + column]
        # Dominant gradient direction, and coherence ((l1 - l2) / (l1 + l2))^2 of the eigenvalues l1 >= l2
        phi = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy)
        coherence = (torch.hypot(jxx - jyy, 2.0 * jxy) / (jxx + jyy).clamp(min=1e-12)).square()
        aspect = torch.sqrt(1.0 + (init_aspect - 1.0) * coherence)
        scale = scale * torch.stack([aspect, 1.0 / aspect], dim=1)
        # The first scale axis is to point along the edge direction phi + pi/2
        rot = -(phi + 0.5 * math.pi).unsqueeze(dim=1)
    return scale, rot


def get_luma(images):
    """
    Luma [H, W] of images [C, H, W]: BT.601 weights for RGB, the mean of the channels otherwise.
    """
    if images.shape[0] == 3:
        return (torch.tensor(RGB_TO_YUV[0], dtype=images.dtype, device=images.device)[:, None, None] * images).sum(dim=0)
    return images.mean(dim=0)


def find_fine_blobs(images, num, scales=FINE_BLOB_SCALES, max_scale=FINE_BLOB_MAX_SCALE):
    """
    Pixel indices [M] and scales [M] (pixels) of the num strongest blobs of the luma of images [C, H, W] with a scale of at most max_scale
    (M <= num; utils.image_utils.find_blobs).
    """
    if num == 0:
        return torch.zeros(0, dtype=torch.long, device=images.device), torch.zeros(0, dtype=images.dtype, device=images.device)
    return find_blobs(get_luma(images), num, scales, max_scale)


def local_std_density(images, gamma, window=LOCAL_STD_WINDOW, floor=LOCAL_STD_FLOOR):
    """
    Sampling probabilities [H*W] of positions proportional to the local standard deviation of the luma of images [C, H, W] (over window x
    window pixels, repeating the edge pixels; at least floor) to the power gamma. Optimized models end with about this density of Gaussians
    at every rate (gamma ~0.8), from flat areas through smooth areas and texture to edges, so that one population sampled from it, sized by
    its spacing, covers every scale.
    """
    luma = F.pad(get_luma(images)[None, None], (window // 2,) * 4, mode="replicate")
    mean = F.avg_pool2d(luma, window, stride=1)
    variance = (F.avg_pool2d(luma.square(), window, stride=1) - mean.square()).clamp(min=0.0)
    density = variance.sqrt().clamp(min=floor).pow(gamma).reshape(-1)
    return density / density.sum()
