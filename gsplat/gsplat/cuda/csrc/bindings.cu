#include "backward.cuh"
#include "bindings.h"
#include "forward.cuh"
#include "forward2d.cuh"
#include "backward2d.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>

namespace cg = cooperative_groups;

__global__ void compute_cov2d_bounds_kernel(
    const unsigned num_pts, const float* __restrict__ covs2d, float* __restrict__ conics, float* __restrict__ radii
) {
    unsigned row = cg::this_grid().thread_rank();
    if (row >= num_pts) {
        return;
    }
    int index = row * 3;
    float3 conic;
    float radius;
    float3 cov2d{
        (float)covs2d[index], (float)covs2d[index + 1], (float)covs2d[index + 2]
    };
    compute_cov2d_bounds(cov2d, conic, radius);
    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    radii[row] = radius;
}

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &covs2d) {
    CHECK_INPUT(covs2d);
    torch::Tensor conics = torch::zeros(
        {num_pts, covs2d.size(1)}, covs2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor radii =
        torch::zeros({num_pts, 1}, covs2d.options().dtype(torch::kFloat32));

    int blocks = (num_pts + N_THREADS - 1) / N_THREADS;

    compute_cov2d_bounds_kernel<<<blocks, N_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_pts,
        covs2d.contiguous().data_ptr<float>(),
        conics.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<float>()
    );
    return std::make_tuple(conics, radii);
}

std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);

    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    map_gaussian_to_intersects<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_points,
        (float2 *)xys.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        cum_tiles_hit.contiguous().data_ptr<int32_t>(),
        tile_bounds_dim3,
        // Outputs.
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}

// Gaussian IDs of the tile intersections sorted by tile (in increasing Gaussian ID within each tile, the order of bin_and_sort_gaussians), and
// each tile's range in them, without reading the number of intersections back to the CPU (CUDA graph capture): the buffers hold capacity
// entries, the unused ones sort last and belong to no tile, and intersections beyond the capacity are dropped, so the caller must check
// their number (cum_tiles_hit[-1]) against it. A stable radix sort of 32-bit tile IDs over only the bits they use, with the Gaussian IDs as
// values.
std::tuple<torch::Tensor, torch::Tensor> sort_intersects_by_tile_tensor(
    const int num_points,
    const int capacity,
    const torch::Tensor &xys,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    const int num_tiles = tile_bounds_dim3.x * tile_bounds_dim3.y;
    const auto int_options = xys.options().dtype(torch::kInt32);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (capacity <= 0) {
        return std::make_tuple(torch::zeros({0}, int_options), torch::zeros({num_tiles, 2}, int_options));
    }

    // padding entries get the key num_tiles, above every tile ID
    torch::Tensor isect_tiles = torch::full({capacity}, num_tiles, int_options);
    torch::Tensor gaussian_ids = torch::zeros({capacity}, int_options);
    if (num_points > 0) {
        map_intersects_to_tiles<<<(capacity + N_THREADS - 1) / N_THREADS, N_THREADS, 0, stream>>>(
            num_points,
            capacity,
            (float2 *)xys.data_ptr<float>(),
            radii.data_ptr<int32_t>(),
            cum_tiles_hit.data_ptr<int32_t>(),
            tile_bounds_dim3,
            (uint32_t *)isect_tiles.data_ptr<int32_t>(),
            gaussian_ids.data_ptr<int32_t>()
        );
        CUDA_CHECK_LAUNCH();
    }

    int end_bit = 1;
    while ((1u << end_bit) <= (unsigned)num_tiles)
        ++end_bit;
    torch::Tensor isect_tiles_sorted = torch::empty_like(isect_tiles);
    torch::Tensor gaussian_ids_sorted = torch::empty_like(gaussian_ids);
    const uint32_t *keys_in = (const uint32_t *)isect_tiles.data_ptr<int32_t>();
    uint32_t *keys_out = (uint32_t *)isect_tiles_sorted.data_ptr<int32_t>();
    size_t temp_bytes = 0;
    TORCH_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys_in, keys_out, gaussian_ids.data_ptr<int32_t>(),
                                                gaussian_ids_sorted.data_ptr<int32_t>(), capacity, 0, end_bit, stream) == cudaSuccess,
                "sort_intersects_by_tile: radix sort failed");
    torch::Tensor temp = torch::empty({(int64_t)temp_bytes}, xys.options().dtype(torch::kUInt8));
    TORCH_CHECK(cub::DeviceRadixSort::SortPairs(temp.data_ptr(), temp_bytes, keys_in, keys_out, gaussian_ids.data_ptr<int32_t>(),
                                                gaussian_ids_sorted.data_ptr<int32_t>(), capacity, 0, end_bit, stream) == cudaSuccess,
                "sort_intersects_by_tile: radix sort failed");

    torch::Tensor tile_bins = torch::zeros({num_tiles, 2}, int_options);
    get_tile_bin_edges_from_tiles<<<(capacity + N_THREADS - 1) / N_THREADS, N_THREADS, 0, stream>>>(
        capacity, num_tiles, keys_out, (int2 *)tile_bins.data_ptr<int32_t>()
    );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(gaussian_ids_sorted, tile_bins);
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects, int num_tiles, const torch::Tensor &isect_ids_sorted
) {
    CHECK_INPUT(isect_ids_sorted);
    torch::Tensor tile_bins = torch::zeros(
        {num_tiles, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_intersects,
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
    );
    CUDA_CHECK_LAUNCH();
    return tile_bins;
}

std::tuple<
    torch::Tensor
> rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );

    // tile:  tile_w * tile_h * 1
    // block: block_w * block_h * 1
    // each thread processes a pixel
    // each block processes a tile
    rasterize_forward<<<tile_bounds_dim3, block_dim3, 0, at::cuda::getCurrentCUDAStream()>>>(
        tile_bounds_dim3,
        img_size_dim3,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        (float3 *)out_img.contiguous().data_ptr<float>()
    );
    
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(out_img);
}

std::tuple<
    torch::Tensor
> nd_rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );

    nd_rasterize_forward<<<tile_bounds_dim3, block_dim3, 0, at::cuda::getCurrentCUDAStream()>>>(
        tile_bounds_dim3,
        img_size_dim3,
        channels,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        out_img.contiguous().data_ptr<float>()
    );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(out_img);
}


std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor  // dL_dcolors
        >
    nd_rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned BLOCK_H,
        const unsigned BLOCK_W,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &v_output  // dL_dout_color
    ) {

    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_W - 1) / BLOCK_W,
        (img_height + BLOCK_H - 1) / BLOCK_H,
        1
    };
    const dim3 block(BLOCK_W, BLOCK_H, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());

    // torch::Tensor workspace;
    // if (channels > 3) {
    //     workspace = torch::zeros(
    //         {img_height, img_width, channels},
    //         xys.options().dtype(torch::kFloat32)
    //     );
    // } else {
    //     workspace = torch::zeros({0}, xys.options().dtype(torch::kFloat32));
    // }

    nd_rasterize_backward_kernel<<<tile_bounds, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        tile_bounds,
        img_size,
        channels,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        v_output.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>()
    );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(v_xy, v_conic, v_colors);
}

std::tuple<
    torch::Tensor, // out_img 
    torch::Tensor  // pixel_topk 
> nd_rasterize_forward_topk_norm_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    // The kernel writes every pixel, channel-planar
    torch::Tensor out_img = torch::empty(
        {channels, img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor pixel_topk = torch::empty(
        {img_height, img_width, TOP_K}, xys.options().dtype(torch::kInt32)
    );
    
    nd_rasterize_forward_topk_norm<<<tile_bounds_dim3, block_dim3, 0, at::cuda::getCurrentCUDAStream()>>>(
        tile_bounds_dim3,
        img_size_dim3,
        channels,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        out_img.contiguous().data_ptr<float>(),
        pixel_topk.contiguous().data_ptr<int32_t>()
    );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(out_img, pixel_topk);
}

std::tuple<
    torch::Tensor, // dL_dxy
    torch::Tensor, // dL_dconic
    torch::Tensor  // dL_dcolors
    >
nd_rasterize_backward_topk_norm_tensor(
    const unsigned img_height,
    const unsigned img_width,
    const unsigned BLOCK_H,
    const unsigned BLOCK_W,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &v_output,  // dL_dout_color, channel-planar [C, H, W]
    const torch::Tensor &pixel_topk
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(v_output);
    CHECK_INPUT(pixel_topk);
    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }
    if (colors.ndimension() != 2) {
        AT_ERROR("colors must have 2 dimensions");
    }
    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_W - 1) / BLOCK_W,
        (img_height + BLOCK_H - 1) / BLOCK_H,
        1
    };
    const dim3 block(BLOCK_W, BLOCK_H, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);
    if (BLOCK_W % 8 != 0 || BLOCK_H % 4 != 0) {
        AT_ERROR("the block width must be a multiple of 8 and its height a multiple of 4 (warps cover 8x4 pixels)");
    }
    if (channels > MAX_CHANNELS) {
        AT_ERROR("at most MAX_CHANNELS channels");
    }
    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    nd_rasterize_backward_topk_norm(
        tile_bounds,
        block,
        img_size,
        channels,
        (float2 *)xys.data_ptr<float>(),
        (float3 *)conics.data_ptr<float>(),
        colors.data_ptr<float>(),
        v_output.data_ptr<float>(),
        (float2 *)v_xy.data_ptr<float>(),
        (float3 *)v_conic.data_ptr<float>(),
        v_colors.data_ptr<float>(),
        pixel_topk.data_ptr<int32_t>(),
        at::cuda::getCurrentCUDAStream()
    );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(v_xy, v_conic, v_colors);
}

std::tuple<
    torch::Tensor, // out_img 
    torch::Tensor  // pixel_topk 
> nd_rasterize_forward_no_tiles_tensor(
    const std::tuple<int, int, int> img_size,
    const unsigned num_points,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);

    dim3 block_dim3;
    block_dim3.x = 32;
    block_dim3.y = 32;
    block_dim3.z = 1;

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    dim3 grid_size_dim3;
    grid_size_dim3.x = (img_size_dim3.x + block_dim3.x - 1) / block_dim3.x;
    grid_size_dim3.y = (img_size_dim3.y + block_dim3.y - 1) / block_dim3.y;
    grid_size_dim3.z = 1;

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor pixel_topk = torch::zeros(
        {img_height, img_width, TOP_K}, xys.options().dtype(torch::kInt32)
    );
    
    nd_rasterize_forward_no_tiles<<<grid_size_dim3, block_dim3, 0, at::cuda::getCurrentCUDAStream()>>>(
        img_size_dim3,
        channels,
        num_points,
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        out_img.contiguous().data_ptr<float>(),
        pixel_topk.contiguous().data_ptr<int32_t>()
    );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(out_img, pixel_topk);
}

std::tuple<
    torch::Tensor, // dL_dxy
    torch::Tensor, // dL_dconic
    torch::Tensor  // dL_dcolors
    >
nd_rasterize_backward_no_tiles_tensor(
    const unsigned img_height,
    const unsigned img_width,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &v_output,  // dL_dout_color
    const torch::Tensor &pixel_topk
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);
    CHECK_INPUT(pixel_topk);
    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }
    if (colors.ndimension() != 2) {
        AT_ERROR("colors must have 2 dimensions");
    }
    const int num_points = xys.size(0);
    const dim3 block_size = {32, 32, 1};
    const dim3 img_size = {img_width, img_height, 1};
    const dim3 grid_size = {(img_width + block_size.x - 1)/block_size.x,
                            (img_height + block_size.y - 1)/block_size.y, 1};
    
    const int channels = colors.size(1);
    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    nd_rasterize_backward_no_tiles_kernel<<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        img_size,
        channels,
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        v_output.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        pixel_topk.contiguous().data_ptr<int32_t>()
    );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(v_xy, v_conic, v_colors);
}


std::tuple<
    torch::Tensor, // out_img 
    torch::Tensor  // pixel_topk 
> nd_rasterize_forward_simple_tensor(
    const std::tuple<int, int, int> img_size,
    const unsigned num_points,
    const torch::Tensor &xys,
    const torch::Tensor &scale,
    const torch::Tensor &rot,
    const torch::Tensor &feat
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(scale);
    CHECK_INPUT(rot);
    CHECK_INPUT(feat);

    dim3 block_dim3;
    block_dim3.x = 32;
    block_dim3.y = 32;
    block_dim3.z = 1;

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    dim3 grid_size_dim3;
    grid_size_dim3.x = (img_size_dim3.x + block_dim3.x - 1) / block_dim3.x;
    grid_size_dim3.y = (img_size_dim3.y + block_dim3.y - 1) / block_dim3.y;
    grid_size_dim3.z = 1;

    const int channels = feat.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor pixel_topk = torch::zeros(
        {img_height, img_width, TOP_K}, xys.options().dtype(torch::kInt32)
    );
    
    // nd_rasterize_forward_no_tiles<<<grid_size_dim3, block_dim3>>>(
    //     img_size_dim3,
    //     channels,
    //     num_points,
    //     (float2 *)xys.contiguous().data_ptr<float>(),
    //     (float3 *)conics.contiguous().data_ptr<float>(),
    //     colors.contiguous().data_ptr<float>(),
    //     out_img.contiguous().data_ptr<float>(),
    //     pixel_topk.contiguous().data_ptr<int32_t>()
    // );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(out_img, pixel_topk);
}


std::tuple<
    torch::Tensor, // dL_dxy
    torch::Tensor, // dL_dscale
    torch::Tensor, // dL_drot
    torch::Tensor  // dL_dfeat
    >
nd_rasterize_backward_simple_tensor(
    const unsigned img_height,
    const unsigned img_width,
    const torch::Tensor &xys,
    const torch::Tensor &scale,
    const torch::Tensor &rot,
    const torch::Tensor &feat,
    const torch::Tensor &v_output,  // dL_dout_color
    const torch::Tensor &pixel_topk
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(scale);
    CHECK_INPUT(rot);
    CHECK_INPUT(feat);
    CHECK_INPUT(pixel_topk);
    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }
    if (feat.ndimension() != 2) {
        AT_ERROR("colors must have 2 dimensions");
    }
    const int num_points = xys.size(0);
    const dim3 block_size = {32, 32, 1};
    const dim3 img_size = {img_width, img_height, 1};
    const dim3 grid_size = {(img_width + block_size.x - 1)/block_size.x,
                            (img_height + block_size.y - 1)/block_size.y, 1};
    
    const int channels = feat.size(1);
    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_scale = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_rot = torch::zeros({num_points, 1}, xys.options());
    torch::Tensor v_feat = torch::zeros({num_points, channels}, xys.options());
    // nd_rasterize_backward_no_tiles_kernel<<<grid_size, block_size>>>(
    //     img_size,
    //     channels,
    //     (float2 *)xys.contiguous().data_ptr<float>(),
    //     (float3 *)conics.contiguous().data_ptr<float>(),
    //     colors.contiguous().data_ptr<float>(),
    //     v_output.contiguous().data_ptr<float>(),
    //     (float2 *)v_xy.contiguous().data_ptr<float>(),
    //     (float3 *)v_conic.contiguous().data_ptr<float>(),
    //     v_colors.contiguous().data_ptr<float>(),
    //     pixel_topk.contiguous().data_ptr<int32_t>()
    // );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(v_xy, v_scale, v_rot, v_feat);
}



std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor,  // dL_dconic
        torch::Tensor  // dL_dcolors
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned BLOCK_H,
        const unsigned BLOCK_W,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &v_output  // dL_dout_color
    ){

    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2 || colors.size(1) != 3) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_W - 1) / BLOCK_W,
        (img_height + BLOCK_H - 1) / BLOCK_H,
        1
    };
    const dim3 block(BLOCK_W, BLOCK_H, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());

    rasterize_backward_kernel<<<tile_bounds, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        tile_bounds,
        img_size,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        (float3 *)v_output.contiguous().data_ptr<float>(),
        // Outputs.
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float3 *)v_colors.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors);
}


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_2d_scale_rot_forward_tensor(
    const int num_points,
    torch::Tensor &means2d,
    torch::Tensor &scales2d,
    torch::Tensor &rotation,
    const unsigned img_height,
    const unsigned img_width,
    const std::tuple<int, int, int> tile_bounds
) {
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);

    // Triangular covariance.
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means2d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means2d.options().dtype(torch::kInt32));

    project_gaussians_2d_scale_rot_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_points,
        (float2 *)means2d.contiguous().data_ptr<float>(),
        (float2 *)scales2d.contiguous().data_ptr<float>(),
        (float *)rotation.contiguous().data_ptr<float>(),
        img_size_dim3,
        tile_bounds_dim3,
        // Outputs.
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(
        xys_d, radii_d, conics_d, num_tiles_hit_d
    );
}


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_2d_scale_rot_backward_tensor(
    const int num_points,
    torch::Tensor &means2d,
    torch::Tensor &scales2d,
    torch::Tensor &rotation,
    const unsigned img_height,
    const unsigned img_width,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_conic
) {
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor v_scale =
        torch::zeros({num_points, 2}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor v_rot =
        torch::zeros({num_points, 1}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean2d =
        torch::zeros({num_points, 2}, means2d.options().dtype(torch::kFloat32));

    project_gaussians_2d_scale_rot_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_points,
        (float2 *)means2d.contiguous().data_ptr<float>(),
        (float2 *)scales2d.contiguous().data_ptr<float>(),
        (float *)rotation.contiguous().data_ptr<float>(),
        img_size_dim3,
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        // Outputs.
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        (float2 *)v_mean2d.contiguous().data_ptr<float>(),
        (float2 *)v_scale.contiguous().data_ptr<float>(),
        (float *)v_rot.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov2d, v_mean2d, v_scale, v_rot);
}

// Leave-one-out loss increase per Gaussian of the top-K normalized rendering (topk_norm_loo_loss_increase): the change of the absolute or
// squared error summed over all pixels and channels, from the top-K lists [H, W, TOP_K] of the forward pass and the target image [C, H, W]
torch::Tensor topk_norm_loo_loss_increase_tensor(
    const unsigned img_height,
    const unsigned img_width,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &pixel_topk,
    const torch::Tensor &targets,
    const bool squared,
    const double eps
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(pixel_topk);
    CHECK_INPUT(targets);
    const int channels = colors.size(1);
    if (channels > MAX_CHANNELS) {
        AT_ERROR("at most MAX_CHANNELS channels");
    }
    TORCH_CHECK(targets.numel() == (int64_t)channels * img_height * img_width, "targets must be [C, H, W]");
    TORCH_CHECK(pixel_topk.numel() == (int64_t)img_height * img_width * TOP_K, "pixel_topk must be [H, W, TOP_K]");
    torch::Tensor loss_increase = torch::zeros({xys.size(0)}, xys.options());
    const int num_pixels = img_height * img_width;
    if (num_pixels == 0) {
        return loss_increase;
    }
    topk_norm_loo_loss_increase<<<(num_pixels + N_THREADS - 1) / N_THREADS, N_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        {img_width, img_height, 1},
        channels,
        (float2 *)xys.data_ptr<float>(),
        (float3 *)conics.data_ptr<float>(),
        colors.data_ptr<float>(),
        pixel_topk.data_ptr<int32_t>(),
        targets.data_ptr<float>(),
        squared,
        (float)eps,
        loss_increase.data_ptr<float>()
    );
    CUDA_CHECK_LAUNCH();
    return loss_increase;
}
