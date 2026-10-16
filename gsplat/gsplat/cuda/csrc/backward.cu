#include "backward.cuh"
#include "helpers.cuh"
#include <climits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}

__global__ void nd_rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb
) {

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float rgbs_batch[BLOCK_SIZE][MAX_CHANNELS];

    // df/d_out for this pixel
    const float *v_out = &(v_output[channels * pix_id]);
    
    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing

    bool valid = inside;
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from front to back
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussians_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            xy_batch[tr] = {xy.x, xy.y};
            conic_batch[tr] = conics[g_id];
            for (int c = 0; c < channels; ++c) 
                rgbs_batch[tr][c] = rgbs[channels * g_id + c];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; t < batch_size; ++t) {

            float3 conic = conic_batch[t];
            float2 xy = xy_batch[t];
            float2 delta = {xy.x - px, xy.y - py};
            float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            float d = __expf(-sigma);
            if (sigma < 0.f || isnan(sigma) || isinf(sigma)) {
                valid = 0;
            }
            
            float  v_rgb_local[MAX_CHANNELS] = {0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            
            if (valid) {
                // update v_rgb for this gaussian
                for (int c = 0; c < channels; ++c)
                    v_rgb_local[c] = d * v_out[c];

                const float* rgb = rgbs_batch[t];
                // update v_sigma for this gaussian
                float v_sigma = 0.f;
                for (int c = 0; c < channels; ++c)
                    v_sigma += rgb[c] * v_out[c];
                v_sigma *= -d;
                
                // update v_conic for this gaussian
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                        v_sigma * delta.x * delta.y, 
                                 0.5f * v_sigma * delta.y * delta.y};
                // update v_xy for this gaussian
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                              v_sigma * (conic.y * delta.x + conic.z * delta.y)};
            }
            
            // sum across the warp
            for (int c = 0; c < channels; ++c)
                warpSum(v_rgb_local[c], warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);

            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                for (int c = 0; c < channels; ++c)
                    atomicAdd(v_rgb_ptr + channels * g + c, v_rgb_local[c]);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
            }
        }
    }
}

// Sums NV values (a power of two, at most 32) over the 32 lanes of a warp by recursive halving: NV - 1 shuffles for the halving plus
// log2(32 / NV) for the final sum, instead of 5 per value. Lane l receives the total of value warp_reduce_scatter_index<NV>(l), which
// lanes with l % (32 / NV) == 0 hold without repetition. All lanes must be active.
template <int NV>
inline __device__ float warp_reduce_scatter(float (&a)[NV], const int lane) {
    #pragma unroll
    for (int half = NV / 2, offset = 16; half >= 1; half /= 2, offset /= 2) {
        const bool upper = lane & offset;
        #pragma unroll
        for (int v = 0; v < half; ++v)
            a[v] = (upper ? a[v + half] : a[v]) + __shfl_xor_sync(0xffffffff, upper ? a[v] : a[v + half], offset);
    }
    float sum = a[0];
    #pragma unroll
    for (int offset = 16 / NV; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    return sum;
}

template <int NV>
inline __device__ int warp_reduce_scatter_index(const int lane) {
    int index = 0;
    #pragma unroll
    for (int half = NV / 2, offset = 16; half >= 1; half /= 2, offset /= 2)
        if (lane & offset) index += half;
    return index;
}

// Gradients of the top-K normalized rendering out = sum_k d_k f_k / (sum_k d_k + EPS), d_k = exp(-sigma_k), for the top-K Gaussians that
// the forward pass stored per pixel. Each warp covers an 8x4 patch of its tile and visits the union of its pixels' top-K Gaussians in
// increasing ID order (a warp-wide minimum per step), so the cost grows with that union (about 10-16 Gaussians at 480p/720p), not with
// the tile's list. Per Gaussian, the warp sums its NV gradient values (v_conic 0-2, v_xy 3-4, v_rgb 5-, zero padding) with one
// reduce-scatter and adds them with one atomic per value.
template <int NV>
__global__ void nd_rasterize_backward_topk_norm_kernel(
    const dim3 img_size,
    const unsigned channels,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    const int32_t* __restrict__ pixel_topk
) {
    constexpr int MAX_C = NV - 5;
    const int tr = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp = tr / 32;
    const int lane = tr % 32;
    const int patches_x = blockDim.x / 8;
    const unsigned j = blockIdx.x * blockDim.x + (warp % patches_x) * 8 + lane % 8;
    const unsigned i = blockIdx.y * blockDim.y + (warp / patches_x) * 4 + lane / 8;
    const float px = (float)j;
    const float py = (float)i;
    // threads outside the image take part in the warp steps with an empty top-K list
    const bool inside = (i < img_size.y && j < img_size.x);
    const int32_t num_pixels = img_size.x * img_size.y;
    const int32_t pix_id = inside ? i * img_size.x + j : 0;

    // df/d_out for this pixel, from the channel-planar [C, H, W] gradient
    float v_out[MAX_C];
    #pragma unroll
    for (int c = 0; c < MAX_C; ++c)
        v_out[c] = (inside && c < channels) ? v_output[c * num_pixels + pix_id] : 0.f;

    // top-K Gaussian IDs of this pixel (INT_MAX for none) and their weights d = exp(-sigma)
    int32_t ids[TOP_K];
    float d_local[TOP_K];
    float denom = EPS;
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        const int32_t g_id = inside ? pixel_topk[pix_id * TOP_K + k] : -1;
        ids[k] = g_id < 0 ? INT_MAX : g_id;
        d_local[k] = 0.f;
        if (g_id < 0) continue;
        const float3 conic = conics[g_id];
        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};
        const float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
        const float d = __expf(-sigma);
        denom += d;
        d_local[k] = d;
    }

    // out = sum_k d_k f_k / denom, so dL/dd_l = (f_l . v_out) / denom - sum_k d_k (f_k . v_out) / denom^2
    float v_norm_d[TOP_K];
    float v_d_shared = 0.f;
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        v_norm_d[k] = 0.f;
        if (ids[k] == INT_MAX) continue;
        const float* rgb = &rgbs[channels * ids[k]];
        #pragma unroll
        for (int c = 0; c < MAX_C; ++c)
            if (c < channels) v_norm_d[k] += rgb[c] * v_out[c];
        v_d_shared += -d_local[k] / (denom * denom) * v_norm_d[k];
    }

    // per top-K entry: the weight of its color and dL/dsigma = -d dL/dd
    float rgb_weight_local[TOP_K];
    float v_sigma_local[TOP_K];
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        rgb_weight_local[k] = d_local[k] / denom;
        v_sigma_local[k] = -d_local[k] * (v_norm_d[k] / denom + v_d_shared);
    }

    const int value_index = warp_reduce_scatter_index<NV>(lane);
    const bool adds = lane % (32 / NV) == 0 && value_index < 5 + (int)channels;
    float* target = value_index < 3 ? (float*)v_conic + value_index
                  : value_index < 5 ? (float*)v_xy + (value_index - 3)
                  : v_rgb + (value_index - 5);
    const int target_stride = value_index < 3 ? 3 : value_index < 5 ? 2 : channels;
    while (true) {
        int32_t head = ids[0];
        #pragma unroll
        for (int k = 1; k < TOP_K; ++k)
            head = min(head, ids[k]);
        const int32_t g_id = __reduce_min_sync(0xffffffff, head);
        if (g_id == INT_MAX)
            break;
        float rgb_weight = 0.f;
        float v_sigma = 0.f;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            if (ids[k] == g_id) {
                rgb_weight = rgb_weight_local[k];
                v_sigma = v_sigma_local[k];
                ids[k] = INT_MAX;
            }
        }
        // lanes without this Gaussian add zeros
        const float3 conic = conics[g_id];
        const float2 xy = xys[g_id];
        const float2 delta = {xy.x - px, xy.y - py};
        float values[NV];
        values[0] = 0.5f * v_sigma * delta.x * delta.x;
        values[1] = v_sigma * delta.x * delta.y;
        values[2] = 0.5f * v_sigma * delta.y * delta.y;
        values[3] = v_sigma * (conic.x * delta.x + conic.y * delta.y);
        values[4] = v_sigma * (conic.y * delta.x + conic.z * delta.y);
        #pragma unroll
        for (int c = 0; c < MAX_C; ++c)
            values[5 + c] = rgb_weight * v_out[c];
        const float sum = warp_reduce_scatter<NV>(values, lane);
        if (adds)
            atomicAdd(target + target_stride * g_id, sum);
    }
}

void nd_rasterize_backward_topk_norm(
    const dim3 tile_bounds,
    const dim3 block,
    const dim3 img_size,
    const unsigned channels,
    const float2* xys,
    const float3* conics,
    const float* rgbs,
    const float* v_output,
    float2* v_xy,
    float3* v_conic,
    float* v_rgb,
    const int32_t* pixel_topk,
    cudaStream_t stream
) {
    if (5 + channels <= 8) {
        nd_rasterize_backward_topk_norm_kernel<8><<<tile_bounds, block, 0, stream>>>(
            img_size, channels, xys, conics, rgbs, v_output, v_xy, v_conic, v_rgb, pixel_topk);
    } else if (5 + channels <= 16) {
        nd_rasterize_backward_topk_norm_kernel<16><<<tile_bounds, block, 0, stream>>>(
            img_size, channels, xys, conics, rgbs, v_output, v_xy, v_conic, v_rgb, pixel_topk);
    } else {
        nd_rasterize_backward_topk_norm_kernel<32><<<tile_bounds, block, 0, stream>>>(
            img_size, channels, xys, conics, rgbs, v_output, v_xy, v_conic, v_rgb, pixel_topk);
    }
}

__global__ void nd_rasterize_backward_no_tiles_kernel(
    const dim3 img_size,
    const unsigned channels,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    int* __restrict__ pixel_topk
) {
    
    auto block = cg::this_thread_block();
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);
    if (!inside) return;

    // df/d_out for this pixel
    const float* v_out = &(v_output[channels * pix_id]);
    // topk gs id for this pixel
    const int* topk = &pixel_topk[pix_id * TOP_K];
    
    // compute the normalization factor

    float d_local[TOP_K] = {0.0f};
    float denom = EPS_no_tiles;
    int cnt = 0;
    for (int k = 0; k < TOP_K; ++k) {
        int g_id = topk[k];
        if (g_id < 0) continue;
        ++cnt;
        float3 conic = conics[g_id];
        float2 xy = xys[g_id];
        float2 delta = {xy.x - px, xy.y - py};
        float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
        float d = __expf(-sigma);
        denom += d;
        d_local[k] = d;
    }
    
    float v_d_local[TOP_K] = {0.f};

    // compute each gaussian's contribution to the gradient
    for (int k = 0; k < TOP_K; ++k) {
        int g_id = topk[k];
        if (g_id < 0) continue;

        float d = d_local[k];
        float norm_d = d / denom;
        float v_rgb_local[MAX_CHANNELS] = {0.f};

        // update v_rgb for this gaussian
        for (int c = 0; c < channels; ++c)
            v_rgb_local[c] = norm_d * v_out[c];
        
        const float* rgb = &rgbs[channels * g_id];
        float* v_rgb_ptr = (float*)(v_rgb);
        for (int c = 0; c < channels; ++c)
            atomicAdd(v_rgb_ptr + channels * g_id + c, v_rgb_local[c]);
        
        float v_norm_d = 0.f;
        for (int c = 0; c < channels; ++c)
            v_norm_d += rgb[c] * v_out[c];

        float tmp = -d / (denom*denom) * v_norm_d;
        for (int l = 0; l < TOP_K; ++l) {
            if (l == k) {
                v_d_local[l] += v_norm_d/denom + tmp;
            } 
            else {
                v_d_local[l] += tmp;
            }
        }        
    }

    for (int k = 0; k < TOP_K; ++k) {
        int g_id = topk[k];
        if (g_id < 0) continue;

        const float* rgb = &rgbs[channels * g_id];
        float v_sigma = 0.f;
        v_sigma = v_d_local[k];
        v_sigma *= -d_local[k];

        float3 conic = conics[g_id];
        float2 xy = xys[g_id];
        float2 delta = {xy.x - px, xy.y - py};
        
        // update v_conic for this gaussian
        float3 v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                v_sigma * delta.x * delta.y, 
                         0.5f * v_sigma * delta.y * delta.y};
        
        // update v_xy for this gaussian
        float2 v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                      v_sigma * (conic.y * delta.x + conic.z * delta.y)};
        

        float* v_conic_ptr = (float*)(v_conic);
        atomicAdd(v_conic_ptr + 3*g_id + 0, v_conic_local.x);
        atomicAdd(v_conic_ptr + 3*g_id + 1, v_conic_local.y);
        atomicAdd(v_conic_ptr + 3*g_id + 2, v_conic_local.z);
        
        float* v_xy_ptr = (float*)(v_xy);
        atomicAdd(v_xy_ptr + 2*g_id + 0, v_xy_local.x);
        atomicAdd(v_xy_ptr + 2*g_id + 1, v_xy_local.y);
    }
}

__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float3* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float3 rgbs_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    bool valid = inside;
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from front to back
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            xy_batch[tr] = {xy.x, xy.y};
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; t < batch_size; ++t) {

            float3 conic = conic_batch[t];
            float2 xy = xy_batch[t];
            float2 delta = {xy.x - px, xy.y - py};
            float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            float d = __expf(-sigma);
            if (sigma < 0.f || isnan(sigma) || isinf(sigma)) {
                valid = 0;
            }
            
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            
            if (valid) {
                // update v_rgb for this gaussian
                v_rgb_local = {d * v_out.x, d * v_out.y, d * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // update v_sigma for this gaussian
                const float v_sigma = (
                    rgb.x * v_out.x + 
                    rgb.y * v_out.y + 
                    rgb.z * v_out.z
                ) * (-d);
                // update v_conic for this gaussian
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                        v_sigma * delta.x * delta.y, 
                                 0.5f * v_sigma * delta.y * delta.y};
                // update v_xy for this gaussian
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                              v_sigma * (conic.y * delta.x + conic.z * delta.y)};
            }
            
            // sum across the warp
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);

            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
            }
        }
    }
}


// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float* __restrict__ v_cov3d,
    float3& __restrict__ v_scale,
    float4& __restrict__ v_quat
) {
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    glm::mat3 v_V = glm::mat3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    glm::mat3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    glm::mat3 v_M = 2.f * v_V * M;
    // glm::mat3 v_S = glm::transpose(R) * v_M;
    v_scale.x = (float)glm::dot(R[0], v_M[0]);
    v_scale.y = (float)glm::dot(R[1], v_M[1]);
    v_scale.z = (float)glm::dot(R[2], v_M[2]);

    glm::mat3 v_R = v_M * S;
    v_quat = quat_to_rotmat_vjp(quat, v_R);
}
