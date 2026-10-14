#include "backward.cuh"
#include "helpers.cuh"
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

// Sum over the 32 lanes of a warp, returned to every lane (all lanes must be active)
inline __device__ float warp_sum_all(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

__global__ void nd_rasterize_backward_topk_norm_kernel(
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
    float* __restrict__ v_rgb,
    int* __restrict__ pixel_topk
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

    // threads outside the image take part in the warp sums with an empty top-K list
    const bool inside = (i < img_size.y && j < img_size.x);

    // df/d_out for this pixel
    const float* v_out = &(v_output[channels * pix_id]);

    // topk gs ids for this pixel and their weights d = exp(-sigma)
    int32_t topk[TOP_K];
    float d_local[TOP_K];
    float denom = EPS;
    int cnt = 0;
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        topk[k] = inside ? pixel_topk[pix_id * TOP_K + k] : -1;
        d_local[k] = 0.f;
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

    // out = sum_k d_k f_k / denom, so dL/dd_l = (f_l . v_out) / denom - sum_k d_k (f_k . v_out) / denom^2
    float v_norm_d[TOP_K];
    float v_d_shared = 0.f;
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        v_norm_d[k] = 0.f;
        int g_id = topk[k];
        if (g_id < 0) continue;
        const float* rgb = &rgbs[channels * g_id];
        for (int c = 0; c < channels; ++c)
            v_norm_d[k] += rgb[c] * v_out[c];
        v_d_shared += -d_local[k] / (denom*denom) * v_norm_d[k];
    }

    // per top-K entry: the weight of its color and dL/dsigma = -d dL/dd
    float rgb_weight_local[TOP_K];
    float v_sigma_local[TOP_K];
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        rgb_weight_local[k] = d_local[k] / denom;
        v_sigma_local[k] = -d_local[k] * (v_norm_d[k] / denom + v_d_shared);
    }

    const int2 range = tile_bins[tile_id];
    if (range.y - range.x > TOPK_WALK_MAX_GAUSSIANS) {
        // crowded tile (the same branch for the whole block): walking its list would cost more than per-pixel atomic adds
        if (!inside) return;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            int g_id = topk[k];
            if (g_id < 0) continue;
            float v_sigma = v_sigma_local[k];
            float3 conic = conics[g_id];
            float2 xy = xys[g_id];
            float2 delta = {xy.x - px, xy.y - py};
            float* v_conic_ptr = (float*)(v_conic);
            atomicAdd(v_conic_ptr + 3*g_id + 0, 0.5f * v_sigma * delta.x * delta.x);
            atomicAdd(v_conic_ptr + 3*g_id + 1, v_sigma * delta.x * delta.y);
            atomicAdd(v_conic_ptr + 3*g_id + 2, 0.5f * v_sigma * delta.y * delta.y);
            float* v_xy_ptr = (float*)(v_xy);
            atomicAdd(v_xy_ptr + 2*g_id + 0, v_sigma * (conic.x * delta.x + conic.y * delta.y));
            atomicAdd(v_xy_ptr + 2*g_id + 1, v_sigma * (conic.y * delta.x + conic.z * delta.y));
            for (int c = 0; c < channels; ++c)
                atomicAdd(v_rgb + channels * g_id + c, rgb_weight_local[k] * v_out[c]);
        }
        return;
    }

    // walk the tile's Gaussians in batches loaded into shared memory, as in the forward pass
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    const int tr = block.thread_rank();
    const int lane = tr % 32;
    // values per Gaussian, lane v adds value v: v_conic (0-2), v_xy (3-4), v_rgb (5-)
    const int num_values = 5 + channels;
    int remaining = cnt;  // top-K Gaussians of this pixel not reached yet

    for (int b = 0; b < num_batches; ++b) {
        // end early once every pixel of the tile has reached all of its top-K Gaussians
        if (__syncthreads_count(remaining > 0) == 0) {
            break;
        }
        int batch_start = range.x + block_size * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussians_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            conic_batch[tr] = conics[g_id];
        }
        block.sync();

        // all branches below are uniform within the warp
        int batch_size = min(block_size, range.y - batch_start);
        for (int t = 0; t < batch_size; ++t) {
            if (!__any_sync(0xffffffff, remaining > 0))
                break;
            const int32_t g_id = id_batch[t];
            float rgb_weight = 0.f;
            float v_sigma = 0.f;
            bool hit = false;
            #pragma unroll
            for (int k = 0; k < TOP_K; ++k) {
                if (topk[k] == g_id) {
                    rgb_weight = rgb_weight_local[k];
                    v_sigma = v_sigma_local[k];
                    hit = true;
                }
            }
            if (!__any_sync(0xffffffff, hit))
                continue;
            remaining -= hit;

            const float3 conic = conic_batch[t];
            const float2 xy = xy_batch[t];
            const float2 delta = {xy.x - px, xy.y - py};
            float value = 0.f;
            float sum;
            sum = warp_sum_all(0.5f * v_sigma * delta.x * delta.x);
            if (lane == 0) value = sum;
            sum = warp_sum_all(v_sigma * delta.x * delta.y);
            if (lane == 1) value = sum;
            sum = warp_sum_all(0.5f * v_sigma * delta.y * delta.y);
            if (lane == 2) value = sum;
            sum = warp_sum_all(v_sigma * (conic.x * delta.x + conic.y * delta.y));
            if (lane == 3) value = sum;
            sum = warp_sum_all(v_sigma * (conic.y * delta.x + conic.z * delta.y));
            if (lane == 4) value = sum;
            for (int c = 0; c < channels; ++c) {
                sum = warp_sum_all(rgb_weight * v_out[c]);
                if (lane == 5 + c) value = sum;
            }
            if (lane < num_values) {
                float* target = lane < 3 ? (float*)v_conic + 3 * g_id + lane
                              : lane < 5 ? (float*)v_xy + 2 * g_id + (lane - 3)
                              : v_rgb + channels * g_id + (lane - 5);
                atomicAdd(target, value);
            }
        }
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
