// Rate term of the training loss in three kernels (compress/rate.py, rate_loss): the smooth Laplace code lengths of the feature residuals of
// the prediction from causal neighbors and of the canonical rotations, with their gradients, as compress.rate.rate_bits computes them in about
// 70 PyTorch kernels. The forward pass also stores each Gaussian's gradient with respect to its own coded features, so that the backward pass
// only gathers them over the Gaussians that predict from it (no atomics, so the gradients are deterministic).
#include "bindings.h"
#include "config.h"
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>

struct RateChannels {
    int num;
    int color[MAX_CHANNELS]; // Reference channel subtracted from the channel before the prediction (-1: none)
    int pred[MAX_CHANNELS];  // Reference channel of the CCLM term (the channel itself without one, where the slope is 0)
};

// The values a channel is predicted in: the features minus their color references
template <int CH>
__device__ inline void rate_coded_values(const float *__restrict__ feat, const RateChannels &channels, const int num, float *values) {
#pragma unroll
    for (int c = 0; c < (CH > 0 ? CH : MAX_CHANNELS); ++c) {
        if (c < num) {
            values[c] = feat[c] - (channels.color[c] >= 0 ? feat[channels.color[c]] : 0.0f);
        }
    }
}

// One thread per Gaussian: bits summed per block (block_bits), the gradient of the Gaussian's bits with respect to its own features
// (own_grad [P, C], of which the neighbors take minus their weight times it) and to its rotation (rot_grad [P]). CH is the number of channels
// (0: channels.num, at most MAX_CHANNELS); with a fixed number, the per-channel arrays stay in registers, as the reference channels are
// selected rather than indexed.
template <int CH>
__global__ void rate_loss_forward_kernel(
    const int num_gaussians,
    const int num_neighbors,
    const RateChannels channels,
    const float *__restrict__ feat,
    const float *__restrict__ rot,
    const int32_t *__restrict__ neighbors,
    const float *__restrict__ weights,
    const float *__restrict__ slope,
    const float *__restrict__ inv_steps,
    const float *__restrict__ feat_coeffs,
    const float *__restrict__ rot_coeffs,
    const float inv_rot_step,
    float *__restrict__ own_grad,
    float *__restrict__ rot_grad,
    float *__restrict__ block_bits
) {
    constexpr int SIZE = CH > 0 ? CH : MAX_CHANNELS;
    typedef cub::BlockReduce<float, N_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num = CH > 0 ? CH : channels.num;
    float bits = 0.0f;
    if (idx < num_gaussians) {
        float values[SIZE], mean[SIZE], neighbor[SIZE], grad[SIZE], coded_grad[SIZE];
        rate_coded_values<CH>(feat + idx * num, channels, num, values);
#pragma unroll
        for (int c = 0; c < SIZE; ++c) {
            mean[c] = 0.0f;
        }
        for (int k = 0; k < num_neighbors; ++k) {
            const float weight = weights[idx * num_neighbors + k];
            if (weight == 0.0f) {
                continue;
            }
            rate_coded_values<CH>(feat + neighbors[idx * num_neighbors + k] * num, channels, num, neighbor);
#pragma unroll
            for (int c = 0; c < SIZE; ++c) {
                if (c < num) {
                    mean[c] += weight * neighbor[c];
                }
            }
        }
        // Residual r in steps, bits (sqrt(r^2 + 1/4) - 1/2) times the coefficient, and the gradient with respect to the residual's own term,
        // then through the CCLM term to the reference channels
#pragma unroll
        for (int c = 0; c < SIZE; ++c) {
            coded_grad[c] = 0.0f;
        }
#pragma unroll
        for (int c = 0; c < SIZE; ++c) {
            if (c < num) {
                float reference = 0.0f;
#pragma unroll
                for (int p = 0; p < SIZE; ++p) {
                    reference = p == channels.pred[c] ? values[p] - mean[p] : reference;
                }
                const float s = slope[idx * num + c];
                const float r = (values[c] - mean[c] - s * reference) * inv_steps[c];
                const float root = sqrtf(r * r + 0.25f);
                bits += (root - 0.5f) * feat_coeffs[c];
                grad[c] = feat_coeffs[c] * r / root * inv_steps[c];
                coded_grad[c] += grad[c];
#pragma unroll
                for (int p = 0; p < SIZE; ++p) {
                    coded_grad[p] -= p == channels.pred[c] ? s * grad[c] : 0.0f;
                }
            }
        }
        // Through the color references to the features
#pragma unroll
        for (int c = 0; c < SIZE; ++c) {
            if (c < num) {
                float feat_grad = coded_grad[c];
#pragma unroll
                for (int q = 0; q < SIZE; ++q) {
                    feat_grad -= q < num && channels.color[q] == c ? coded_grad[q] : 0.0f;
                }
                own_grad[idx * num + c] = feat_grad;
            }
        }
        // Canonical rotation in steps (within [-pi/4, pi/4] by quarter turns, which carry no gradient)
        const float angle = rot[idx];
        const float t = (angle - rintf(angle * (float)M_2_PI) * (float)M_PI_2) * inv_rot_step;
        const float root = sqrtf(t * t + 0.25f);
        bits += (root - 0.5f) * rot_coeffs[idx];
        rot_grad[idx] = rot_coeffs[idx] * t / root * inv_rot_step;
    }
    const float block_sum = BlockReduce(temp_storage).Sum(bits);
    if (threadIdx.x == 0) {
        block_bits[blockIdx.x] = block_sum;
    }
}

// One block: the loss (factor / max(l1_loss / l1_ratio, 1e-4) times the bits, summed in a fixed order) and that weight
__global__ void rate_loss_reduce_kernel(
    const int num_blocks,
    const float *__restrict__ block_bits,
    const float *__restrict__ l1_loss,
    const float l1_ratio,
    const float factor,
    float *__restrict__ loss,
    float *__restrict__ loss_weight
) {
    typedef cub::BlockReduce<float, N_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float bits = 0.0f;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        bits += block_bits[b];
    }
    const float total = BlockReduce(temp_storage).Sum(bits);
    if (threadIdx.x == 0) {
        const float weight = factor / fmaxf(l1_loss[0] / l1_ratio, 1e-4f);
        loss[0] = weight * total;
        loss_weight[0] = weight;
    }
}

// One thread per Gaussian: its own gradient minus the weighted ones of the Gaussians that predict from it (sources[offsets[j]:offsets[j+1]]
// with their neighbor weights), times the loss weight and the incoming gradient; CH as rate_loss_forward_kernel
template <int CH>
__global__ void rate_loss_backward_kernel(
    const int num_gaussians,
    const int channels,
    const float *__restrict__ grad_loss,
    const float *__restrict__ loss_weight,
    const float *__restrict__ own_grad,
    const float *__restrict__ rot_grad,
    const int32_t *__restrict__ offsets,
    const int32_t *__restrict__ sources,
    const float *__restrict__ source_weights,
    float *__restrict__ grad_feat,
    float *__restrict__ grad_rot
) {
    constexpr int SIZE = CH > 0 ? CH : MAX_CHANNELS;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) {
        return;
    }
    const int num = CH > 0 ? CH : channels;
    const float scale = grad_loss[0] * loss_weight[0];
    float grad[SIZE];
#pragma unroll
    for (int c = 0; c < SIZE; ++c) {
        grad[c] = c < num ? own_grad[idx * num + c] : 0.0f;
    }
    for (int e = offsets[idx]; e < offsets[idx + 1]; ++e) {
        const int source = sources[e];
        const float weight = source_weights[e];
#pragma unroll
        for (int c = 0; c < SIZE; ++c) {
            if (c < num) {
                grad[c] -= weight * own_grad[source * num + c];
            }
        }
    }
#pragma unroll
    for (int c = 0; c < SIZE; ++c) {
        if (c < num) {
            grad_feat[idx * num + c] = scale * grad[c];
        }
    }
    grad_rot[idx] = scale * rot_grad[idx];
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rate_loss_forward_tensor(
    const torch::Tensor &feat,
    const torch::Tensor &rot,
    const torch::Tensor &l1_loss,
    const torch::Tensor &neighbors,
    const torch::Tensor &weights,
    const torch::Tensor &slope,
    const std::vector<int64_t> &color_refs,
    const std::vector<int64_t> &pred_refs,
    const torch::Tensor &inv_steps,
    const torch::Tensor &feat_coeffs,
    const torch::Tensor &rot_coeffs,
    const double inv_rot_step,
    const double l1_ratio,
    const double factor
) {
    CHECK_INPUT(feat);
    CHECK_INPUT(rot);
    CHECK_INPUT(l1_loss);
    CHECK_INPUT(neighbors);
    CHECK_INPUT(weights);
    CHECK_INPUT(slope);
    CHECK_INPUT(inv_steps);
    CHECK_INPUT(feat_coeffs);
    CHECK_INPUT(rot_coeffs);
    const int num_gaussians = feat.size(0);
    const int num_channels = feat.size(1);
    TORCH_CHECK(num_channels <= MAX_CHANNELS, "at most MAX_CHANNELS feature channels");
    TORCH_CHECK(color_refs.size() == (size_t)num_channels && pred_refs.size() == (size_t)num_channels, "one reference per channel");
    TORCH_CHECK(rot.numel() == num_gaussians && rot_coeffs.numel() == num_gaussians, "one rotation per Gaussian");
    TORCH_CHECK(neighbors.size(0) == num_gaussians && weights.sizes() == neighbors.sizes(), "neighbors and weights must be [P, K]");
    TORCH_CHECK(neighbors.scalar_type() == torch::kInt32, "neighbors must be int32");
    RateChannels channels;
    channels.num = num_channels;
    for (int c = 0; c < num_channels; ++c) {
        TORCH_CHECK(color_refs[c] < num_channels && pred_refs[c] >= 0 && pred_refs[c] < num_channels, "invalid reference channel");
        channels.color[c] = (int)color_refs[c];
        channels.pred[c] = (int)pred_refs[c];
    }
    const int num_blocks = (num_gaussians + N_THREADS - 1) / N_THREADS;
    torch::Tensor own_grad = torch::empty_like(feat);
    torch::Tensor rot_grad = torch::empty({num_gaussians}, feat.options());
    torch::Tensor block_bits = torch::empty({num_blocks}, feat.options());
    torch::Tensor loss = torch::empty({}, feat.options());
    torch::Tensor loss_weight = torch::empty({1}, feat.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_blocks > 0) {
        auto kernel = num_channels == 3 ? rate_loss_forward_kernel<3> : rate_loss_forward_kernel<0>;
        kernel<<<num_blocks, N_THREADS, 0, stream>>>(
            num_gaussians,
            neighbors.size(1),
            channels,
            feat.data_ptr<float>(),
            rot.data_ptr<float>(),
            neighbors.data_ptr<int32_t>(),
            weights.data_ptr<float>(),
            slope.data_ptr<float>(),
            inv_steps.data_ptr<float>(),
            feat_coeffs.data_ptr<float>(),
            rot_coeffs.data_ptr<float>(),
            (float)inv_rot_step,
            own_grad.data_ptr<float>(),
            rot_grad.data_ptr<float>(),
            block_bits.data_ptr<float>()
        );
        CUDA_CHECK_LAUNCH();
    }
    rate_loss_reduce_kernel<<<1, N_THREADS, 0, stream>>>(
        num_blocks,
        block_bits.data_ptr<float>(),
        l1_loss.data_ptr<float>(),
        (float)l1_ratio,
        (float)factor,
        loss.data_ptr<float>(),
        loss_weight.data_ptr<float>()
    );
    CUDA_CHECK_LAUNCH();
    return std::make_tuple(loss, loss_weight, own_grad, rot_grad);
}

std::tuple<torch::Tensor, torch::Tensor> rate_loss_backward_tensor(
    const torch::Tensor &grad_loss,
    const torch::Tensor &loss_weight,
    const torch::Tensor &own_grad,
    const torch::Tensor &rot_grad,
    const torch::Tensor &offsets,
    const torch::Tensor &sources,
    const torch::Tensor &source_weights
) {
    CHECK_INPUT(grad_loss);
    CHECK_INPUT(loss_weight);
    CHECK_INPUT(own_grad);
    CHECK_INPUT(rot_grad);
    CHECK_INPUT(offsets);
    CHECK_INPUT(sources);
    CHECK_INPUT(source_weights);
    const int num_gaussians = own_grad.size(0);
    const int num_channels = own_grad.size(1);
    TORCH_CHECK(offsets.numel() == num_gaussians + 1, "offsets must be [P + 1]");
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32 && sources.scalar_type() == torch::kInt32, "offsets and sources must be int32");
    torch::Tensor grad_feat = torch::empty_like(own_grad);
    torch::Tensor grad_rot = torch::empty_like(rot_grad);
    if (num_gaussians > 0) {
        auto kernel = num_channels == 3 ? rate_loss_backward_kernel<3> : rate_loss_backward_kernel<0>;
        kernel<<<(num_gaussians + N_THREADS - 1) / N_THREADS, N_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_gaussians,
            num_channels,
            grad_loss.data_ptr<float>(),
            loss_weight.data_ptr<float>(),
            own_grad.data_ptr<float>(),
            rot_grad.data_ptr<float>(),
            offsets.data_ptr<int32_t>(),
            sources.data_ptr<int32_t>(),
            source_weights.data_ptr<float>(),
            grad_feat.data_ptr<float>(),
            grad_rot.data_ptr<float>()
        );
        CUDA_CHECK_LAUNCH();
    }
    return std::make_tuple(grad_feat, grad_rot);
}
