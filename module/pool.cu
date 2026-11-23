#include <cuda_runtime.h>

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "pool.cuh"
#include "tensor.cuh"

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t _err = (call);                                                                \
        if (_err != cudaSuccess) {                                                                \
            fprintf(stderr, "[cuda] %s failed at %s:%d: %s\n", #call, __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                                    \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

namespace {

/* Attached to the pooled output's meta (the output tensor is also the backward
 * context), so backward can recover the window geometry. */
typedef struct PoolMeta {
    int kernel;
    int stride;
} PoolMeta;

/* Offset of the first maximum in the window of output (b,c,ho,wo). Forward and
 * backward both call this, so they always agree on which element won a tie. */
__device__ int window_argmax(const float *x, int plane_off, int H, int W, int k, int stride, int ho, int wo) {
    int best = -1;
    float best_v = -FLT_MAX;
    for (int ky = 0; ky < k; ++ky) {
        for (int kx = 0; kx < k; ++kx) {
            int off = plane_off + (ho * stride + ky) * W + (wo * stride + kx);
            if (best < 0 || x[off] > best_v) {
                best = off;
                best_v = x[off];
            }
        }
    }
    (void)H;
    return best;
}

__global__ void maxpool_forward_kernel(const float *x, float *y, int BC, int H, int W, int k, int stride, int Ho,
                                       int Wo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BC * Ho * Wo) return;
    int wo = i % Wo, ho = (i / Wo) % Ho, bc = i / (Wo * Ho);
    y[i] = x[window_argmax(x, bc * H * W, H, W, k, stride, ho, wo)];
}

__global__ void maxpool_backward_kernel(const float *x, const float *dy, float *dx, int BC, int H, int W, int k,
                                        int stride, int Ho, int Wo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BC * Ho * Wo) return;
    int wo = i % Wo, ho = (i / Wo) % Ho, bc = i / (Wo * Ho);
    atomicAdd(&dx[window_argmax(x, bc * H * W, H, W, k, stride, ho, wo)], dy[i]); /* windows may overlap */
}

__global__ void gap_forward_kernel(const float *x, float *y, int BC, int HW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BC) return;
    float sum = 0.0f;
    for (int p = 0; p < HW; ++p) sum += x[(size_t)i * HW + p];
    y[i] = sum / (float)HW;
}

__global__ void gap_backward_kernel(const float *dy, float *dx, int BC, int HW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BC * HW) return;
    dx[i] = dy[i / HW] / (float)HW;
}

/* one thread per output value */
__global__ void upsample_forward_kernel(const float *x, float *y, int BC, int H, int W, int f) {
    int Ho = H * f, Wo = W * f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BC * Ho * Wo) return;
    int wo = i % Wo, ho = (i / Wo) % Ho, bc = i / (Wo * Ho);
    y[i] = x[(bc * H + ho / f) * W + wo / f];
}

/* one thread per INPUT value: sums its f x f block of the gradient, so no atomics are needed */
__global__ void upsample_backward_kernel(const float *dy, float *dx, int BC, int H, int W, int f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BC * H * W) return;
    int w = i % W, h = (i / W) % H, bc = i / (W * H);
    int Wo = W * f, Ho = H * f;
    float sum = 0.0f;
    for (int dy_ = 0; dy_ < f; ++dy_)
        for (int dx_ = 0; dx_ < f; ++dx_) sum += dy[(bc * Ho + h * f + dy_) * Wo + w * f + dx_];
    dx[i] = sum;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

Tensor *upsample_nearest_forward(Tensor *x, int factor) {
    if (x->ndim != 4 || factor < 1) {
        printf("upsample_nearest_forward needs a 4D tensor and factor >= 1.\n");
        return NULL;
    }
    int B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int shape[4] = {B, C, H * factor, W * factor};
    Tensor *y = tensor_create(4, shape, x->requires_grad);
    int n = B * C * H * factor * W * factor;
    upsample_forward_kernel<<<(n + 255) / 256, 256>>>(x->data, y->data, B * C, H, W, factor);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (x->requires_grad) {
        PoolMeta *meta = (PoolMeta *)malloc(sizeof(PoolMeta));
        meta->kernel = factor;
        meta->stride = factor;
        y->meta = meta;
        tensor_add_dependency_with_ctx(y, x, y, upsample_nearest_backward);
    }
    return y;
}

Tensor *upsample_nearest_backward(Tensor *x, Tensor *out, Tensor *grad_out) {
    PoolMeta *meta = (PoolMeta *)out->meta;
    int B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    Tensor *grad_x = tensor_create(4, x->shape, 0);
    int n = B * C * H * W;
    upsample_backward_kernel<<<(n + 255) / 256, 256>>>(grad_out->data, grad_x->data, B * C, H, W, meta->kernel);
    CUDA_CHECK(cudaDeviceSynchronize());
    return grad_x;
}

Tensor *maxpool2d_forward(Tensor *x, int kernel, int stride) {
    if (x->ndim != 4) {
        printf("maxpool2d_forward only supports 4D tensors (B, C, H, W).\n");
        return NULL;
    }
    int B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int Ho = (H - kernel) / stride + 1, Wo = (W - kernel) / stride + 1;
    if (kernel < 1 || stride < 1 || H < kernel || W < kernel || Ho < 1 || Wo < 1) {
        printf("maxpool2d_forward: window %d (stride %d) does not fit a %dx%d input\n", kernel, stride, H, W);
        return NULL;
    }
    int shape[4] = {B, C, Ho, Wo};
    Tensor *y = tensor_create(4, shape, x->requires_grad);
    int n = B * C * Ho * Wo;
    maxpool_forward_kernel<<<(n + 255) / 256, 256>>>(x->data, y->data, B * C, H, W, kernel, stride, Ho, Wo);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (x->requires_grad) {
        PoolMeta *meta = (PoolMeta *)malloc(sizeof(PoolMeta));
        meta->kernel = kernel;
        meta->stride = stride;
        y->meta = meta;
        tensor_add_dependency_with_ctx(y, x, y, maxpool2d_backward);
    }
    return y;
}

Tensor *maxpool2d_backward(Tensor *x, Tensor *out, Tensor *grad_out) {
    PoolMeta *meta = (PoolMeta *)out->meta;
    int B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int Ho = out->shape[2], Wo = out->shape[3];
    Tensor *grad_x = tensor_create(4, x->shape, 0);
    int n = B * C * Ho * Wo;
    maxpool_backward_kernel<<<(n + 255) / 256, 256>>>(x->data, grad_out->data, grad_x->data, B * C, H, W, meta->kernel,
                                                      meta->stride, Ho, Wo);
    CUDA_CHECK(cudaDeviceSynchronize());
    return grad_x;
}

Tensor *global_avg_pool_forward(Tensor *x) {
    if (x->ndim != 4) {
        printf("global_avg_pool_forward only supports 4D tensors (B, C, H, W).\n");
        return NULL;
    }
    int B = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];
    int shape[2] = {B, C};
    Tensor *y = tensor_create(2, shape, x->requires_grad);
    gap_forward_kernel<<<(B * C + 255) / 256, 256>>>(x->data, y->data, B * C, HW);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (x->requires_grad) tensor_add_dependency(y, x, global_avg_pool_backward);
    return y;
}

Tensor *global_avg_pool_backward(Tensor *x, Tensor *ctx, Tensor *grad_out) {
    (void)ctx;
    int BC = x->shape[0] * x->shape[1], HW = x->shape[2] * x->shape[3];
    Tensor *grad_x = tensor_create(4, x->shape, 0);
    gap_backward_kernel<<<(BC * HW + 255) / 256, 256>>>(grad_out->data, grad_x->data, BC, HW);
    CUDA_CHECK(cudaDeviceSynchronize());
    return grad_x;
}

#ifdef __cplusplus
}
#endif
