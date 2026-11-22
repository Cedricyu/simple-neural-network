#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "batchnorm.cuh"
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

#define BN_THREADS 256

namespace {

/* Attached to gamma->meta so the backward functions, which only receive tensors,
 * can find the statistics the matching forward normalized with. The device
 * buffers belong to the BatchNorm; tensor_free() only frees this small struct. */
typedef struct BatchNormMeta {
    float *save_mean;
    float *save_invstd;
} BatchNormMeta;

/* Block-wide sum; every thread must call it, thread 0's return value is the total. */
__device__ float block_sum(float v, float *sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float total = sh[0];
    __syncthreads();
    return total;
}

/* x is (B, C, S) contiguous. One block per channel. */
__global__ void bn_stats_kernel(const float *x, float *mean, float *invstd, float *run_mean, float *run_var, int B,
                                int C, int S, float eps, float momentum) {
    __shared__ float sh[BN_THREADS];
    int c = blockIdx.x;
    int n = B * S;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += x[((size_t)(i / S) * C + c) * S + (i % S)];
    float m = block_sum(sum, sh) / (float)n;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float d = x[((size_t)(i / S) * C + c) * S + (i % S)] - m;
        sq += d * d;
    }
    float var = block_sum(sq, sh) / (float)n;
    if (threadIdx.x == 0) {
        mean[c] = m;
        invstd[c] = rsqrtf(var + eps);
        float unbiased = n > 1 ? var * (float)n / (float)(n - 1) : var;
        run_mean[c] = (1.0f - momentum) * run_mean[c] + momentum * m;
        run_var[c] = (1.0f - momentum) * run_var[c] + momentum * unbiased;
    }
}

__global__ void bn_eval_stats_kernel(const float *run_mean, const float *run_var, float *mean, float *invstd, int C,
                                     float eps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        mean[c] = run_mean[c];
        invstd[c] = rsqrtf(run_var[c] + eps);
    }
}

__global__ void bn_apply_kernel(const float *x, const float *gamma, const float *beta, const float *mean,
                                const float *invstd, float *y, int total, int C, int S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int c = (i / S) % C;
        y[i] = gamma[c] * (x[i] - mean[c]) * invstd[c] + beta[c];
    }
}

/* sum_dy[c] = sum(dy), sum_dyxhat[c] = sum(dy * xhat), xhat = (x - mean) * invstd. One block per channel. */
__global__ void bn_reduce_kernel(const float *x, const float *dy, const float *mean, const float *invstd,
                                 float *sum_dy, float *sum_dyxhat, int B, int C, int S) {
    __shared__ float sh[BN_THREADS];
    int c = blockIdx.x;
    int n = B * S;
    float a = 0.0f, b = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        size_t idx = ((size_t)(i / S) * C + c) * S + (i % S);
        a += dy[idx];
        b += dy[idx] * (x[idx] - mean[c]) * invstd[c];
    }
    float ta = block_sum(a, sh);
    float tb = block_sum(b, sh);
    if (threadIdx.x == 0) {
        sum_dy[c] = ta;
        sum_dyxhat[c] = tb;
    }
}

__global__ void bn_backward_input_kernel(const float *x, const float *dy, const float *gamma, const float *mean,
                                         const float *invstd, const float *sum_dy, const float *sum_dyxhat, float *dx,
                                         int total, int C, int S, float inv_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int c = (i / S) % C;
        float xhat = (x[i] - mean[c]) * invstd[c];
        dx[i] = gamma[c] * invstd[c] * (dy[i] - sum_dy[c] * inv_n - xhat * sum_dyxhat[c] * inv_n);
    }
}

__global__ void fill_kernel(float *p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

void bn_dims(const Tensor *x, int *B, int *C, int *S) {
    *B = x->shape[0];
    *C = x->shape[1];
    *S = tensor_numel(x->ndim, x->shape) / (*B * *C);
}

float *dev_alloc(int n) {
    float *p = (float *)tensor_pool_alloc(sizeof(float) * n);
    CUDA_CHECK(cudaMemset(p, 0, sizeof(float) * n));
    return p;
}

/* Reduces dy against the statistics gamma's forward saved; the two (C)-sized
 * results are left in *sum_dy / *sum_dyxhat (device, caller frees). */
void reduce_grad(const Tensor *x, const Tensor *gamma, const Tensor *grad_out, float **sum_dy, float **sum_dyxhat) {
    BatchNormMeta *m = (BatchNormMeta *)gamma->meta;
    int B, C, S;
    bn_dims(x, &B, &C, &S);
    *sum_dy = dev_alloc(C);
    *sum_dyxhat = dev_alloc(C);
    bn_reduce_kernel<<<C, BN_THREADS>>>(x->data, grad_out->data, m->save_mean, m->save_invstd, *sum_dy, *sum_dyxhat, B,
                                        C, S);
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

BatchNorm *batchnorm_new(int channels) {
    BatchNorm *self = (BatchNorm *)malloc(sizeof(BatchNorm));
    if (!self) return NULL;
    self->channels = channels;
    self->eps = 1e-5f;
    self->momentum = 0.1f;

    int shape[1] = {channels};
    self->gamma = tensor_create(1, shape, 1);
    self->beta = tensor_create(1, shape, 1);
    int grid = (channels + 255) / 256;
    fill_kernel<<<grid, 256>>>(self->gamma->data, 1.0f, channels);

    self->running_mean = dev_alloc(channels);
    self->running_var = dev_alloc(channels);
    fill_kernel<<<grid, 256>>>(self->running_var, 1.0f, channels);
    self->save_mean = dev_alloc(channels);
    self->save_invstd = dev_alloc(channels);
    CUDA_CHECK(cudaDeviceSynchronize());

    BatchNormMeta *meta = (BatchNormMeta *)malloc(sizeof(BatchNormMeta));
    meta->save_mean = self->save_mean;
    meta->save_invstd = self->save_invstd;
    self->gamma->meta = meta;
    return self;
}

void batchnorm_free(BatchNorm *self) {
    if (!self) return;
    tensor_free(self->gamma); /* also frees the meta struct */
    tensor_free(self->beta);
    size_t bytes = sizeof(float) * (size_t)self->channels;
    tensor_pool_free(self->running_mean, bytes);
    tensor_pool_free(self->running_var, bytes);
    tensor_pool_free(self->save_mean, bytes);
    tensor_pool_free(self->save_invstd, bytes);
    free(self);
}

Tensor *batchnorm_forward(BatchNorm *self, Tensor *x, int training) {
    if ((x->ndim != 4 && x->ndim != 2) || x->shape[1] != self->channels) {
        printf("batchnorm_forward: expected (B, %d, H, W) or (B, %d), got %d dims with %d channels\n", self->channels,
               self->channels, x->ndim, x->ndim > 1 ? x->shape[1] : 0);
        return NULL;
    }
    int B, C, S;
    bn_dims(x, &B, &C, &S);
    int total = B * C * S;
    if (training && B * S < 2) {
        printf("batchnorm_forward: training needs more than one value per channel (batch %d, %d positions)\n", B, S);
        return NULL;
    }

    Tensor *y = tensor_create(x->ndim, x->shape, x->requires_grad || self->gamma->requires_grad);
    if (training) {
        bn_stats_kernel<<<C, BN_THREADS>>>(x->data, self->save_mean, self->save_invstd, self->running_mean,
                                           self->running_var, B, C, S, self->eps, self->momentum);
    } else {
        bn_eval_stats_kernel<<<(C + 255) / 256, 256>>>(self->running_mean, self->running_var, self->save_mean,
                                                       self->save_invstd, C, self->eps);
    }
    bn_apply_kernel<<<(total + 255) / 256, 256>>>(x->data, self->gamma->data, self->beta->data, self->save_mean,
                                                  self->save_invstd, y->data, total, C, S);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (x->requires_grad) tensor_add_dependency_with_ctx(y, x, self->gamma, batchnorm_backward_input);
    if (self->gamma->requires_grad) tensor_add_dependency_with_ctx(y, self->gamma, x, batchnorm_backward_gamma);
    if (self->beta->requires_grad) tensor_add_dependency(y, self->beta, batchnorm_backward_beta);
    return y;
}

Tensor *batchnorm_backward_input(Tensor *x, Tensor *gamma, Tensor *grad_out) {
    BatchNormMeta *m = (BatchNormMeta *)gamma->meta;
    int B, C, S;
    bn_dims(x, &B, &C, &S);
    int total = B * C * S;
    float *sum_dy, *sum_dyxhat;
    reduce_grad(x, gamma, grad_out, &sum_dy, &sum_dyxhat);

    Tensor *grad_x = tensor_create(x->ndim, x->shape, 0);
    bn_backward_input_kernel<<<(total + 255) / 256, 256>>>(x->data, grad_out->data, gamma->data, m->save_mean,
                                                           m->save_invstd, sum_dy, sum_dyxhat, grad_x->data, total, C,
                                                           S, 1.0f / (float)(B * S));
    CUDA_CHECK(cudaDeviceSynchronize());
    tensor_pool_free(sum_dy, sizeof(float) * C);
    tensor_pool_free(sum_dyxhat, sizeof(float) * C);
    return grad_x;
}

Tensor *batchnorm_backward_gamma(Tensor *gamma, Tensor *x, Tensor *grad_out) {
    float *sum_dy, *sum_dyxhat;
    reduce_grad(x, gamma, grad_out, &sum_dy, &sum_dyxhat);
    int shape[1] = {gamma->shape[0]};
    Tensor *grad_g = tensor_create(1, shape, 0);
    CUDA_CHECK(cudaMemcpy(grad_g->data, sum_dyxhat, sizeof(float) * shape[0], cudaMemcpyDeviceToDevice));
    tensor_pool_free(sum_dy, sizeof(float) * shape[0]);
    tensor_pool_free(sum_dyxhat, sizeof(float) * shape[0]);
    return grad_g;
}

Tensor *batchnorm_backward_beta(Tensor *beta, Tensor *ctx, Tensor *grad_out) {
    (void)ctx;
    int B = grad_out->shape[0], C = grad_out->shape[1];
    int S = tensor_numel(grad_out->ndim, grad_out->shape) / (B * C);
    /* sum(dy) per channel needs no statistics; reuse the reduction kernel with dummy mean/invstd */
    float *zero = dev_alloc(C), *one = dev_alloc(C);
    fill_kernel<<<(C + 255) / 256, 256>>>(one, 1.0f, C);
    float *sum_dy = dev_alloc(C), *sum_dyxhat = dev_alloc(C);
    bn_reduce_kernel<<<C, BN_THREADS>>>(grad_out->data, grad_out->data, zero, one, sum_dy, sum_dyxhat, B, C, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    int shape[1] = {beta->shape[0]};
    Tensor *grad_b = tensor_create(1, shape, 0);
    CUDA_CHECK(cudaMemcpy(grad_b->data, sum_dy, sizeof(float) * shape[0], cudaMemcpyDeviceToDevice));
    tensor_pool_free(zero, sizeof(float) * C);
    tensor_pool_free(one, sizeof(float) * C);
    tensor_pool_free(sum_dy, sizeof(float) * C);
    tensor_pool_free(sum_dyxhat, sizeof(float) * C);
    return grad_b;
}

#ifdef __cplusplus
}
#endif
