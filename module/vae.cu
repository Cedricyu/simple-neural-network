#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.cuh"
#include "vae.cuh"

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

#define LOGVAR_MAX 10.0f

/* Attached to the output's meta; the output tensor is also the backward context. eps is not owned. */
typedef struct ReparamMeta {
    float beta;
    Tensor *eps;
} ReparamMeta;

__global__ void reparam_forward_kernel(const float *x, const float *eps, float *z, int B, int Z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * Z) return;
    int b = i / Z, k = i % Z;
    float mu = x[(size_t)b * 2 * Z + k];
    float lv = fminf(fmaxf(x[(size_t)b * 2 * Z + Z + k], -LOGVAR_MAX), LOGVAR_MAX);
    z[i] = mu + expf(0.5f * lv) * eps[i];
}

__global__ void reparam_backward_kernel(const float *x, const float *eps, const float *dz, float *dx, int B, int Z,
                                        float beta, float inv_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * Z) return;
    int b = i / Z, k = i % Z;
    float mu = x[(size_t)b * 2 * Z + k];
    float lv_raw = x[(size_t)b * 2 * Z + Z + k];
    float lv = fminf(fmaxf(lv_raw, -LOGVAR_MAX), LOGVAR_MAX);
    float inside = (lv_raw >= -LOGVAR_MAX && lv_raw <= LOGVAR_MAX) ? 1.0f : 0.0f;
    float sd = expf(0.5f * lv);
    dx[(size_t)b * 2 * Z + k] = dz[i] + beta * mu * inv_b;
    dx[(size_t)b * 2 * Z + Z + k] = inside * (dz[i] * 0.5f * eps[i] * sd + beta * 0.5f * (sd * sd - 1.0f) * inv_b);
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

Tensor *reparam_forward(Tensor *x, Tensor *eps, float beta) {
    if (x->ndim != 2 || x->shape[1] % 2 != 0 || eps->ndim != 2 || eps->shape[0] != x->shape[0] ||
        eps->shape[1] != x->shape[1] / 2) {
        printf("reparam_forward: x must be (B, 2*Z) and eps (B, Z).\n");
        return NULL;
    }
    int B = x->shape[0], Z = x->shape[1] / 2;
    int shape[2] = {B, Z};
    Tensor *z = tensor_create(2, shape, x->requires_grad);
    reparam_forward_kernel<<<(B * Z + 255) / 256, 256>>>(x->data, eps->data, z->data, B, Z);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (x->requires_grad) {
        ReparamMeta *meta = (ReparamMeta *)malloc(sizeof(ReparamMeta));
        meta->beta = beta;
        meta->eps = eps;
        z->meta = meta;
        tensor_add_dependency_with_ctx(z, x, z, reparam_backward);
    }
    return z;
}

Tensor *reparam_backward(Tensor *x, Tensor *out, Tensor *grad_out) {
    ReparamMeta *meta = (ReparamMeta *)out->meta;
    int B = x->shape[0], Z = x->shape[1] / 2;
    Tensor *grad_x = tensor_create(2, x->shape, 0);
    reparam_backward_kernel<<<(B * Z + 255) / 256, 256>>>(x->data, meta->eps->data, grad_out->data, grad_x->data, B, Z,
                                                          meta->beta, 1.0f / (float)B);
    CUDA_CHECK(cudaDeviceSynchronize());
    return grad_x;
}

#ifdef __cplusplus
}
#endif
