#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.cuh"
#include "tensorops.cuh"
#include "transformer.cuh"

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t _err = (call);                                                                \
        if (_err != cudaSuccess) {                                                                \
            fprintf(stderr, "[cuda] %s failed at %s:%d: %s\n", #call, __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                                    \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

#define TF_THREADS 128

namespace {

/* Block-wide sum / max; every thread must call, all threads get the result. sh has blockDim.x floats. */
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

__device__ float block_max(float v, float *sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    float m = sh[0];
    __syncthreads();
    return m;
}

float *dev_alloc(size_t n) { return (float *)tensor_pool_alloc(sizeof(float) * n); }
void dev_free(float *p, size_t n) { tensor_pool_free(p, sizeof(float) * n); }

void ln_dims(const Tensor *x, int *B, int *C, int *S) {
    *B = x->shape[0];
    *C = x->shape[1];
    *S = tensor_numel(x->ndim, x->shape) / (*B * *C);
}

/* ---- layer norm: x is (B, C, S); a "row" is one (b, s) pair, whose C values are S apart ---- */

__global__ void ln_stats_kernel(const float *x, float *mean, float *rstd, int C, int S, float eps) {
    __shared__ float sh[TF_THREADS];
    int row = blockIdx.x, b = row / S, s = row % S;
    const float *p = x + (size_t)b * C * S + s;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) sum += p[(size_t)c * S];
    float m = block_sum(sum, sh) / (float)C;
    float sq = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = p[(size_t)c * S] - m;
        sq += d * d;
    }
    float var = block_sum(sq, sh) / (float)C;
    if (threadIdx.x == 0) {
        mean[row] = m;
        rstd[row] = rsqrtf(var + eps);
    }
}

__global__ void ln_apply_kernel(const float *x, const float *gamma, const float *beta, const float *mean,
                                const float *rstd, float *y, int total, int C, int S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (i / S) % C, row = (i / (C * S)) * S + i % S;
    y[i] = gamma[c] * (x[i] - mean[row]) * rstd[row] + beta[c];
}

/* dx = rstd * (g*dy - mean_c(g*dy) - xhat * mean_c(g*dy*xhat)), one block per row */
__global__ void ln_backward_input_kernel(const float *x, const float *dy, const float *gamma, const float *mean,
                                         const float *rstd, float *dx, int C, int S) {
    __shared__ float sh[TF_THREADS];
    int row = blockIdx.x, b = row / S, s = row % S;
    size_t base = (size_t)b * C * S + s;
    float m = mean[row], r = rstd[row];
    float a = 0.0f, bsum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float g = gamma[c] * dy[base + (size_t)c * S];
        a += g;
        bsum += g * (x[base + (size_t)c * S] - m) * r;
    }
    float sum_g = block_sum(a, sh) / (float)C;
    float sum_gx = block_sum(bsum, sh) / (float)C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float xhat = (x[base + (size_t)c * S] - m) * r;
        dx[base + (size_t)c * S] = r * (gamma[c] * dy[base + (size_t)c * S] - sum_g - xhat * sum_gx);
    }
}

/* dgamma[c] = sum dy * xhat, dbeta[c] = sum dy over every row; one block per channel */
__global__ void ln_param_grad_kernel(const float *x, const float *dy, const float *mean, const float *rstd, float *dgamma,
                                     float *dbeta, int B, int C, int S) {
    __shared__ float sh[TF_THREADS];
    int c = blockIdx.x;
    float a = 0.0f, bsum = 0.0f;
    for (int i = threadIdx.x; i < B * S; i += blockDim.x) {
        int b = i / S, s = i % S;
        size_t idx = ((size_t)b * C + c) * S + s;
        a += dy[idx] * (x[idx] - mean[i]) * rstd[i];
        bsum += dy[idx];
    }
    float ta = block_sum(a, sh), tb = block_sum(bsum, sh);
    if (threadIdx.x == 0) {
        dgamma[c] = ta;
        dbeta[c] = tb;
    }
}

__global__ void fill_kernel(float *p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

/* ---- gelu ---- */

__global__ void gelu_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118f));
}

__global__ void gelu_backward_kernel(const float *x, const float *dy, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float cdf = 0.5f * (1.0f + erff(v * 0.70710678118f));
    float pdf = 0.39894228040f * expf(-0.5f * v * v);
    dx[i] = dy[i] * (cdf + v * pdf);
}

/* ---- position embedding ---- */

__global__ void pos_add_kernel(const float *x, const float *pos, float *y, int total, int per_sample) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] = x[i] + pos[i % per_sample];
}

__global__ void pos_grad_kernel(const float *dy, float *dpos, int B, int per_sample) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= per_sample) return;
    float s = 0.0f;
    for (int b = 0; b < B; ++b) s += dy[(size_t)b * per_sample + i];
    dpos[i] = s;
}

/* ---- attention. qkv is (B, 3D, T); head h of batch b reads channels h*dh.. of the q, k and v thirds ---- */

typedef struct AttnMeta {
    int heads;
} AttnMeta;

/* P[(b*heads + h), i, j] = softmax_j(q_i . k_j / sqrt(dh)); one block per (b, h, i), T floats of shared memory */
__global__ void attn_probs_kernel(const float *qkv, float *P, int D, int T, int heads, float scale) {
    extern __shared__ float row[];
    __shared__ float sh[TF_THREADS];
    int bh = blockIdx.x / T, i = blockIdx.x % T, b = bh / heads, h = bh % heads, dh = D / heads;
    const float *q = qkv + ((size_t)b * 3 * D + (size_t)h * dh) * T;
    const float *k = qkv + ((size_t)b * 3 * D + D + (size_t)h * dh) * T;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < dh; ++d) s += q[(size_t)d * T + i] * k[(size_t)d * T + j];
        row[j] = s * scale;
        mx = fmaxf(mx, row[j]);
    }
    mx = block_max(mx, sh);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    sum = block_sum(sum, sh);
    float *out = P + ((size_t)bh * T + i) * T;
    for (int j = threadIdx.x; j < T; j += blockDim.x) out[j] = row[j] / sum;
}

/* out[b, ch, i] = sum_j P[b, h(ch), i, j] * v[ch, j] */
__global__ void attn_out_kernel(const float *qkv, const float *P, float *out, int B, int D, int T, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D * T) return;
    int i = idx % T, ch = (idx / T) % D, b = idx / (T * D), h = ch / (D / heads);
    const float *p = P + (((size_t)b * heads + h) * T + i) * T;
    const float *v = qkv + ((size_t)b * 3 * D + 2 * D + ch) * T;
    float s = 0.0f;
    for (int j = 0; j < T; ++j) s += p[j] * v[j];
    out[idx] = s;
}

/* dS[bh, i, j] = scale * P_ij * (dP_ij - sum_j' P_ij' dP_ij'), dP_ij = dO_i . v_j; one block per (b, h, i) */
__global__ void attn_dscore_kernel(const float *qkv, const float *P, const float *dO, float *dS, int D, int T, int heads,
                                   float scale) {
    extern __shared__ float row[];
    __shared__ float sh[TF_THREADS];
    int bh = blockIdx.x / T, i = blockIdx.x % T, b = bh / heads, h = bh % heads, dh = D / heads;
    const float *v = qkv + ((size_t)b * 3 * D + 2 * D + (size_t)h * dh) * T;
    const float *go = dO + ((size_t)b * D + (size_t)h * dh) * T;
    const float *p = P + ((size_t)bh * T + i) * T;
    float dot = 0.0f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float dp = 0.0f;
        for (int d = 0; d < dh; ++d) dp += go[(size_t)d * T + i] * v[(size_t)d * T + j];
        row[j] = dp;
        dot += p[j] * dp;
    }
    dot = block_sum(dot, sh);
    float *out = dS + ((size_t)bh * T + i) * T;
    for (int j = threadIdx.x; j < T; j += blockDim.x) out[j] = scale * p[j] * (row[j] - dot);
}

/* grad_q[ch, i] = sum_j dS[i, j] k[ch, j];  grad_k[ch, j] = sum_i dS[i, j] q[ch, i];  grad_v[ch, j] = sum_i P[i, j] dO[ch, i] */
__global__ void attn_dqkv_kernel(const float *qkv, const float *P, const float *dS, const float *dO, float *grad, int B,
                                 int D, int T, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 3 * B * D * T) return;
    int t = idx % T, ch = (idx / T) % D, rest = idx / (T * D), which = rest % 3, b = rest / 3, h = ch / (D / heads);
    size_t bh = (size_t)b * heads + h;
    const float *q = qkv + ((size_t)b * 3 * D + ch) * T;
    const float *k = qkv + ((size_t)b * 3 * D + D + ch) * T;
    const float *go = dO + ((size_t)b * D + ch) * T;
    float s = 0.0f;
    if (which == 0) {
        for (int j = 0; j < T; ++j) s += dS[(bh * T + t) * T + j] * k[j];
    } else if (which == 1) {
        for (int i = 0; i < T; ++i) s += dS[(bh * T + i) * T + t] * q[i];
    } else {
        for (int i = 0; i < T; ++i) s += P[(bh * T + i) * T + t] * go[i];
    }
    grad[((size_t)b * 3 * D + which * D + ch) * T + t] = s;
}

int attn_dims(const Tensor *qkv, int heads, int *B, int *D, int *T) {
    if (qkv->ndim != 4 || heads < 1 || qkv->shape[1] % (3 * heads) != 0) return 0;
    *B = qkv->shape[0];
    *D = qkv->shape[1] / 3;
    *T = qkv->shape[2] * qkv->shape[3];
    return 1;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

/* ---- layer norm ---- */

LayerNorm *layernorm_new(int channels) {
    LayerNorm *self = (LayerNorm *)malloc(sizeof(LayerNorm));
    if (!self) return NULL;
    self->channels = channels;
    self->eps = 1e-5f;
    int shape[1] = {channels};
    self->gamma = tensor_create(1, shape, 1);
    self->beta = tensor_create(1, shape, 1);
    fill_kernel<<<(channels + 255) / 256, 256>>>(self->gamma->data, 1.0f, channels);
    CUDA_CHECK(cudaMemset(self->beta->data, 0, sizeof(float) * channels));
    CUDA_CHECK(cudaDeviceSynchronize());
    return self;
}

void layernorm_free(LayerNorm *self) {
    if (!self) return;
    tensor_free(self->gamma);
    tensor_free(self->beta);
    free(self);
}

Tensor *layernorm_forward(LayerNorm *self, Tensor *x) {
    if ((x->ndim != 4 && x->ndim != 2) || x->shape[1] != self->channels) {
        printf("layernorm_forward: expected (B, %d, H, W) or (B, %d), got %d dims\n", self->channels, self->channels, x->ndim);
        return NULL;
    }
    int B, C, S;
    ln_dims(x, &B, &C, &S);
    float *mean = dev_alloc((size_t)B * S), *rstd = dev_alloc((size_t)B * S);
    Tensor *y = tensor_create(x->ndim, x->shape, x->requires_grad || self->gamma->requires_grad);
    ln_stats_kernel<<<B * S, TF_THREADS>>>(x->data, mean, rstd, C, S, self->eps);
    int total = B * C * S;
    ln_apply_kernel<<<(total + 255) / 256, 256>>>(x->data, self->gamma->data, self->beta->data, mean, rstd, y->data, total, C, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_free(mean, (size_t)B * S);
    dev_free(rstd, (size_t)B * S);
    if (x->requires_grad) tensor_add_dependency_with_ctx(y, x, self->gamma, layernorm_backward_input);
    if (self->gamma->requires_grad) tensor_add_dependency_with_ctx(y, self->gamma, x, layernorm_backward_gamma);
    if (self->beta->requires_grad) tensor_add_dependency_with_ctx(y, self->beta, x, layernorm_backward_beta);
    return y;
}

/* The statistics are recomputed from x in each backward function, so a forward leaves no state behind. */
Tensor *layernorm_backward_input(Tensor *x, Tensor *gamma, Tensor *grad_out) {
    int B, C, S;
    ln_dims(x, &B, &C, &S);
    float *mean = dev_alloc((size_t)B * S), *rstd = dev_alloc((size_t)B * S);
    ln_stats_kernel<<<B * S, TF_THREADS>>>(x->data, mean, rstd, C, S, 1e-5f);
    Tensor *dx = tensor_create(x->ndim, x->shape, 0);
    ln_backward_input_kernel<<<B * S, TF_THREADS>>>(x->data, grad_out->data, gamma->data, mean, rstd, dx->data, C, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_free(mean, (size_t)B * S);
    dev_free(rstd, (size_t)B * S);
    return dx;
}

static Tensor *ln_param_grad(Tensor *param, Tensor *x, Tensor *grad_out, int want_gamma) {
    int B, C, S;
    ln_dims(x, &B, &C, &S);
    float *mean = dev_alloc((size_t)B * S), *rstd = dev_alloc((size_t)B * S);
    float *dg = dev_alloc(C), *db = dev_alloc(C);
    ln_stats_kernel<<<B * S, TF_THREADS>>>(x->data, mean, rstd, C, S, 1e-5f);
    ln_param_grad_kernel<<<C, TF_THREADS>>>(x->data, grad_out->data, mean, rstd, dg, db, B, C, S);
    CUDA_CHECK(cudaDeviceSynchronize());
    int shape[1] = {param->shape[0]};
    Tensor *g = tensor_create(1, shape, 0);
    CUDA_CHECK(cudaMemcpy(g->data, want_gamma ? dg : db, sizeof(float) * C, cudaMemcpyDeviceToDevice));
    dev_free(mean, (size_t)B * S);
    dev_free(rstd, (size_t)B * S);
    dev_free(dg, C);
    dev_free(db, C);
    return g;
}

Tensor *layernorm_backward_gamma(Tensor *gamma, Tensor *x, Tensor *grad_out) { return ln_param_grad(gamma, x, grad_out, 1); }
Tensor *layernorm_backward_beta(Tensor *beta, Tensor *x, Tensor *grad_out) { return ln_param_grad(beta, x, grad_out, 0); }

/* ---- gelu ---- */

Tensor *tensor_gelu(Tensor *x) {
    int n = tensor_numel(x->ndim, x->shape);
    Tensor *y = tensor_create(x->ndim, x->shape, x->requires_grad);
    gelu_kernel<<<(n + 255) / 256, 256>>>(x->data, y->data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (x->requires_grad) tensor_add_dependency(y, x, tensor_gelu_backward);
    return y;
}

Tensor *tensor_gelu_backward(Tensor *x, Tensor *ctx, Tensor *grad_out) {
    (void)ctx;
    int n = tensor_numel(x->ndim, x->shape);
    Tensor *dx = tensor_create(x->ndim, x->shape, 0);
    gelu_backward_kernel<<<(n + 255) / 256, 256>>>(x->data, grad_out->data, dx->data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    return dx;
}

/* ---- position embedding ---- */

Tensor *posembed_new(int c, int h, int w) {
    int shape[3] = {c, h, w};
    Tensor *pos = tensor_create(3, shape, 1);
    int n = c * h * w;
    float *host = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) {
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0), u2 = (double)rand() / ((double)RAND_MAX + 1.0);
        host[i] = 0.02f * (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
    }
    tensor_set_data(pos, host);
    free(host);
    return pos;
}

Tensor *posembed_forward(Tensor *pos, Tensor *x) {
    int per = tensor_numel(x->ndim, x->shape) / x->shape[0];
    if (x->ndim != 4 || tensor_numel(pos->ndim, pos->shape) != per || pos->shape[0] != x->shape[1] ||
        pos->shape[1] != x->shape[2] || pos->shape[2] != x->shape[3]) {
        printf("posembed_forward: the embedding is %dx%dx%d but the input is not\n", pos->shape[0], pos->shape[1], pos->shape[2]);
        return NULL;
    }
    int n = tensor_numel(x->ndim, x->shape);
    Tensor *y = tensor_create(x->ndim, x->shape, x->requires_grad || pos->requires_grad);
    pos_add_kernel<<<(n + 255) / 256, 256>>>(x->data, pos->data, y->data, n, per);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (x->requires_grad) tensor_add_dependency(y, x, tensor_add_backward);
    if (pos->requires_grad) tensor_add_dependency_with_ctx(y, pos, x, posembed_backward_pos);
    return y;
}

Tensor *posembed_backward_pos(Tensor *pos, Tensor *x, Tensor *grad_out) {
    int per = tensor_numel(pos->ndim, pos->shape);
    Tensor *g = tensor_create(pos->ndim, pos->shape, 0);
    pos_grad_kernel<<<(per + 255) / 256, 256>>>(grad_out->data, g->data, x->shape[0], per);
    CUDA_CHECK(cudaDeviceSynchronize());
    return g;
}

/* ---- attention ---- */

Tensor *attention_forward(Tensor *qkv, int heads) {
    int B, D, T;
    if (!attn_dims(qkv, heads, &B, &D, &T)) {
        printf("attention_forward: %d channels cannot be split into q, k, v of %d heads\n", qkv->ndim == 4 ? qkv->shape[1] : 0, heads);
        return NULL;
    }
    int dh = D / heads;
    size_t pn = (size_t)B * heads * T * T;
    float *P = dev_alloc(pn);
    int shape[4] = {B, D, qkv->shape[2], qkv->shape[3]};
    Tensor *out = tensor_create(4, shape, qkv->requires_grad);
    attn_probs_kernel<<<B * heads * T, TF_THREADS, sizeof(float) * T>>>(qkv->data, P, D, T, heads, 1.0f / sqrtf((float)dh));
    attn_out_kernel<<<(B * D * T + 255) / 256, 256>>>(qkv->data, P, out->data, B, D, T, heads);
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_free(P, pn);
    if (qkv->requires_grad) {
        AttnMeta *meta = (AttnMeta *)malloc(sizeof(AttnMeta));
        meta->heads = heads;
        out->meta = meta;
        tensor_add_dependency_with_ctx(out, qkv, out, attention_backward);
    }
    return out;
}

/* The attention probabilities are recomputed here rather than kept from the forward pass. */
Tensor *attention_backward(Tensor *qkv, Tensor *out, Tensor *grad_out) {
    int heads = ((AttnMeta *)out->meta)->heads, B, D, T;
    attn_dims(qkv, heads, &B, &D, &T);
    int dh = D / heads;
    float scale = 1.0f / sqrtf((float)dh);
    size_t pn = (size_t)B * heads * T * T;
    float *P = dev_alloc(pn), *dS = dev_alloc(pn);
    Tensor *grad = tensor_create(4, qkv->shape, 0);
    attn_probs_kernel<<<B * heads * T, TF_THREADS, sizeof(float) * T>>>(qkv->data, P, D, T, heads, scale);
    attn_dscore_kernel<<<B * heads * T, TF_THREADS, sizeof(float) * T>>>(qkv->data, P, grad_out->data, dS, D, T, heads, scale);
    int total = 3 * B * D * T;
    attn_dqkv_kernel<<<(total + 255) / 256, 256>>>(qkv->data, P, dS, grad_out->data, grad->data, B, D, T, heads);
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_free(P, pn);
    dev_free(dS, pn);
    return grad;
}

#ifdef __cplusplus
}
#endif
