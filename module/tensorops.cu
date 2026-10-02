#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "activationkernels.cuh"
#include "cudakernels.cuh"
#include "tensorops.cuh"

namespace {
float *g_workspace_bt = NULL;
size_t g_workspace_bt_elems = 0;
float *g_workspace_at = NULL;
size_t g_workspace_at_elems = 0;

float *ensure_workspace(float **buf, size_t *cap, size_t need_elems) {
    if (*cap < need_elems) {
        if (*buf) cudaFree(*buf);
        cudaMalloc((void **)buf, need_elems * sizeof(float));
        *cap = need_elems;
    }
    return *buf;
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (a->ndim != 2 || b->ndim != 2) {
        printf("MatMul only supports 2D tensors.\n");
        return NULL;
    }

    int M = a->shape[0];
    int K = a->shape[1];
    int Kb = b->shape[0];
    int N = b->shape[1];
    if (K != Kb) {
        printf("Shape mismatch in tensor_matmul: (%d x %d) x (%d x %d)\n", M, K, Kb, N);
        return NULL;
    }

    int out_shape[2] = {M, N};
    Tensor *out = tensor_create(2, out_shape, a->requires_grad || b->requires_grad);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel<<<grid, block>>>(a->data, b->data, out->data, M, K, N);
    cudaDeviceSynchronize();

    if (a->requires_grad || b->requires_grad) {
        tensor_add_dependency_with_ctx(out, a, b, tensor_matmul_backward_a);
        tensor_add_dependency_with_ctx(out, b, a, tensor_matmul_backward_b);
    }
    return out;
}

Tensor *tensor_matmul_backward_a(Tensor *a, Tensor *b, Tensor *grad_out) {
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    int shape[2] = {M, K};
    Tensor *grad_a = tensor_create(2, shape, 0);

    float *d_b_t = ensure_workspace(&g_workspace_bt, &g_workspace_bt_elems, (size_t)N * (size_t)K);

    /* matrixTransposeKernel has no shared-memory tiling, so any block size
     * works for it; 16x16 is just a reasonable default. It maps thread x to
     * the column index (range [0,cols)) and y to the row index (range
     * [0,rows)) - so grid.x must be sized off cols and grid.y off rows. b
     * here has shape (K,N): rows=K, cols=N. This grid was backwards
     * ((K+15)/16, (N+15)/16), which silently worked only when K<=16 (every
     * existing caller's dimensions happened to stay under that); any larger
     * K left rows beyond 16 never written by the transpose, so the matmul
     * below read uninitialized workspace memory for them. */
    dim3 t_block(16, 16);
    dim3 grid_t((N + 15) / 16, (K + 15) / 16);
    matrixTransposeKernel<<<grid_t, t_block>>>(b->data, d_b_t, K, N);
    cudaDeviceSynchronize();

    /* Unlike the transpose, matrixMultiplyKernel DOES use shared-memory
     * tiles sized BLOCK_SIZE x BLOCK_SIZE (cudakernels.cuh, currently 32) -
     * launching it with a block smaller than that (this used to hardcode
     * 16x16) leaves the rest of each tile never written by any thread, so
     * the reduction sums in whatever was already sitting in that shared
     * memory. Both bugs were invisible together: at the small dimensions
     * every existing caller used, the stale tail of each tile happened to
     * still be zero this session, so the extra terms added nothing - a
     * numerical gradient check only caught it once a bigger Linear layer
     * (1568->128) pushed past that coincidence. */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_mm((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel<<<grid_mm, block>>>(grad_out->data, d_b_t, grad_a->data, M, N, K);
    cudaDeviceSynchronize();
    return grad_a;
}

Tensor *tensor_matmul_backward_b(Tensor *a, Tensor *b, Tensor *grad_out) {
    // Here: a = weight tensor (target), b = input tensor (context)
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    int K = a->shape[0];
    int N = a->shape[1];
    int M = b->shape[0];

    int shape[2] = {K, N};
    Tensor *grad_b = tensor_create(2, shape, 0);

    float *d_a_t = ensure_workspace(&g_workspace_at, &g_workspace_at_elems, (size_t)K * (size_t)M);

    dim3 t_block(16, 16);
    dim3 grid_t((K + 15) / 16, (M + 15) / 16);
    matrixTransposeKernel<<<grid_t, t_block>>>(b->data, d_a_t, M, K);
    cudaDeviceSynchronize();

    /* Must match BLOCK_SIZE, not an arbitrary block size - see the comment
     * in tensor_matmul_backward_a above. */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_mm((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel<<<grid_mm, block>>>(d_a_t, grad_out->data, grad_b->data, K, M, N);
    cudaDeviceSynchronize();
    return grad_b;
}

Tensor *tensor_add_bias(Tensor *x, Tensor *bias) {
    if (x->ndim != 2 || bias->ndim != 2 || bias->shape[0] != 1 || x->shape[1] != bias->shape[1]) {
        printf("Shape mismatch in tensor_add_bias.\n");
        return NULL;
    }

    int M = x->shape[0];
    int N = x->shape[1];
    int out_shape[2] = {M, N};
    Tensor *out = tensor_create(2, out_shape, x->requires_grad || bias->requires_grad);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matrixCopyKernel<<<grid, block>>>(x->data, out->data, M, N);
    addBiasKernel<<<grid, block>>>(out->data, bias->data, M, N);
    cudaDeviceSynchronize();

    if (x->requires_grad || bias->requires_grad) {
        tensor_add_dependency(out, x, tensor_add_bias_backward_input);
        tensor_add_dependency(out, bias, tensor_add_bias_backward_bias);
    }
    return out;
}

Tensor *tensor_add_bias_backward_input(Tensor *x, Tensor *bias, Tensor *grad_out) {
    (void)x;
    (void)bias;
    int shape[2] = {grad_out->shape[0], grad_out->shape[1]};
    Tensor *grad_x = tensor_create(2, shape, 0);

    int M = grad_out->shape[0];
    int N = grad_out->shape[1];
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matrixCopyKernel<<<grid, block>>>(grad_out->data, grad_x->data, M, N);
    cudaDeviceSynchronize();
    return grad_x;
}

Tensor *tensor_add_bias_backward_bias(Tensor *x, Tensor *bias, Tensor *grad_out) {
    // In generic autograd: x is the bias tensor, bias is optional context tensor.
    (void)bias;
    if (x->ndim != 2 || x->shape[0] != 1 || grad_out->ndim != 2 || grad_out->shape[1] != x->shape[1]) {
        return NULL;
    }
    int N = x->shape[1];
    int shape[2] = {1, N};
    Tensor *grad_b = tensor_create(2, shape, 0);

    int block = 256;
    int grid = (N + block - 1) / block;
    biasGradientKernel<<<grid, block>>>(grad_out->data, grad_b->data, grad_out->shape[0], N);
    cudaDeviceSynchronize();
    return grad_b;
}

Tensor *tensor_relu(Tensor *x) {
    int size = tensor_numel(x->ndim, x->shape);
    Tensor *out = tensor_create(x->ndim, x->shape, x->requires_grad);

    int block = 256;
    int grid = (size + block - 1) / block;
    reluKernel<<<grid, block>>>(x->data, out->data, size);
    cudaDeviceSynchronize();

    if (x->requires_grad) tensor_add_dependency(out, x, tensor_relu_backward);
    return out;
}

Tensor *tensor_relu_backward(Tensor *x, Tensor *n, Tensor *grad_out) {
    (void)n;
    int size = tensor_numel(x->ndim, x->shape);
    Tensor *grad_x = tensor_create(x->ndim, x->shape, 0);

    int block = 256;
    int grid = (size + block - 1) / block;
    reluBackwardKernel<<<grid, block>>>(x->data, grad_out->data, grad_x->data, size);
    cudaDeviceSynchronize();
    return grad_x;
}

/* Collapses any (dim0, dim1, ..., dimN) tensor to 2D (dim0, dim1*...*dimN) -
 * e.g. a conv output (B, C, H, W) into the (B, C*H*W) a Linear layer expects.
 * Row-major layout makes this a pure reshape: same bytes, same order, so
 * forward is a zero-copy view onto x's own device buffer (no kernel, no
 * allocation) and backward is a single flat memcpy back to x's original
 * shape - unlike every other op in this file, there's no elementwise or
 * reduction math to do, just metadata. */
Tensor *tensor_flatten(Tensor *x) {
    int batch = x->shape[0];
    int rest = tensor_numel(x->ndim, x->shape) / batch;
    int shape[2] = {batch, rest};
    Tensor *out = tensor_from_device(x->data, 2, shape, x->requires_grad);
    if (x->requires_grad) tensor_add_dependency(out, x, tensor_flatten_backward);
    return out;
}

Tensor *tensor_flatten_backward(Tensor *x, Tensor *ctx, Tensor *grad_out) {
    (void)ctx;
    int n = tensor_numel(x->ndim, x->shape);
    Tensor *grad_x = tensor_create(x->ndim, x->shape, 0);
    cudaMemcpy(grad_x->data, grad_out->data, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    return grad_x;
}

static void tensor_print_graph_dot_rec(Tensor *self) {
    if (!self) return;
    printf("  \"%p\" [label=\"Tensor %p\"];\n", self, self);
    for (int i = 0; i < self->num_deps; ++i) {
        Dependency *dep = &self->deps[i];
        if (dep && dep->tensor) {
            printf("  \"%p\" -> \"%p\";\n", self, dep->tensor);
            tensor_print_graph_dot_rec(dep->tensor);
        }
    }
}

void tensor_print_graph_dot(Tensor *self) {
    printf("digraph G {\n");
    tensor_print_graph_dot_rec(self);
    printf("}\n");
}

void fill_tensor_with_random(Tensor *t) {
    int n = tensor_numel(t->ndim, t->shape);
    float *host = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) {
        float r = (float)rand() / (float)RAND_MAX;
        host[i] = r * 2.0f - 1.0f;
    }
    cudaMemcpy(t->data, host, sizeof(float) * n, cudaMemcpyHostToDevice);
    free(host);
}

Tensor *tensor_clone(Tensor *t) {
    Tensor *clone = tensor_create(t->ndim, t->shape, t->requires_grad);
    int n = tensor_numel(t->ndim, t->shape);
    cudaMemcpy(clone->data, t->data, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    if (t->requires_grad && t->grad) {
        cudaMemcpy(clone->grad, t->grad, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    }
    return clone;
}

#ifdef __cplusplus
}
#endif
