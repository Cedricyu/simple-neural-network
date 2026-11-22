#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.cuh"

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t _err = (call);                                                                \
        if (_err != cudaSuccess) {                                                                \
            fprintf(stderr, "[cuda] %s failed at %s:%d: %s\n", #call, __FILE__, __LINE__,        \
                    cudaGetErrorString(_err));                                                    \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

/* -- device memory pool ---------------------------------------------------- */

#define POOL_MAX_ENTRIES 4096
typedef struct PoolEntry {
    void *ptr;
    size_t bytes;
} PoolEntry;
static PoolEntry g_pool[POOL_MAX_ENTRIES];
static int g_pool_n = 0;

void *tensor_pool_alloc(size_t bytes) {
    for (int i = g_pool_n - 1; i >= 0; --i) {
        if (g_pool[i].bytes == bytes) {
            void *p = g_pool[i].ptr;
            g_pool[i] = g_pool[--g_pool_n];
            return p;
        }
    }
    void *p = NULL;
    cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
        /* out of memory: hand back everything cached and try once more before giving up */
        cudaGetLastError();
        tensor_pool_release();
        CUDA_CHECK(cudaMalloc(&p, bytes));
    }
    return p;
}

void tensor_pool_free(void *ptr, size_t bytes) {
    if (!ptr) return;
    if (g_pool_n < POOL_MAX_ENTRIES) {
        g_pool[g_pool_n].ptr = ptr;
        g_pool[g_pool_n].bytes = bytes;
        g_pool_n++;
    } else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void tensor_pool_release(void) {
    for (int i = 0; i < g_pool_n; ++i) CUDA_CHECK(cudaFree(g_pool[i].ptr));
    g_pool_n = 0;
}

__global__ void tensor_set_zero_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

__global__ void tensor_add_inplace_kernel(float *dst, const float *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void tensor_momentum_step_kernel(float *w, const float *g, float *v, float lr, float mu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float vi = mu * v[i] + g[i];
        v[i] = vi;
        w[i] -= lr * vi;
    }
}

__global__ void tensor_sgd_step_kernel(float *w, const float *g, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * g[i];
}

int tensor_numel(int ndim, int *shape) {
    int total = 1;
    for (int i = 0; i < ndim; ++i) total *= shape[i];
    return total;
}

Tensor *tensor_create(int ndim, int *shape, int requires_grad) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->requires_grad = requires_grad;
    t->owns_data = 1;
    t->owns_grad = requires_grad ? 1 : 0;
    t->deps = NULL;
    t->num_deps = 0;
    t->meta = NULL;
    t->velocity = NULL;

    t->shape = (int *)malloc(sizeof(int) * ndim);
    memcpy(t->shape, shape, sizeof(int) * ndim);

    int n = tensor_numel(ndim, shape);
    t->data = (float_t *)tensor_pool_alloc(n * sizeof(float_t));
    CUDA_CHECK(cudaMemset(t->data, 0, n * sizeof(float_t)));

    if (requires_grad) {
        t->grad = (float_t *)tensor_pool_alloc(n * sizeof(float_t));
        CUDA_CHECK(cudaMemset(t->grad, 0, n * sizeof(float_t)));
    } else {
        t->grad = NULL;
    }

    return t;
}

Tensor *tensor_from_data(float *host_data, int ndim, int *shape) {
    Tensor *t = tensor_create(ndim, shape, 0);
    if (!t) return NULL;
    int n = tensor_numel(ndim, shape);
    CUDA_CHECK(cudaMemcpy(t->data, host_data, n * sizeof(float_t), cudaMemcpyHostToDevice));
    return t;
}

Tensor *tensor_from_data_2d(float *host_data, int dim0, int dim1) {
    int shape[2] = {dim0, dim1};
    return tensor_from_data(host_data, 2, shape);
}

Tensor *tensor_from_device(float *device_data, int ndim, int *shape, int requires_grad) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->requires_grad = requires_grad;
    t->owns_data = 0;
    t->owns_grad = requires_grad ? 1 : 0;
    t->deps = NULL;
    t->num_deps = 0;
    t->meta = NULL;
    t->velocity = NULL;

    t->shape = (int *)malloc(sizeof(int) * ndim);
    memcpy(t->shape, shape, sizeof(int) * ndim);
    t->data = device_data;

    if (requires_grad) {
        int n = tensor_numel(ndim, shape);
        t->grad = (float_t *)tensor_pool_alloc(n * sizeof(float_t));
        CUDA_CHECK(cudaMemset(t->grad, 0, n * sizeof(float_t)));
    } else {
        t->grad = NULL;
    }
    return t;
}

void tensor_to_host(Tensor *t, float *host_out) {
    int n = tensor_numel(t->ndim, t->shape);
    CUDA_CHECK(cudaMemcpy(host_out, t->data, n * sizeof(float_t), cudaMemcpyDeviceToHost));
}

void tensor_set_data(Tensor *t, const float *host_data) {
    int n = tensor_numel(t->ndim, t->shape);
    CUDA_CHECK(cudaMemcpy(t->data, host_data, n * sizeof(float_t), cudaMemcpyHostToDevice));
}

void tensor_grad_to_host(Tensor *t, float *host_out) {
    int n = tensor_numel(t->ndim, t->shape);
    if (!t->grad) {
        memset(host_out, 0, n * sizeof(float_t));
        return;
    }
    CUDA_CHECK(cudaMemcpy(host_out, t->grad, n * sizeof(float_t), cudaMemcpyDeviceToHost));
}

void tensor_zero_grad(Tensor *t) {
    if (!t || !t->requires_grad || !t->grad) return;
    int n = tensor_numel(t->ndim, t->shape);
    CUDA_CHECK(cudaMemset(t->grad, 0, n * sizeof(float_t)));
}

void tensor_add_dependency(Tensor *t, Tensor *dep_tensor, BackwardFn fn) {
    t->deps = (Dependency *)realloc(t->deps, sizeof(Dependency) * (t->num_deps + 1));
    t->deps[t->num_deps].tensor = dep_tensor;
    t->deps[t->num_deps].ctx = NULL;
    t->deps[t->num_deps].backward_fn = fn;
    t->num_deps++;
}

void tensor_add_dependency_with_ctx(Tensor *t, Tensor *dep_tensor, Tensor *ctx, BackwardFn fn) {
    t->deps = (Dependency *)realloc(t->deps, sizeof(Dependency) * (t->num_deps + 1));
    t->deps[t->num_deps].tensor = dep_tensor;
    t->deps[t->num_deps].ctx = ctx;
    t->deps[t->num_deps].backward_fn = fn;
    t->num_deps++;
}

void tensor_grad(Tensor *t, Tensor *grad) {
    if (!t || !grad || !t->requires_grad) return;
    int n = tensor_numel(t->ndim, t->shape);
    if (!t->grad) {
        t->grad = (float_t *)tensor_pool_alloc(n * sizeof(float_t));
        CUDA_CHECK(cudaMemset(t->grad, 0, n * sizeof(float_t)));
        t->owns_grad = 1;
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    tensor_add_inplace_kernel<<<grid, block>>>(t->grad, grad->data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* Appends t after everything it depends on (post-order), visiting each tensor once. */
static void backward_order(Tensor *t, Tensor ***order, int *n, int *cap) {
    for (int i = 0; i < *n; ++i)
        if ((*order)[i] == t) return;
    for (int i = 0; i < t->num_deps; ++i) {
        Tensor *d = t->deps[i].tensor;
        if (d && d->requires_grad) backward_order(d, order, n, cap);
    }
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 64;
        *order = (Tensor **)realloc(*order, sizeof(Tensor *) * (size_t)*cap);
    }
    (*order)[(*n)++] = t;
}

/* Reverse-mode pass over the graph reachable from `self`. Tensors are processed in reverse
 * topological order, so by the time a tensor's own backward functions run, every consumer of it
 * has already added its share into t->grad. That matters as soon as one tensor feeds two ops (a
 * residual connection): a plain recursion would push each path's gradient through everything
 * upstream separately, which repeats the work once per path - exponentially many times in a
 * stack of residual blocks. The root's incoming gradient is `grad_out`; for every other tensor
 * it is the accumulated t->grad (so callers must zero the grads they care about beforehand,
 * as they already do for weights). */
void tensor_backward(Tensor *self, Tensor *grad_out) {
    if (!self || !self->requires_grad) return;

    Tensor **order = NULL;
    int n = 0, cap = 0;
    backward_order(self, &order, &n, &cap);

    for (int k = n - 1; k >= 0; --k) {
        Tensor *t = order[k];
        if (t != self && !t->grad) continue; /* nothing reached it */
        for (int i = 0; i < t->num_deps; ++i) {
            Dependency *dep = &t->deps[i];
            Tensor *ctx = dep->ctx;
            if (!dep->backward_fn || !dep->tensor || !dep->tensor->requires_grad) continue;

            if (!ctx) {
                if (t->num_deps == 1) {
                    ctx = NULL;
                } else if (t->num_deps == 2) {
                    int other = (i == 0) ? 1 : 0;
                    ctx = t->deps[other].tensor;
                } else {
                    // For complex ops, use output tensor as default context.
                    ctx = t;
                }
            }

            Tensor *grad;
            if (t == self) {
                grad = dep->backward_fn(dep->tensor, ctx, grad_out);
            } else {
                /* backward functions read grad_out->data, so wrap t->grad without copying it */
                Tensor view = *t;
                view.data = t->grad;
                grad = dep->backward_fn(dep->tensor, ctx, &view);
            }
            if (!grad) continue;
            tensor_grad(dep->tensor, grad);
            tensor_free(grad);
        }
    }
    free(order);
}

void tensor_update(Tensor *t, float lr) {
    if (!t || !t->requires_grad || !t->grad) return;
    int n = tensor_numel(t->ndim, t->shape);
    int block = 256;
    int grid = (n + block - 1) / block;
    tensor_sgd_step_kernel<<<grid, block>>>(t->data, t->grad, lr, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void tensor_update_momentum(Tensor *t, float lr, float momentum) {
    if (momentum <= 0.0f) {
        tensor_update(t, lr);
        return;
    }
    if (!t || !t->requires_grad || !t->grad) return;
    int n = tensor_numel(t->ndim, t->shape);
    if (!t->velocity) {
        t->velocity = (float_t *)tensor_pool_alloc(n * sizeof(float_t));
        CUDA_CHECK(cudaMemset(t->velocity, 0, n * sizeof(float_t)));
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    tensor_momentum_step_kernel<<<grid, block>>>(t->data, t->grad, t->velocity, lr, momentum, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void tensor_free(Tensor *t) {
    if (!t) return;
    size_t bytes = (size_t)tensor_numel(t->ndim, t->shape) * sizeof(float_t);
    if (t->velocity) tensor_pool_free(t->velocity, bytes);
    if (t->owns_grad && t->grad) tensor_pool_free(t->grad, bytes);
    if (t->owns_data && t->data) tensor_pool_free(t->data, bytes);
    if (t->shape) free(t->shape);
    if (t->deps) free(t->deps);
    if (t->meta) free(t->meta);
    free(t);
}

void tensor_print(Tensor *t) {
    if (!t) {
        printf("Tensor is NULL\n");
        return;
    }
    int n = tensor_numel(t->ndim, t->shape);
    float *host = (float *)malloc(sizeof(float) * n);
    CUDA_CHECK(cudaMemcpy(host, t->data, sizeof(float_t) * n, cudaMemcpyDeviceToHost));
    printf("Tensor data: [");
    for (int i = 0; i < n; ++i) printf("%f ", host[i]);
    printf("]\n");
    free(host);
}

void tensor_print_grad(Tensor *t) {
    if (!t || !t->grad) {
        printf("Tensor grad: (null)\n");
        return;
    }
    int n = tensor_numel(t->ndim, t->shape);
    float *host = (float *)malloc(sizeof(float) * n);
    CUDA_CHECK(cudaMemcpy(host, t->grad, sizeof(float_t) * n, cudaMemcpyDeviceToHost));
    printf("Tensor grad: [");
    for (int i = 0; i < n; ++i) printf("%f ", host[i]);
    printf("]\n");
    free(host);
}
