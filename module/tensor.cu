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

__global__ void tensor_set_zero_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

__global__ void tensor_add_inplace_kernel(float *dst, const float *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
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

    t->shape = (int *)malloc(sizeof(int) * ndim);
    memcpy(t->shape, shape, sizeof(int) * ndim);

    int n = tensor_numel(ndim, shape);
    CUDA_CHECK(cudaMalloc((void **)&t->data, n * sizeof(float_t)));
    CUDA_CHECK(cudaMemset(t->data, 0, n * sizeof(float_t)));

    if (requires_grad) {
        CUDA_CHECK(cudaMalloc((void **)&t->grad, n * sizeof(float_t)));
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

    t->shape = (int *)malloc(sizeof(int) * ndim);
    memcpy(t->shape, shape, sizeof(int) * ndim);
    t->data = device_data;

    if (requires_grad) {
        int n = tensor_numel(ndim, shape);
        CUDA_CHECK(cudaMalloc((void **)&t->grad, n * sizeof(float_t)));
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

void tensor_zero_grad(Tensor *t) {
    if (!t || !t->requires_grad || !t->grad) return;
    int n = tensor_numel(t->ndim, t->shape);
    CUDA_CHECK(cudaMemset(t->grad, 0, n * sizeof(float_t)));
}

void tensor_add_dependency(Tensor *t, Tensor *dep_tensor, BackwardFn fn) {
    t->deps = (Dependency *)realloc(t->deps, sizeof(Dependency) * (t->num_deps + 1));
    t->deps[t->num_deps].tensor = dep_tensor;
    t->deps[t->num_deps].backward_fn = fn;
    t->num_deps++;
}

void tensor_grad(Tensor *t, Tensor *grad) {
    if (!t || !grad || !t->requires_grad) return;
    int n = tensor_numel(t->ndim, t->shape);
    if (!t->grad) {
        CUDA_CHECK(cudaMalloc((void **)&t->grad, n * sizeof(float_t)));
        CUDA_CHECK(cudaMemset(t->grad, 0, n * sizeof(float_t)));
        t->owns_grad = 1;
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    tensor_add_inplace_kernel<<<grid, block>>>(t->grad, grad->data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void tensor_backward(Tensor *self, Tensor *grad_out) {
    if (!self || !self->requires_grad) return;

    if (self->num_deps == 3) {
        Dependency *dep0 = &self->deps[0];
        Dependency *dep1 = &self->deps[1];
        Dependency *dep2 = &self->deps[2];

        // dep0/dep1: gradients from matmul
        if (dep0->backward_fn) {
            Tensor *grad_a = dep0->backward_fn(dep0->tensor, dep1->tensor, grad_out);
            tensor_grad(dep0->tensor, grad_a);
            tensor_backward(dep0->tensor, grad_a);
            tensor_free(grad_a);
        }
        if (dep1->backward_fn) {
            Tensor *grad_b = dep1->backward_fn(dep0->tensor, dep1->tensor, grad_out);
            tensor_grad(dep1->tensor, grad_b);
            tensor_backward(dep1->tensor, grad_b);
            tensor_free(grad_b);
        }

        // dep2: bias gradient (x should be current output tensor shape [batch, out_features])
        if (dep2->backward_fn) {
            Tensor *grad_c = dep2->backward_fn(self, dep2->tensor, grad_out);
            tensor_grad(dep2->tensor, grad_c);
            tensor_backward(dep2->tensor, grad_c);
            tensor_free(grad_c);
        }
    } else if (self->num_deps == 2) {
        Dependency *dep0 = &self->deps[0];
        Dependency *dep1 = &self->deps[1];

        if (dep0->backward_fn) {
            Tensor *grad_a = dep0->backward_fn(dep0->tensor, dep1->tensor, grad_out);
            tensor_grad(dep0->tensor, grad_a);
            tensor_backward(dep0->tensor, grad_a);
            tensor_free(grad_a);
        }
        if (dep1->backward_fn) {
            Tensor *grad_b = dep1->backward_fn(dep0->tensor, dep1->tensor, grad_out);
            tensor_grad(dep1->tensor, grad_b);
            tensor_backward(dep1->tensor, grad_b);
            tensor_free(grad_b);
        }
    } else if (self->num_deps == 1) {
        Dependency *dep = &self->deps[0];
        if (dep->backward_fn) {
            Tensor *grad = dep->backward_fn(dep->tensor, NULL, grad_out);
            tensor_grad(dep->tensor, grad);
            tensor_backward(dep->tensor, grad);
            tensor_free(grad);
        }
    }
}

void tensor_update(Tensor *t, float lr) {
    if (!t || !t->requires_grad || !t->grad) return;
    int n = tensor_numel(t->ndim, t->shape);
    int block = 256;
    int grid = (n + block - 1) / block;
    tensor_sgd_step_kernel<<<grid, block>>>(t->data, t->grad, lr, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->owns_grad && t->grad) CUDA_CHECK(cudaFree(t->grad));
    if (t->owns_data && t->data) CUDA_CHECK(cudaFree(t->data));
    if (t->shape) free(t->shape);
    if (t->deps) free(t->deps);
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
