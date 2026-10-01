#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudakernels.cuh"
#include "linear.cuh"
#include "tensor.cuh"
#include "tensorops.cuh"

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t _err = (call);                                                                \
        if (_err != cudaSuccess) {                                                                \
            fprintf(stderr, "[cuda] %s failed at %s:%d: %s\n", #call, __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                                    \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

static void xavier_init(Tensor *t, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    int size = tensor_numel(t->ndim, t->shape);
    float *host = (float *)malloc(sizeof(float) * size);
    for (int i = 0; i < size; ++i) {
        float r = (float)rand() / (float)RAND_MAX;
        host[i] = r * 2.0f * limit - limit;
    }
    CUDA_CHECK(cudaMemcpy(t->data, host, sizeof(float) * size, cudaMemcpyHostToDevice));
    free(host);
}

#ifdef __cplusplus
extern "C" {
#endif

Linear *linear_new(int in_f, int out_f) {
    Linear *self = (Linear *)malloc(sizeof(Linear));
    if (!self) return NULL;

    self->in_features = in_f;
    self->out_features = out_f;

    int w_shape[2] = {in_f, out_f};
    int b_shape[2] = {1, out_f};
    self->weight = tensor_create(2, w_shape, 1);
    self->bias = tensor_create(2, b_shape, 1);

    xavier_init(self->weight, in_f, out_f);
    return self;
}

void linear_free(Linear *self) {
    if (!self) return;
    tensor_free(self->weight);
    tensor_free(self->bias);
    free(self);
}

Tensor *linear_forward(Linear *self, Tensor *input) {
    Tensor *out = tensor_matmul(input, self->weight);

    int m = out->shape[0];
    int n = out->shape[1];
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    addBiasKernel<<<grid, block>>>(out->data, self->bias->data, m, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (self->bias->requires_grad) {
        tensor_add_dependency(out, self->bias, tensor_add_bias_backward_bias);
    }

    return out;
}

Tensor *linear_tensor(Linear *self) { return self->weight; }

void linear_print_weight(Linear *self, const char *name) {
    if (name && name[0]) printf("%s:\n", name);
    tensor_print(self->weight);
}

void linear_print_grad(Linear *self, const char *name) {
    if (name && name[0]) printf("%s:\n", name);
    tensor_print_grad(self->weight);
}

#ifdef __cplusplus
}
#endif
