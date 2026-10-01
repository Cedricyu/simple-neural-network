#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "conv2d.cuh"
#include "convkernels.cuh"
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

// Attached to Conv2D::weight->meta so the backward_fns (which only ever see
// the weight tensor as one of their two Tensor* args) can recover stride and
// padding without widening the generic BackwardFn signature.
typedef struct Conv2DMeta {
    int stride;
    int padding;
} Conv2DMeta;

/* He/Kaiming-uniform: tuned for ReLU activations (Xavier assumes tanh/linear
 * and under-scales here, which lets deep conv stacks collapse into dead
 * ReLUs within the first few SGD steps). */
static void kaiming_init(Tensor *t, int fan_in) {
    float limit = sqrtf(6.0f / fan_in);
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

Conv2D *conv2d_new(int in_channels, int out_channels, int kernel_size, int stride, int padding, int use_bias) {
    Conv2D *self = (Conv2D *)malloc(sizeof(Conv2D));
    if (!self) return NULL;

    self->in_channels = in_channels;
    self->out_channels = out_channels;
    self->kernel_size = kernel_size;
    self->stride = stride;
    self->padding = padding;

    int w_shape[4] = {out_channels, in_channels, kernel_size, kernel_size};
    self->weight = tensor_create(4, w_shape, 1);

    int fan_in = in_channels * kernel_size * kernel_size;
    kaiming_init(self->weight, fan_in);

    Conv2DMeta *meta = (Conv2DMeta *)malloc(sizeof(Conv2DMeta));
    meta->stride = stride;
    meta->padding = padding;
    self->weight->meta = meta;

    if (use_bias) {
        int b_shape[1] = {out_channels};
        self->bias = tensor_create(1, b_shape, 1);
    } else {
        self->bias = NULL;
    }

    return self;
}

void conv2d_free(Conv2D *self) {
    if (!self) return;
    tensor_free(self->weight);
    if (self->bias) tensor_free(self->bias);
    free(self);
}

Tensor *conv2d_forward(Conv2D *self, Tensor *x) {
    if (x->ndim != 4) {
        printf("conv2d_forward only supports 4D tensors (B, C_in, H, W).\n");
        return NULL;
    }
    if (x->shape[1] != self->in_channels) {
        printf("conv2d_forward: channel mismatch (input has %d, layer expects %d)\n", x->shape[1],
               self->in_channels);
        return NULL;
    }

    int B = x->shape[0];
    int C_in = x->shape[1];
    int H = x->shape[2];
    int W = x->shape[3];
    int C_out = self->out_channels;
    int K = self->kernel_size;
    int stride = self->stride;
    int padding = self->padding;

    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    if (H_out <= 0 || W_out <= 0) {
        printf("conv2d_forward: kernel/stride/padding too large for input size %dx%d\n", H, W);
        return NULL;
    }

    int y_shape[4] = {B, C_out, H_out, W_out};
    int requires_grad = x->requires_grad || self->weight->requires_grad || (self->bias && self->bias->requires_grad);
    Tensor *y = tensor_create(4, y_shape, requires_grad);

    dim3 block(16, 16);
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, B * C_out);
    conv2d_forward_kernel<<<grid, block>>>(x->data, self->weight->data, y->data, B, C_in, H, W, K, C_out, H_out,
                                            W_out, padding, stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (self->bias) {
        int total = B * C_out * H_out * W_out;
        int bblock = 256;
        int bgrid = (total + bblock - 1) / bblock;
        conv2d_add_bias_kernel<<<bgrid, bblock>>>(y->data, self->bias->data, B, C_out, H_out, W_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (x->requires_grad) tensor_add_dependency_with_ctx(y, x, self->weight, conv2d_backward_input);
    if (self->weight->requires_grad) tensor_add_dependency_with_ctx(y, self->weight, x, conv2d_backward_weight);
    if (self->bias && self->bias->requires_grad) tensor_add_dependency(y, self->bias, conv2d_backward_bias);

    return y;
}

Tensor *conv2d_backward_input(Tensor *x, Tensor *weight, Tensor *grad_out) {
    Conv2DMeta *meta = (Conv2DMeta *)weight->meta;
    int B = x->shape[0];
    int C_in = x->shape[1];
    int H = x->shape[2];
    int W = x->shape[3];
    int C_out = weight->shape[0];
    int K = weight->shape[2];
    int H_out = grad_out->shape[2];
    int W_out = grad_out->shape[3];

    int shape[4] = {B, C_in, H, W};
    Tensor *grad_x = tensor_create(4, shape, 0);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16, B * C_in);
    conv2d_backward_input_kernel<<<grid, block>>>(grad_out->data, weight->data, grad_x->data, B, C_in, H, W, K,
                                                    C_out, H_out, W_out, meta->padding, meta->stride);
    CUDA_CHECK(cudaDeviceSynchronize());
    return grad_x;
}

Tensor *conv2d_backward_weight(Tensor *weight, Tensor *x, Tensor *grad_out) {
    Conv2DMeta *meta = (Conv2DMeta *)weight->meta;
    int C_out = weight->shape[0];
    int C_in = weight->shape[1];
    int K = weight->shape[2];
    int B = x->shape[0];
    int H = x->shape[2];
    int W = x->shape[3];
    int H_out = grad_out->shape[2];
    int W_out = grad_out->shape[3];

    int shape[4] = {C_out, C_in, K, K};
    Tensor *grad_w = tensor_create(4, shape, 0);

    dim3 block(K, K);
    dim3 grid(1, 1, C_out * C_in);
    conv2d_backward_weight_kernel<<<grid, block>>>(x->data, grad_out->data, grad_w->data, B, C_in, H, W, K, C_out,
                                                     H_out, W_out, meta->padding, meta->stride);
    CUDA_CHECK(cudaDeviceSynchronize());
    return grad_w;
}

Tensor *conv2d_backward_bias(Tensor *bias, Tensor *ctx, Tensor *grad_out) {
    (void)bias;
    (void)ctx;
    int B = grad_out->shape[0];
    int C_out = grad_out->shape[1];
    int H_out = grad_out->shape[2];
    int W_out = grad_out->shape[3];

    int shape[1] = {C_out};
    Tensor *grad_b = tensor_create(1, shape, 0);

    int block = 256;
    int grid = (C_out + block - 1) / block;
    conv2d_bias_grad_kernel<<<grid, block>>>(grad_out->data, grad_b->data, B, C_out, H_out, W_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    return grad_b;
}

Tensor *conv2d_tensor(Conv2D *self) { return self->weight; }

void conv2d_print_weight(Conv2D *self, const char *name) {
    if (name && name[0]) printf("%s:\n", name);
    tensor_print(self->weight);
}

void conv2d_print_grad(Conv2D *self, const char *name) {
    if (name && name[0]) printf("%s:\n", name);
    tensor_print_grad(self->weight);
}

#ifdef __cplusplus
}
#endif
