#ifndef CONV2D_H
#define CONV2D_H

#include "tensor.cuh"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Conv2D {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    Tensor *weight;  // (out_channels, in_channels, kernel_size, kernel_size)
    Tensor *bias;    // (out_channels), or NULL if use_bias == 0
} Conv2D;

Conv2D *conv2d_new(int in_channels, int out_channels, int kernel_size, int stride, int padding, int use_bias);
void conv2d_free(Conv2D *self);

// x must be a 4D tensor (B, in_channels, H, W). Returns (B, out_channels, H_out, W_out)
// with H_out = (H + 2*padding - kernel_size) / stride + 1 (same for W_out).
Tensor *conv2d_forward(Conv2D *self, Tensor *x);

Tensor *conv2d_backward_input(Tensor *x, Tensor *weight, Tensor *grad_out);
Tensor *conv2d_backward_weight(Tensor *weight, Tensor *x, Tensor *grad_out);
Tensor *conv2d_backward_bias(Tensor *bias, Tensor *ctx, Tensor *grad_out);

Tensor *conv2d_tensor(Conv2D *self);
void conv2d_print_weight(Conv2D *self, const char *name);
void conv2d_print_grad(Conv2D *self, const char *name);

#ifdef __cplusplus
}
#endif

#endif
