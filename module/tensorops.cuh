#ifndef TENSOR_OPS_CUH
#define TENSOR_OPS_CUH

#include "tensor.cuh"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

Tensor *tensor_matmul_backward_a(Tensor *a, Tensor *b, Tensor *grad_out);
Tensor *tensor_matmul_backward_b(Tensor *a, Tensor *b, Tensor *grad_out);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
Tensor *tensor_add_bias(Tensor *x, Tensor *bias);
Tensor *tensor_add_bias_backward_input(Tensor *x, Tensor *bias, Tensor *grad_out);
Tensor *tensor_add_bias_backward_bias(Tensor *x, Tensor *bias, Tensor *grad_out);
Tensor *tensor_relu(Tensor *x);
Tensor *tensor_relu_backward(Tensor *x, Tensor *n, Tensor *grad_out);
Tensor *tensor_flatten(Tensor *x);
Tensor *tensor_flatten_backward(Tensor *x, Tensor *ctx, Tensor *grad_out);
// Elementwise a + b of two same-shape tensors (the residual connection of a ResNet block).
// Both inputs get the incoming gradient unchanged.
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_add_backward(Tensor *x, Tensor *ctx, Tensor *grad_out);
// (B, ...) -> (B, c, h, w), c*h*w values per sample: a zero-copy view like tensor_flatten, for turning the
// flat output of a Linear layer back into a feature map (the VAE decoder).
Tensor *tensor_reshape4(Tensor *x, int c, int h, int w);
// Channel concatenation of (B, C1, H, W) and (B, C2, H, W) into (B, C1+C2, H, W): the U-Net skip connection.
// Each input gets back its own channel slice of the incoming gradient.
Tensor *tensor_concat_channels(Tensor *a, Tensor *b);
Tensor *tensor_concat_backward_first(Tensor *a, Tensor *b, Tensor *grad_out);
Tensor *tensor_concat_backward_second(Tensor *b, Tensor *a, Tensor *grad_out);
void fill_tensor_with_random(Tensor *t);
Tensor *tensor_clone(Tensor *t);
void tensor_print_graph_dot(Tensor *self);
void tensor_print_dimesions(Tensor *self);

#ifdef __cplusplus
}
#endif

#endif
