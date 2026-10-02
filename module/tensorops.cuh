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
void fill_tensor_with_random(Tensor *t);
Tensor *tensor_clone(Tensor *t);
void tensor_print_graph_dot(Tensor *self);
void tensor_print_dimesions(Tensor *self);

#ifdef __cplusplus
}
#endif

#endif
