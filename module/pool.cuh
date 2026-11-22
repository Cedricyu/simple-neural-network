#ifndef POOL_H
#define POOL_H

#include "tensor.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/* x is (B, C, H, W). Window kernel x kernel, no padding:
 * H_out = (H - kernel) / stride + 1. The gradient goes to the first maximum of each window. */
Tensor *maxpool2d_forward(Tensor *x, int kernel, int stride);
Tensor *maxpool2d_backward(Tensor *x, Tensor *out, Tensor *grad_out);

/* (B, C, H, W) -> (B, C): the mean of every channel's H*W values. The result is already flat,
 * so a Linear layer can follow directly. */
Tensor *global_avg_pool_forward(Tensor *x);
Tensor *global_avg_pool_backward(Tensor *x, Tensor *ctx, Tensor *grad_out);

#ifdef __cplusplus
}
#endif

#endif
