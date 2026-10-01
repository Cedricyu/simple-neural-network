#ifndef CONVKERNELS_H
#define CONVKERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// x: (B, C_in, H, W), w: (C_out, C_in, K, K), y: (B, C_out, H_out, W_out)
__global__ void conv2d_forward_kernel(const float *x, const float *w, float *y, int B, int C_in, int H, int W,
                                       int K, int C_out, int H_out, int W_out, int padding, int stride);

// grad_y: (B, C_out, H_out, W_out), w: (C_out, C_in, K, K) -> grad_x: (B, C_in, H, W)
__global__ void conv2d_backward_input_kernel(const float *grad_y, const float *w, float *grad_x, int B, int C_in,
                                              int H, int W, int K, int C_out, int H_out, int W_out, int padding,
                                              int stride);

// x: (B, C_in, H, W), grad_y: (B, C_out, H_out, W_out) -> grad_w: (C_out, C_in, K, K)
__global__ void conv2d_backward_weight_kernel(const float *x, const float *grad_y, float *grad_w, int B, int C_in,
                                               int H, int W, int K, int C_out, int H_out, int W_out, int padding,
                                               int stride);

// y: (B, C_out, H_out, W_out) += bias[C_out], broadcast over batch and spatial dims
__global__ void conv2d_add_bias_kernel(float *y, const float *bias, int B, int C_out, int H_out, int W_out);

// grad_y: (B, C_out, H_out, W_out) -> grad_bias: (C_out)
__global__ void conv2d_bias_grad_kernel(const float *grad_y, float *grad_bias, int B, int C_out, int H_out,
                                         int W_out);

#ifdef __cplusplus
}
#endif

#endif  // CONVKERNELS_H
