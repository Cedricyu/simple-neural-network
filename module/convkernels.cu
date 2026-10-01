#include "convkernels.cuh"

__global__ void conv2d_forward_kernel(const float *x, const float *w, float *y, int B, int C_in, int H, int W,
                                       int K, int C_out, int H_out, int W_out, int padding, int stride) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;  // flattened (b, c_out)
    if (out_x >= W_out || out_y >= H_out) return;

    int b = bc / C_out;
    int c_out = bc % C_out;

    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                int in_y = out_y * stride - padding + i;
                int in_x = out_x * stride - padding + j;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    int x_idx = ((b * C_in + c_in) * H + in_y) * W + in_x;
                    int w_idx = ((c_out * C_in + c_in) * K + i) * K + j;
                    acc += x[x_idx] * w[w_idx];
                }
            }
        }
    }

    int y_idx = ((b * C_out + c_out) * H_out + out_y) * W_out + out_x;
    y[y_idx] = acc;
}

__global__ void conv2d_backward_input_kernel(const float *grad_y, const float *w, float *grad_x, int B, int C_in,
                                              int H, int W, int K, int C_out, int H_out, int W_out, int padding,
                                              int stride) {
    int in_x = blockIdx.x * blockDim.x + threadIdx.x;
    int in_y = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;  // flattened (b, c_in)
    if (in_x >= W || in_y >= H) return;

    int b = bc / C_in;
    int c_in = bc % C_in;

    float acc = 0.0f;
    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                int py = in_y + padding - i;
                int px = in_x + padding - j;
                if (py % stride != 0 || px % stride != 0) continue;
                int out_y = py / stride;
                int out_x = px / stride;
                if (out_y >= 0 && out_y < H_out && out_x >= 0 && out_x < W_out) {
                    int gy_idx = ((b * C_out + c_out) * H_out + out_y) * W_out + out_x;
                    int w_idx = ((c_out * C_in + c_in) * K + i) * K + j;
                    acc += grad_y[gy_idx] * w[w_idx];
                }
            }
        }
    }

    int x_idx = ((b * C_in + c_in) * H + in_y) * W + in_x;
    grad_x[x_idx] = acc;
}

__global__ void conv2d_backward_weight_kernel(const float *x, const float *grad_y, float *grad_w, int B, int C_in,
                                               int H, int W, int K, int C_out, int H_out, int W_out, int padding,
                                               int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int co_ci = blockIdx.z;  // flattened (c_out, c_in)
    if (j >= K || i >= K) return;

    int c_out = co_ci / C_in;
    int c_in = co_ci % C_in;

    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int out_y = 0; out_y < H_out; ++out_y) {
            for (int out_x = 0; out_x < W_out; ++out_x) {
                int in_y = out_y * stride - padding + i;
                int in_x = out_x * stride - padding + j;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    int x_idx = ((b * C_in + c_in) * H + in_y) * W + in_x;
                    int gy_idx = ((b * C_out + c_out) * H_out + out_y) * W_out + out_x;
                    acc += x[x_idx] * grad_y[gy_idx];
                }
            }
        }
    }

    int w_idx = ((c_out * C_in + c_in) * K + i) * K + j;
    grad_w[w_idx] = acc;
}

__global__ void conv2d_add_bias_kernel(float *y, const float *bias, int B, int C_out, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_out * H_out * W_out;
    if (idx >= total) return;

    int hw = H_out * W_out;
    int c = (idx / hw) % C_out;
    y[idx] += bias[c];
}

__global__ void conv2d_bias_grad_kernel(const float *grad_y, float *grad_bias, int B, int C_out, int H_out,
                                         int W_out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C_out) return;

    int hw = H_out * W_out;
    float sum = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < hw; ++k) {
            sum += grad_y[(b * C_out + c) * hw + k];
        }
    }
    grad_bias[c] = sum;
}
