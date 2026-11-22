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

/* One block per (c_out, c_in) pair, CONV_WGRAD_THREADS threads: every thread walks a strided share
 * of the (b, out_y, out_x) positions - consecutive threads on consecutive out_x, so the loads
 * coalesce - and keeps one partial sum per kernel tap; the partials are then reduced across the
 * block. (The first version used K*K threads per block, each walking every position serially.) */
#define CONV_WGRAD_MAX_TAPS 49
__global__ void conv2d_backward_weight_kernel(const float *x, const float *grad_y, float *grad_w, int B, int C_in,
                                               int H, int W, int K, int C_out, int H_out, int W_out, int padding,
                                               int stride) {
    __shared__ float sh[CONV_WGRAD_THREADS];
    int co_ci = blockIdx.x;  // flattened (c_out, c_in)
    int c_out = co_ci / C_in;
    int c_in = co_ci % C_in;
    int taps = K * K;
    int positions = B * H_out * W_out;

    float acc[CONV_WGRAD_MAX_TAPS];
    for (int t = 0; t < taps; ++t) acc[t] = 0.0f;

    for (int p = threadIdx.x; p < positions; p += blockDim.x) {
        int out_x = p % W_out;
        int out_y = (p / W_out) % H_out;
        int b = p / (W_out * H_out);
        float gy = grad_y[((b * C_out + c_out) * H_out + out_y) * W_out + out_x];
        const float *xp = x + (size_t)(b * C_in + c_in) * H * W;
        for (int i = 0; i < K; ++i) {
            int in_y = out_y * stride - padding + i;
            if (in_y < 0 || in_y >= H) continue;
            for (int j = 0; j < K; ++j) {
                int in_x = out_x * stride - padding + j;
                if (in_x >= 0 && in_x < W) acc[i * K + j] += xp[in_y * W + in_x] * gy;
            }
        }
    }

    for (int t = 0; t < taps; ++t) {
        sh[threadIdx.x] = acc[t];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) grad_w[(size_t)co_ci * taps + t] = sh[0];
        __syncthreads();
    }
}

__global__ void conv2d_add_bias_kernel(float *y, const float *bias, int B, int C_out, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_out * H_out * W_out;
    if (idx >= total) return;

    int hw = H_out * W_out;
    int c = (idx / hw) % C_out;
    y[idx] += bias[c];
}

/* One block per output channel; the block's threads split the B * H_out * W_out values and reduce. */
__global__ void conv2d_bias_grad_kernel(const float *grad_y, float *grad_bias, int B, int C_out, int H_out,
                                         int W_out) {
    __shared__ float sh[CONV_WGRAD_THREADS];
    int c = blockIdx.x;
    int hw = H_out * W_out;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < B * hw; i += blockDim.x) {
        int b = i / hw, k = i % hw;
        sum += grad_y[((size_t)b * C_out + c) * hw + k];
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_bias[c] = sh[0];
}
