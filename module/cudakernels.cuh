#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#define BLOCK_SIZE 32

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int M, int N, int K);

__global__ void addBiasKernel(float *C, float *bias, int M, int N);

__global__ void biasGradientKernel(float *grad_output, float *grad_bias, int batch_size, int out_features);

__global__ void matrixTransposeKernel(float *input, float *output, int rows, int cols);
__global__ void matrixCopyKernel(float *input, float *output, int rows, int cols);

#endif
