#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CudaDeviceInfo.cuh"

#ifdef __cplusplus
extern "C" {
#endif

DeviceInfo *cuda_device_info_get_all(int *count) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        if (count) *count = 0;
        return NULL;
    }

    DeviceInfo *devices = (DeviceInfo *)calloc((size_t)device_count, sizeof(DeviceInfo));
    if (!devices) {
        if (count) *count = 0;
        return NULL;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        devices[i].device_id = i;
        strncpy(devices[i].name, prop.name, sizeof(devices[i].name) - 1);
        devices[i].name[sizeof(devices[i].name) - 1] = '\0';
        devices[i].max_grid_dim_x = prop.maxGridSize[0];
        devices[i].max_grid_dim_y = prop.maxGridSize[1];
        devices[i].max_grid_dim_z = prop.maxGridSize[2];
        devices[i].max_threads_per_block = prop.maxThreadsPerBlock;
        devices[i].max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        devices[i].total_global_mem_mb = (size_t)(prop.totalGlobalMem / (1024 * 1024));
        devices[i].num_sms = prop.multiProcessorCount;
    }

    if (count) *count = device_count;
    return devices;
}

void cuda_device_info_free(DeviceInfo *devices) { free(devices); }

void cuda_device_info_print_all(void) {
    int count = 0;
    DeviceInfo *devices = cuda_device_info_get_all(&count);

    if (!devices || count == 0) {
        printf("No CUDA device available.\n");
        return;
    }

    for (int i = 0; i < count; ++i) {
        DeviceInfo *d = &devices[i];
        printf("Device %d: %s\n", d->device_id, d->name);
        printf("  Max Grid Dimensions: %d x %d x %d\n", d->max_grid_dim_x, d->max_grid_dim_y, d->max_grid_dim_z);
        printf("  Max Threads per Block: %d\n", d->max_threads_per_block);
        printf("  Max Threads per SM: %d\n", d->max_threads_per_sm);
        printf("  Total Global Memory: %zu MB\n", d->total_global_mem_mb);
        printf("  Number of SMs: %d\n", d->num_sms);
    }

    cuda_device_info_free(devices);
}

#ifdef __cplusplus
}
#endif
