#ifndef CUDA_DEVICE_INFO_H
#define CUDA_DEVICE_INFO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DeviceInfo {
    int device_id;
    char name[256];
    int max_grid_dim_x;
    int max_grid_dim_y;
    int max_grid_dim_z;
    int max_threads_per_block;
    int max_threads_per_sm;
    size_t total_global_mem_mb;
    int num_sms;
} DeviceInfo;

DeviceInfo *cuda_device_info_get_all(int *count);
void cuda_device_info_free(DeviceInfo *devices);
void cuda_device_info_print_all(void);

#ifdef __cplusplus
}
#endif

#endif
