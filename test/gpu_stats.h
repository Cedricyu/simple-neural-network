/* GPU numbers for the dashboard, read through NVML. libnvidia-ml is loaded at run time (dlopen), so a machine without
 * it still builds and runs: every value is then -1 ("unknown") except what the CUDA runtime tells us (the SM count).
 *
 * What the numbers mean: util_gpu is NVML's "GPU utilization" - the share of the last sample period (a fraction of a
 * second) during which at least one kernel was running on the device. It is a time share, not a count of busy SMs;
 * the dashboard turns it into "about N of the SMs" only as an estimate. Memory is device-wide (every process). */
#ifndef GPU_STATS_H
#define GPU_STATS_H

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "CudaDeviceInfo.cuh"
#include "train_record.h"

typedef struct NvmlUtil {
    unsigned int gpu, memory;
} NvmlUtil;
typedef struct NvmlMem {
    unsigned long long total, free, used;
} NvmlMem;

typedef struct NvmlApi {
    int ready;
    void *handle;
    int (*init)(void);
    int (*by_index)(unsigned int, void **);
    int (*name)(void *, char *, unsigned int);
    int (*util)(void *, NvmlUtil *);
    int (*mem)(void *, NvmlMem *);
    int (*temp)(void *, int, unsigned int *);
    int (*power)(void *, unsigned int *);
    int (*clock)(void *, int, unsigned int *);
} NvmlApi;

static NvmlApi g_nvml;
static void *g_nvml_dev;
static GpuStats g_gpu_base; /* what does not change: the name and the SM count */

/* Call once; safe to call again. Fills the static part and tries to open NVML. */
static void gpu_stats_init(void) {
    if (g_gpu_base.sm_total > 0 || g_nvml.ready) return;
    memset(&g_gpu_base, 0, sizeof(g_gpu_base));
    g_gpu_base.util_gpu = g_gpu_base.util_mem = g_gpu_base.mem_used_mb = g_gpu_base.mem_total_mb = -1;
    g_gpu_base.temp_c = g_gpu_base.power_w = g_gpu_base.sm_clock_mhz = -1;
    int n = 0;
    DeviceInfo *d = cuda_device_info_get_all(&n);
    if (d && n > 0) {
        snprintf(g_gpu_base.name, sizeof(g_gpu_base.name), "%.63s", d[0].name);
        g_gpu_base.sm_total = d[0].num_sms;
    }
    if (d) cuda_device_info_free(d);

    g_nvml.handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!g_nvml.handle) return;
    *(void **)&g_nvml.init = dlsym(g_nvml.handle, "nvmlInit_v2");
    *(void **)&g_nvml.by_index = dlsym(g_nvml.handle, "nvmlDeviceGetHandleByIndex_v2");
    *(void **)&g_nvml.name = dlsym(g_nvml.handle, "nvmlDeviceGetName");
    *(void **)&g_nvml.util = dlsym(g_nvml.handle, "nvmlDeviceGetUtilizationRates");
    *(void **)&g_nvml.mem = dlsym(g_nvml.handle, "nvmlDeviceGetMemoryInfo");
    *(void **)&g_nvml.temp = dlsym(g_nvml.handle, "nvmlDeviceGetTemperature");
    *(void **)&g_nvml.power = dlsym(g_nvml.handle, "nvmlDeviceGetPowerUsage");
    *(void **)&g_nvml.clock = dlsym(g_nvml.handle, "nvmlDeviceGetClockInfo");
    if (!g_nvml.init || !g_nvml.by_index || g_nvml.init() != 0 || g_nvml.by_index(0, &g_nvml_dev) != 0) return;
    g_nvml.ready = 1;
}

/* The latest readings (the static part from gpu_stats_init plus whatever NVML answers). */
static GpuStats gpu_stats_read(void) {
    gpu_stats_init();
    GpuStats s = g_gpu_base;
    if (!g_nvml.ready) return s;
    NvmlUtil u;
    NvmlMem m;
    unsigned int v;
    if (g_nvml.util && g_nvml.util(g_nvml_dev, &u) == 0) {
        s.util_gpu = (int)u.gpu;
        s.util_mem = (int)u.memory;
    }
    if (g_nvml.mem && g_nvml.mem(g_nvml_dev, &m) == 0) {
        s.mem_used_mb = (int)(m.used >> 20);
        s.mem_total_mb = (int)(m.total >> 20);
    }
    if (g_nvml.temp && g_nvml.temp(g_nvml_dev, 0, &v) == 0) s.temp_c = (int)v;
    if (g_nvml.power && g_nvml.power(g_nvml_dev, &v) == 0) s.power_w = (int)(v / 1000);
    if (g_nvml.clock && g_nvml.clock(g_nvml_dev, 1, &v) == 0) s.sm_clock_mhz = (int)v;
    return s;
}

#endif
