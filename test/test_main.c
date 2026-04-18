#include <stdio.h>

#include "CudaDeviceInfo.cuh"
#include "test.h"

int main(void) {
    cuda_device_info_print_all();
    printf("Running tests...\n");

    test_linear();
    return 0;
}
