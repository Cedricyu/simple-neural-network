#include <stdio.h>

#include "CudaDeviceInfo.cuh"
#include "test.h"

int main(void) {
    cuda_device_info_print_all();
    printf("Running tests...\n");

    test_linear();
    test_conv2d();
    test_mnist();
    return 0;
}
