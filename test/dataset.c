#include "dataset.h"

#include <stdlib.h>

void generate_checkerboard_data(float *inputs, int *targets, int total_data, int num_squares) {
    int i;
    for (i = 0; i < total_data; ++i) {
        float x1 = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
        float x2 = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
        int grid_x;
        int grid_y;

        inputs[i * 2 + 0] = x1;
        inputs[i * 2 + 1] = x2;

        grid_x = (int)((x1 + 1.0f) * (num_squares / 2));
        grid_y = (int)((x2 + 1.0f) * (num_squares / 2));
        targets[i] = (grid_x + grid_y) % 2;
    }
}
