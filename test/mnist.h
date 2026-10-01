#ifndef MNIST_H
#define MNIST_H

typedef struct MnistData {
    float *images; /* count * rows * cols, row-major, normalized to [0, 1] */
    int *labels;   /* count */
    int count;
    int rows;
    int cols;
} MnistData;

/* Returns 0 on success, -1 on failure (bad path, truncated file, magic mismatch). */
int mnist_load(const char *images_path, const char *labels_path, MnistData *out);
void mnist_free(MnistData *data);

#endif
