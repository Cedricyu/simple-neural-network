#ifndef MNIST_H
#define MNIST_H

typedef struct MnistData {
    float *images; /* count * channels * rows * cols (planar: all of channel 0, then channel 1, ...), in [0, 1] */
    int *labels;   /* count */
    int count;
    int rows;
    int cols;
    int channels; /* 1 for MNIST-style files; 3 for the colour variant (IDX magic 0x804) */
    unsigned char *masks; /* segmentation only (see mnist_load_masks): count * rows * cols class indices, else NULL */
} MnistData;

/* Returns 0 on success, -1 on failure (bad path, truncated file, magic mismatch). */
int mnist_load(const char *images_path, const char *labels_path, MnistData *out);
/* Loads a segmentation mask file (IDX magic 0x803, one class index per pixel) for already-loaded images;
 * the sample count and image size must match. Returns 0 on success. */
int mnist_load_masks(const char *masks_path, MnistData *data);
void mnist_free(MnistData *data);

#endif
