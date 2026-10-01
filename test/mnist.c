#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>

static unsigned int read_be_u32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0xFFFFFFFFu;
    return ((unsigned int)b[0] << 24) | ((unsigned int)b[1] << 16) | ((unsigned int)b[2] << 8) | (unsigned int)b[3];
}

int mnist_load(const char *images_path, const char *labels_path, MnistData *out) {
    FILE *f_img = NULL, *f_lbl = NULL;
    unsigned int magic, n_images, rows, cols, n_labels;
    unsigned char *raw_pixels = NULL, *raw_labels = NULL;
    long n;
    int i;

    out->images = NULL;
    out->labels = NULL;
    out->count = 0;

    f_img = fopen(images_path, "rb");
    if (!f_img) {
        fprintf(stderr, "mnist_load: failed to open %s\n", images_path);
        return -1;
    }
    f_lbl = fopen(labels_path, "rb");
    if (!f_lbl) {
        fprintf(stderr, "mnist_load: failed to open %s\n", labels_path);
        fclose(f_img);
        return -1;
    }

    magic = read_be_u32(f_img);
    if (magic != 0x00000803u) {
        fprintf(stderr, "mnist_load: bad image magic 0x%08x in %s\n", magic, images_path);
        goto fail;
    }
    n_images = read_be_u32(f_img);
    rows = read_be_u32(f_img);
    cols = read_be_u32(f_img);

    magic = read_be_u32(f_lbl);
    if (magic != 0x00000801u) {
        fprintf(stderr, "mnist_load: bad label magic 0x%08x in %s\n", magic, labels_path);
        goto fail;
    }
    n_labels = read_be_u32(f_lbl);

    if (n_images != n_labels) {
        fprintf(stderr, "mnist_load: image/label count mismatch (%u vs %u)\n", n_images, n_labels);
        goto fail;
    }

    n = (long)n_images * (long)rows * (long)cols;
    raw_pixels = (unsigned char *)malloc((size_t)n);
    raw_labels = (unsigned char *)malloc((size_t)n_images);
    if (!raw_pixels || !raw_labels) {
        fprintf(stderr, "mnist_load: out of memory\n");
        goto fail;
    }

    if (fread(raw_pixels, 1, (size_t)n, f_img) != (size_t)n) {
        fprintf(stderr, "mnist_load: truncated image file %s\n", images_path);
        goto fail;
    }
    if (fread(raw_labels, 1, (size_t)n_images, f_lbl) != (size_t)n_images) {
        fprintf(stderr, "mnist_load: truncated label file %s\n", labels_path);
        goto fail;
    }

    out->images = (float *)malloc(sizeof(float) * (size_t)n);
    out->labels = (int *)malloc(sizeof(int) * (size_t)n_images);
    if (!out->images || !out->labels) {
        fprintf(stderr, "mnist_load: out of memory\n");
        free(out->images);
        free(out->labels);
        out->images = NULL;
        out->labels = NULL;
        goto fail;
    }

    for (i = 0; i < n; ++i) out->images[i] = (float)raw_pixels[i] / 255.0f;
    for (i = 0; i < (int)n_images; ++i) out->labels[i] = (int)raw_labels[i];

    out->count = (int)n_images;
    out->rows = (int)rows;
    out->cols = (int)cols;

    free(raw_pixels);
    free(raw_labels);
    fclose(f_img);
    fclose(f_lbl);
    return 0;

fail:
    free(raw_pixels);
    free(raw_labels);
    fclose(f_img);
    fclose(f_lbl);
    return -1;
}

void mnist_free(MnistData *data) {
    if (!data) return;
    free(data->images);
    free(data->labels);
    data->images = NULL;
    data->labels = NULL;
    data->count = 0;
}
