#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "conv2d.cuh"
#include "tensor.cuh"
#include "tensorops.cuh"
#include "test.h"

#define IMG_SIZE 8
#define NUM_CLASSES 2
#define CONV1_OUT 4

typedef struct ConvModel {
    Conv2D *conv1;  // (1 -> CONV1_OUT), 3x3, stride 1, padding 1  => stays IMG_SIZE x IMG_SIZE
    Conv2D *conv2;  // (CONV1_OUT -> NUM_CLASSES), IMG_SIZE x IMG_SIZE kernel, stride 1, padding 0 => 1x1
} ConvModel;

typedef struct ForwardCache {
    Tensor *x;
    Tensor *h1;
    Tensor *a1;
    Tensor *logits;  // (B, NUM_CLASSES, 1, 1)
} ForwardCache;

static void softmax(const float *logits, float *output, int size) {
    float max_logit = logits[0];
    int i;
    for (i = 1; i < size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float sum_exp = 0.0f;
    for (i = 0; i < size; i++) {
        output[i] = expf(logits[i] - max_logit);
        sum_exp += output[i];
    }
    for (i = 0; i < size; i++) output[i] /= sum_exp;
}

static int argmax(const float *arr, int size) {
    int i, max_idx = 0;
    for (i = 1; i < size; i++) {
        if (arr[i] > arr[max_idx]) max_idx = i;
    }
    return max_idx;
}

/* class 0: bright vertical stripe on the left half, class 1: on the right half. */
static void generate_stripe_data(float *images, int *labels, int n) {
    int idx, y, x;
    for (idx = 0; idx < n; ++idx) {
        int label = rand() % NUM_CLASSES;
        labels[idx] = label;
        for (y = 0; y < IMG_SIZE; ++y) {
            for (x = 0; x < IMG_SIZE; ++x) {
                float noise = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
                int on_stripe = (label == 0) ? (x < IMG_SIZE / 2) : (x >= IMG_SIZE / 2);
                float val = on_stripe ? 0.8f : 0.1f;
                images[(idx * IMG_SIZE + y) * IMG_SIZE + x] = val + noise;
            }
        }
    }
}

static void model_init(ConvModel *m) {
    m->conv1 = conv2d_new(1, CONV1_OUT, 3, 1, 1, 1);
    m->conv2 = conv2d_new(CONV1_OUT, NUM_CLASSES, IMG_SIZE, 1, 0, 1);
}

static void model_free(ConvModel *m) {
    conv2d_free(m->conv1);
    conv2d_free(m->conv2);
}

static ForwardCache model_forward(ConvModel *m, Tensor *x) {
    ForwardCache c;
    c.x = x;
    c.h1 = conv2d_forward(m->conv1, x);
    c.a1 = tensor_relu(c.h1);
    c.logits = conv2d_forward(m->conv2, c.a1);
    return c;
}

static void model_zero_grad(ConvModel *m) {
    tensor_zero_grad(conv2d_tensor(m->conv1));
    tensor_zero_grad(m->conv1->bias);
    tensor_zero_grad(conv2d_tensor(m->conv2));
    tensor_zero_grad(m->conv2->bias);
}

static void model_update(ConvModel *m, float lr) {
    tensor_update(conv2d_tensor(m->conv1), lr);
    tensor_update(m->conv1->bias, lr);
    tensor_update(conv2d_tensor(m->conv2), lr);
    tensor_update(m->conv2->bias, lr);
}

static void free_cache(ForwardCache *c) {
    tensor_free(c->logits);
    tensor_free(c->a1);
    tensor_free(c->h1);
    tensor_free(c->x);
}

void test_conv2d(void) {
    const int total_data = 512;
    const int batch_size = 16;
    const int epochs = 30;
    const float lr = 0.05f;

    float *all_images = (float *)malloc(sizeof(float) * total_data * IMG_SIZE * IMG_SIZE);
    int *all_labels = (int *)malloc(sizeof(int) * total_data);
    float *host_logits = (float *)malloc(sizeof(float) * batch_size * NUM_CLASSES);
    float *grad_output = (float *)malloc(sizeof(float) * batch_size * NUM_CLASSES);
    ConvModel model;
    int epoch;

    if (!all_images || !all_labels || !host_logits || !grad_output) {
        fprintf(stderr, "test_conv2d: failed to allocate buffers.\n");
        free(all_images);
        free(all_labels);
        free(host_logits);
        free(grad_output);
        return;
    }

    srand(42);
    generate_stripe_data(all_images, all_labels, total_data);
    model_init(&model);

    for (epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        int i;

        for (i = 0; i < total_data; i += batch_size) {
            int curr_batch = batch_size;
            float batch_loss = 0.0f;
            int correct = 0;
            int j;
            int x_shape[4];
            Tensor *grad_tensor;
            ForwardCache cache;

            if (i + curr_batch > total_data) curr_batch = total_data - i;

            x_shape[0] = curr_batch;
            x_shape[1] = 1;
            x_shape[2] = IMG_SIZE;
            x_shape[3] = IMG_SIZE;
            cache = model_forward(&model, tensor_from_data(all_images + (size_t)i * IMG_SIZE * IMG_SIZE, 4, x_shape));
            tensor_to_host(cache.logits, host_logits);

            for (j = 0; j < curr_batch; ++j) {
                float probs[NUM_CLASSES];
                int k, label = all_labels[i + j];
                const float *logits = host_logits + j * NUM_CLASSES;

                softmax(logits, probs, NUM_CLASSES);
                batch_loss += -logf(probs[label] + 1e-7f);

                for (k = 0; k < NUM_CLASSES; ++k) {
                    grad_output[j * NUM_CLASSES + k] = probs[k] - ((k == label) ? 1.0f : 0.0f);
                }
                if (argmax(probs, NUM_CLASSES) == label) correct++;
            }

            model_zero_grad(&model);
            {
                int grad_shape[4] = {curr_batch, NUM_CLASSES, 1, 1};
                grad_tensor = tensor_from_data(grad_output, 4, grad_shape);
            }
            tensor_backward(cache.logits, grad_tensor);
            tensor_free(grad_tensor);
            model_update(&model, lr);

            epoch_loss += batch_loss;
            epoch_correct += correct;

            free_cache(&cache);
        }

        printf("=== [conv2d] Epoch %2d Summary: Avg Loss=%.4f, Accuracy=%.4f ===\n", epoch,
               epoch_loss / (float)total_data, (float)epoch_correct / (float)total_data);
    }

    model_free(&model);
    free(all_images);
    free(all_labels);
    free(host_logits);
    free(grad_output);
}
