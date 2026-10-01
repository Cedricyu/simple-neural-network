#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "conv2d.cuh"
#include "mnist.h"
#include "tensor.cuh"
#include "tensorops.cuh"
#include "test.h"

#define IMG_SIZE 28
#define NUM_CLASSES 10

typedef struct MnistModel {
    Conv2D *conv1; /* 1  -> 8,  3x3 stride 1 pad 1 : 28x28 -> 28x28 */
    Conv2D *conv2; /* 8  -> 16, 3x3 stride 2 pad 1 : 28x28 -> 14x14 */
    Conv2D *conv3; /* 16 -> 32, 3x3 stride 2 pad 1 : 14x14 -> 7x7   */
    Conv2D *conv4; /* 32 -> 10, 7x7 stride 1 pad 0 : 7x7   -> 1x1  */
} MnistModel;

typedef struct ForwardCache {
    Tensor *x;
    Tensor *h1, *a1;
    Tensor *h2, *a2;
    Tensor *h3, *a3;
    Tensor *logits; /* (B, NUM_CLASSES, 1, 1) */
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

static void model_init(MnistModel *m) {
    m->conv1 = conv2d_new(1, 8, 3, 1, 1, 1);
    m->conv2 = conv2d_new(8, 16, 3, 2, 1, 1);
    m->conv3 = conv2d_new(16, 32, 3, 2, 1, 1);
    m->conv4 = conv2d_new(32, NUM_CLASSES, 7, 1, 0, 1);
}

static void model_free(MnistModel *m) {
    conv2d_free(m->conv1);
    conv2d_free(m->conv2);
    conv2d_free(m->conv3);
    conv2d_free(m->conv4);
}

static ForwardCache model_forward(MnistModel *m, Tensor *x) {
    ForwardCache c;
    c.x = x;
    c.h1 = conv2d_forward(m->conv1, x);
    c.a1 = tensor_relu(c.h1);
    c.h2 = conv2d_forward(m->conv2, c.a1);
    c.a2 = tensor_relu(c.h2);
    c.h3 = conv2d_forward(m->conv3, c.a2);
    c.a3 = tensor_relu(c.h3);
    c.logits = conv2d_forward(m->conv4, c.a3);
    return c;
}

static void model_zero_grad(MnistModel *m) {
    tensor_zero_grad(conv2d_tensor(m->conv1));
    tensor_zero_grad(m->conv1->bias);
    tensor_zero_grad(conv2d_tensor(m->conv2));
    tensor_zero_grad(m->conv2->bias);
    tensor_zero_grad(conv2d_tensor(m->conv3));
    tensor_zero_grad(m->conv3->bias);
    tensor_zero_grad(conv2d_tensor(m->conv4));
    tensor_zero_grad(m->conv4->bias);
}

static void model_update(MnistModel *m, float lr) {
    tensor_update(conv2d_tensor(m->conv1), lr);
    tensor_update(m->conv1->bias, lr);
    tensor_update(conv2d_tensor(m->conv2), lr);
    tensor_update(m->conv2->bias, lr);
    tensor_update(conv2d_tensor(m->conv3), lr);
    tensor_update(m->conv3->bias, lr);
    tensor_update(conv2d_tensor(m->conv4), lr);
    tensor_update(m->conv4->bias, lr);
}

static void free_cache(ForwardCache *c) {
    tensor_free(c->logits);
    tensor_free(c->a3);
    tensor_free(c->h3);
    tensor_free(c->a2);
    tensor_free(c->h2);
    tensor_free(c->a1);
    tensor_free(c->h1);
    tensor_free(c->x);
}

/* Forward-only pass over `data`, no autograd bookkeeping, for test-set accuracy. */
static float evaluate(MnistModel *m, const MnistData *data, int limit, int batch_size) {
    int n = (limit > 0 && limit < data->count) ? limit : data->count;
    int correct = 0;
    int i;
    float *host_logits = (float *)malloc(sizeof(float) * batch_size * NUM_CLASSES);

    for (i = 0; i < n; i += batch_size) {
        int curr_batch = batch_size;
        int x_shape[4];
        ForwardCache cache;
        int j;

        if (i + curr_batch > n) curr_batch = n - i;
        x_shape[0] = curr_batch;
        x_shape[1] = 1;
        x_shape[2] = data->rows;
        x_shape[3] = data->cols;

        cache = model_forward(
            m, tensor_from_data(data->images + (size_t)i * data->rows * data->cols, 4, x_shape));
        tensor_to_host(cache.logits, host_logits);

        for (j = 0; j < curr_batch; ++j) {
            float probs[NUM_CLASSES];
            softmax(host_logits + j * NUM_CLASSES, probs, NUM_CLASSES);
            if (argmax(probs, NUM_CLASSES) == data->labels[i + j]) correct++;
        }

        free_cache(&cache);
    }

    free(host_logits);
    return (float)correct / (float)n;
}

void test_mnist(void) {
    const char *train_images_path = "data/mnist/train-images-idx3-ubyte";
    const char *train_labels_path = "data/mnist/train-labels-idx1-ubyte";
    const char *test_images_path = "data/mnist/t10k-images-idx3-ubyte";
    const char *test_labels_path = "data/mnist/t10k-labels-idx1-ubyte";

    const int batch_size = 64;
    const int epochs = 3;
    const float lr = 0.001f;

    MnistData train, test;
    MnistModel model;
    float *host_logits;
    float *grad_output;
    int epoch;

    if (mnist_load(train_images_path, train_labels_path, &train) != 0) {
        fprintf(stderr, "test_mnist: could not load training set (expected files under data/mnist/), skipping.\n");
        return;
    }
    if (mnist_load(test_images_path, test_labels_path, &test) != 0) {
        fprintf(stderr, "test_mnist: could not load test set, skipping.\n");
        mnist_free(&train);
        return;
    }

    printf("test_mnist: loaded %d train / %d test images (%dx%d)\n", train.count, test.count, train.rows,
           train.cols);

    host_logits = (float *)malloc(sizeof(float) * batch_size * NUM_CLASSES);
    grad_output = (float *)malloc(sizeof(float) * batch_size * NUM_CLASSES);

    srand((unsigned int)time(NULL));
    model_init(&model);

    for (epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        int i;
        clock_t t_start = clock();

        for (i = 0; i < train.count; i += batch_size) {
            int curr_batch = batch_size;
            float batch_loss = 0.0f;
            int correct = 0;
            int j;
            int x_shape[4];
            Tensor *grad_tensor;
            ForwardCache cache;

            if (i + curr_batch > train.count) curr_batch = train.count - i;

            x_shape[0] = curr_batch;
            x_shape[1] = 1;
            x_shape[2] = train.rows;
            x_shape[3] = train.cols;
            cache = model_forward(
                &model, tensor_from_data(train.images + (size_t)i * train.rows * train.cols, 4, x_shape));
            tensor_to_host(cache.logits, host_logits);

            for (j = 0; j < curr_batch; ++j) {
                float probs[NUM_CLASSES];
                int k, label = train.labels[i + j];
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

            if ((i / batch_size) % 50 == 0) {
                printf("  [mnist] epoch %d batch %d/%d  batch_loss=%.4f batch_acc=%.3f\n", epoch, i / batch_size,
                       (train.count + batch_size - 1) / batch_size, batch_loss / curr_batch,
                       (float)correct / curr_batch);
            }
        }

        {
            double secs = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            float test_acc = evaluate(&model, &test, 2000, batch_size);
            printf("=== [mnist] Epoch %d: Train Loss=%.4f Train Acc=%.4f | Test Acc=%.4f (%.1fs) ===\n", epoch,
                   epoch_loss / (float)train.count, (float)epoch_correct / (float)train.count, test_acc, secs);
        }
    }

    model_free(&model);
    mnist_free(&train);
    mnist_free(&test);
    free(host_logits);
    free(grad_output);
}
