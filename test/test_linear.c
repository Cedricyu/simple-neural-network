#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "linear.cuh"
#include "tensor.cuh"
#include "tensorops.cuh"
#include "test.h"
#include "dataset.h"
#include "train_config.h"

typedef struct ForwardCache {
    Tensor *x;
    Tensor *h1;
    Tensor *a1;
    Tensor *h2;
    Tensor *a2;
    Tensor *h3;
    Tensor *a3;
    Tensor *y;
} ForwardCache;

typedef struct Model {
    Linear *linear1;
    Linear *linear2;
    Linear *linear3;
    Linear *linear4;
} Model;

static void softmax(const float *logits, float *output, int size) {
    float max_logit = logits[0];
    int i;
    for (i = 1; i < size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    {
        float sum_exp = 0.0f;
        for (i = 0; i < size; i++) {
            output[i] = expf(logits[i] - max_logit);
            sum_exp += output[i];
        }
        for (i = 0; i < size; i++) output[i] /= sum_exp;
    }
}

static int argmax(const float *arr, int size) {
    int i;
    int max_idx = 0;
    for (i = 1; i < size; i++) {
        if (arr[i] > arr[max_idx]) max_idx = i;
    }
    return max_idx;
}

static void model_init(Model *m, int input_dim, int h1, int h2, int h3, int output_dim) {
    m->linear1 = linear_new(input_dim, h1);
    m->linear2 = linear_new(h1, h2);
    m->linear3 = linear_new(h2, h3);
    m->linear4 = linear_new(h3, output_dim);
}

static void model_free(Model *m) {
    linear_free(m->linear1);
    linear_free(m->linear2);
    linear_free(m->linear3);
    linear_free(m->linear4);
}

static ForwardCache model_forward(Model *m, Tensor *x) {
    ForwardCache c;
    c.x = x;
    c.h1 = linear_forward(m->linear1, x);
    c.a1 = tensor_relu(c.h1);
    c.h2 = linear_forward(m->linear2, c.a1);
    c.a2 = tensor_relu(c.h2);
    c.h3 = linear_forward(m->linear3, c.a2);
    c.a3 = tensor_relu(c.h3);
    c.y = linear_forward(m->linear4, c.a3);
    return c;
}

static void model_zero_grad(Model *m) {
    tensor_zero_grad(linear_tensor(m->linear1));
    tensor_zero_grad(linear_tensor(m->linear2));
    tensor_zero_grad(linear_tensor(m->linear3));
    tensor_zero_grad(linear_tensor(m->linear4));

    tensor_zero_grad(m->linear1->bias);
    tensor_zero_grad(m->linear2->bias);
    tensor_zero_grad(m->linear3->bias);
    tensor_zero_grad(m->linear4->bias);
}

static void model_backward(Model *m, Tensor *output, float *grad_output, int batch_size, int output_dim) {
    int shape[2] = {batch_size, output_dim};
    Tensor *grad = tensor_from_data(grad_output, 2, shape);
    tensor_backward(output, grad);
    tensor_free(grad);
    (void)m;
}

static void model_update(Model *m, float lr) {
    tensor_update(linear_tensor(m->linear1), lr);
    tensor_update(linear_tensor(m->linear2), lr);
    tensor_update(linear_tensor(m->linear3), lr);
    tensor_update(linear_tensor(m->linear4), lr);

    tensor_update(m->linear1->bias, lr);
    tensor_update(m->linear2->bias, lr);
    tensor_update(m->linear3->bias, lr);
    tensor_update(m->linear4->bias, lr);
}

static void free_cache(ForwardCache *c) {
    tensor_free(c->y);
    tensor_free(c->a3);
    tensor_free(c->h3);
    tensor_free(c->a2);
    tensor_free(c->h2);
    tensor_free(c->a1);
    tensor_free(c->h1);
    tensor_free(c->x);
}

void test_linear(void) {
    TrainingConfig cfg = training_config_default();
    float *all_inputs = (float *)malloc(sizeof(float) * cfg.total_data * cfg.input_dim);
    int *all_targets = (int *)malloc(sizeof(int) * cfg.total_data);
    float *host_logits = (float *)malloc(sizeof(float) * cfg.batch_size * cfg.output_dim);
    float *grad_output = (float *)malloc(sizeof(float) * cfg.batch_size * cfg.output_dim);
    Model model;
    int epoch;

    if (!all_inputs || !all_targets || !host_logits || !grad_output) {
        fprintf(stderr, "Failed to allocate training buffers.\n");
        free(all_inputs);
        free(all_targets);
        free(host_logits);
        free(grad_output);
        return;
    }

    srand(cfg.seed);
    generate_checkerboard_data(all_inputs, all_targets, cfg.total_data, cfg.num_squares);
    model_init(&model, cfg.input_dim, cfg.hidden_dim1, cfg.hidden_dim2, cfg.hidden_dim3, cfg.output_dim);

    for (epoch = 0; epoch < cfg.epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        int i;

        for (i = 0; i < cfg.total_data; i += cfg.batch_size) {
            int curr_batch = cfg.batch_size;
            float *inputs;
            ForwardCache cache;
            float batch_loss = 0.0f;
            int correct = 0;
            int j;

            if (i + curr_batch > cfg.total_data) curr_batch = cfg.total_data - i;
            inputs = (float *)malloc(sizeof(float) * curr_batch * cfg.input_dim);
            if (!inputs) {
                fprintf(stderr, "Failed to allocate batch buffers.\n");
                free(inputs);
                model_free(&model);
                free(all_inputs);
                free(all_targets);
                free(host_logits);
                free(grad_output);
                return;
            }

            for (j = 0; j < curr_batch; ++j) {
                int idx = i + j;
                inputs[j * cfg.input_dim + 0] = all_inputs[idx * cfg.input_dim + 0];
                inputs[j * cfg.input_dim + 1] = all_inputs[idx * cfg.input_dim + 1];
            }

            cache = model_forward(&model, tensor_from_data_2d(inputs, curr_batch, cfg.input_dim));
            tensor_to_host(cache.y, host_logits);
            free(inputs);

            for (j = 0; j < curr_batch; ++j) {
                float probs[cfg.output_dim];
                int k;
                int label = all_targets[i + j];
                const float *logits = host_logits + j * cfg.output_dim;

                softmax(logits, probs, cfg.output_dim);
                batch_loss += -logf(probs[label] + 1e-7f);

                for (k = 0; k < cfg.output_dim; ++k) {
                    grad_output[j * cfg.output_dim + k] = probs[k] - ((k == label) ? 1.0f : 0.0f);
                }

                if (argmax(probs, cfg.output_dim) == label) correct++;
            }

            model_zero_grad(&model);
            model_backward(&model, cache.y, grad_output, curr_batch, cfg.output_dim);
            model_update(&model, cfg.learning_rate);

            epoch_loss += batch_loss;
            epoch_correct += correct;

            free_cache(&cache);
        }

        printf("=== Epoch %d Summary: Avg Loss=%.4f, Accuracy=%.4f ===\n", epoch,
               epoch_loss / (float)cfg.total_data, (float)epoch_correct / (float)cfg.total_data);
    }

    model_free(&model);
    free(all_inputs);
    free(all_targets);
    free(host_logits);
    free(grad_output);
}
