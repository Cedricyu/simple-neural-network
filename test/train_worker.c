/* MNIST ConvNet trainer, written to be the CUDA half of the live-training
 * viewer. It does NOT link SDL2.
 *
 * Why a separate process: an earlier single-process version linked SDL2
 * directly into this training binary, and segfaulted inside NVIDIA's WSL
 * driver (confirmed with AddressSanitizer: a double-free inside cuInit()'s
 * adapter enumeration, dxgdmalEnumAdapters). Reordering the SDL/CUDA init
 * calls didn't help, because the dynamic linker loads every linked shared
 * library (SDL2 and its transitive GL/X11/Wayland dependencies included)
 * before main() ever runs - by the time any of our own code executes, SDL2
 * is already resident in the process regardless of call order, and that's
 * what the driver's adapter enumeration trips on under WSLg. Keeping CUDA
 * and SDL2 in separate processes avoids the conflict entirely. This process
 * does the training and streams progress over a FIFO; train_viewer.c (no
 * CUDA, SDL2 only) reads the FIFO and draws the window. `make train_gui`
 * runs both together. */
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "conv2d.cuh"
#include "linear.cuh"
#include "mnist.h"
#include "tensor.cuh"
#include "tensorops.cuh"
#include "train_record.h"

#define NUM_CLASSES 10
#define FC1_IN (32 * 7 * 7) /* conv4's output flattened: 32 channels, 7x7 spatial */
#define FC1_OUT 128

typedef struct MnistModel {
    Conv2D *conv1;
    Conv2D *conv2;
    Conv2D *conv3;
    Conv2D *conv4;
    Linear *fc1;
    Linear *fc2;
} MnistModel;

typedef struct ForwardCache {
    Tensor *x;
    Tensor *h1, *a1;
    Tensor *h2, *a2;
    Tensor *h3, *a3;
    Tensor *h4, *a4;
    Tensor *flat;
    Tensor *h5, *a5;
    Tensor *logits;
} ForwardCache;

static void softmax(const float *logits, float *output, int size) {
    float max_logit = logits[0];
    int i;
    for (i = 1; i < size; i++) if (logits[i] > max_logit) max_logit = logits[i];
    float sum_exp = 0.0f;
    for (i = 0; i < size; i++) { output[i] = expf(logits[i] - max_logit); sum_exp += output[i]; }
    for (i = 0; i < size; i++) output[i] /= sum_exp;
}

static int argmax(const float *arr, int size) {
    int i, max_idx = 0;
    for (i = 1; i < size; i++) if (arr[i] > arr[max_idx]) max_idx = i;
    return max_idx;
}

/* conv2-4 kernels are (out, in, k, k) with in > 1, so there's no single
 * in-channel slice that reads as "the filter" the way conv1's does. Instead,
 * for each output channel, average |weight| over the input-channel axis -
 * `out[o][ky][kx] = mean_over_in(|raw[o][in][ky][kx]|)` - collapsing the
 * channel mixing into one same-size-as-the-kernel map of "how much this
 * output channel weighs each kernel position overall". `raw` must already
 * hold c_out*c_in*k*k host floats (see tensor_to_host in the caller). */
static void reduce_filter_magnitude(const float *raw, int c_out, int c_in, int k, float *out) {
    int kk = k * k;
    for (int o = 0; o < c_out; ++o) {
        for (int p = 0; p < kk; ++p) {
            float sum = 0.0f;
            for (int ci = 0; ci < c_in; ++ci) sum += fabsf(raw[(size_t)(o * c_in + ci) * kk + p]);
            out[o * kk + p] = sum / c_in;
        }
    }
}

/* fc1's weight is (in_f, out_f) row-major - average-pools every `pool`
 * consecutive input rows into one, shrinking (1568,128) to (98,128) so it's
 * both cheap to stream every batch and small enough to read as a heatmap.
 * Unlike reduce_filter_magnitude this doesn't take |.|: a weight matrix's
 * sign is part of what's interesting to look at (excitatory vs inhibitory),
 * there's no channel-mixing to collapse away here. */
static void reduce_fc1_viz(const float *raw, int in_f, int out_f, int pool, float *out) {
    int viz_rows = in_f / pool;
    for (int r = 0; r < viz_rows; ++r) {
        for (int c = 0; c < out_f; ++c) {
            float sum = 0.0f;
            for (int p = 0; p < pool; ++p) sum += raw[(size_t)(r * pool + p) * out_f + c];
            out[r * out_f + c] = sum / pool;
        }
    }
}

/* conv4 is extra depth: a same-resolution (7x7->7x7, stride 1) layer
 * inserted after the last downsampling conv, so the net gets one more
 * nonlinearity/capacity bump without touching the downsampling plan. Then,
 * instead of the old trick of using one more conv with a kernel exactly as
 * big as the remaining spatial size to fake a fully-connected layer, this
 * flattens conv4's output and feeds a real two-layer MLP head
 * (fc1 1568->128, relu, fc2 128->10) - what should have been there from the
 * start; the conv-as-FC trick was only ever a workaround for not having a
 * flatten op yet (see tensor_flatten in tensorops.cu). */
static void model_init(MnistModel *m) {
    m->conv1 = conv2d_new(1, 8, 3, 1, 1, 1);
    m->conv2 = conv2d_new(8, 16, 3, 2, 1, 1);
    m->conv3 = conv2d_new(16, 32, 3, 2, 1, 1);
    m->conv4 = conv2d_new(32, 32, 3, 1, 1, 1);
    m->fc1 = linear_new(FC1_IN, FC1_OUT);
    m->fc2 = linear_new(FC1_OUT, NUM_CLASSES);
}

static void model_free(MnistModel *m) {
    conv2d_free(m->conv1); conv2d_free(m->conv2); conv2d_free(m->conv3); conv2d_free(m->conv4);
    linear_free(m->fc1); linear_free(m->fc2);
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
    c.h4 = conv2d_forward(m->conv4, c.a3);
    c.a4 = tensor_relu(c.h4);
    c.flat = tensor_flatten(c.a4);
    c.h5 = linear_forward(m->fc1, c.flat);
    c.a5 = tensor_relu(c.h5);
    c.logits = linear_forward(m->fc2, c.a5);
    return c;
}

static void model_zero_grad(MnistModel *m) {
    tensor_zero_grad(conv2d_tensor(m->conv1)); tensor_zero_grad(m->conv1->bias);
    tensor_zero_grad(conv2d_tensor(m->conv2)); tensor_zero_grad(m->conv2->bias);
    tensor_zero_grad(conv2d_tensor(m->conv3)); tensor_zero_grad(m->conv3->bias);
    tensor_zero_grad(conv2d_tensor(m->conv4)); tensor_zero_grad(m->conv4->bias);
    tensor_zero_grad(linear_tensor(m->fc1)); tensor_zero_grad(m->fc1->bias);
    tensor_zero_grad(linear_tensor(m->fc2)); tensor_zero_grad(m->fc2->bias);
}

static void model_update(MnistModel *m, float lr) {
    tensor_update(conv2d_tensor(m->conv1), lr); tensor_update(m->conv1->bias, lr);
    tensor_update(conv2d_tensor(m->conv2), lr); tensor_update(m->conv2->bias, lr);
    tensor_update(conv2d_tensor(m->conv3), lr); tensor_update(m->conv3->bias, lr);
    tensor_update(conv2d_tensor(m->conv4), lr); tensor_update(m->conv4->bias, lr);
    tensor_update(linear_tensor(m->fc1), lr); tensor_update(m->fc1->bias, lr);
    tensor_update(linear_tensor(m->fc2), lr); tensor_update(m->fc2->bias, lr);
}

static void free_cache(ForwardCache *c) {
    tensor_free(c->logits); tensor_free(c->a5); tensor_free(c->h5); tensor_free(c->flat);
    tensor_free(c->a4); tensor_free(c->h4);
    tensor_free(c->a3); tensor_free(c->h3);
    tensor_free(c->a2); tensor_free(c->h2); tensor_free(c->a1); tensor_free(c->h1); tensor_free(c->x);
}

/* Reads (and discards) every byte currently queued on a non-blocking
 * `cmd_fd` without blocking - called right after deciding to (re)start, so a
 * user's rapid double-click doesn't leave a second queued byte that the new
 * run's very first per-batch poll would see and misread as an immediate,
 * confusing restart. */
static void drain_pending_commands(int cmd_fd) {
    unsigned char b;
    while (read(cmd_fd, &b, 1) > 0) { /* discard */ }
}

int main(void) {
    const int batch_size = 64;
    const int epochs = 3;
    const float lr = 0.001f;

    MnistData train;
    FILE *fifo;
    int cmd_fd;

    /* Without this, writing a batch's record to the data FIFO after the
     * viewer has exited (closing its read end) raises SIGPIPE, whose default
     * disposition kills the process outright - before the existing
     * `fwrite(...) != 1` check below ever gets a chance to catch it and shut
     * down cleanly. Ignoring it makes write() return -1/EPIPE instead, which
     * fwrite() surfaces as its documented short-item-count failure. */
    signal(SIGPIPE, SIG_IGN);

    if (mnist_load("data/fashion-mnist/train-images-idx3-ubyte", "data/fashion-mnist/train-labels-idx1-ubyte",
                   &train) != 0) {
        fprintf(stderr,
                "train_worker: could not load Fashion-MNIST training set (expected files under "
                "data/fashion-mnist/).\n");
        fprintf(stderr, "See README.md 'Downloading Fashion-MNIST' for the curl commands.\n");
        return 1;
    }

    if (mkfifo(TRAIN_FIFO_PATH, 0600) != 0 && errno != EEXIST) {
        fprintf(stderr, "train_worker: mkfifo(%s) failed: %s\n", TRAIN_FIFO_PATH, strerror(errno));
        mnist_free(&train);
        return 1;
    }
    fprintf(stderr, "train_worker: waiting for train_viewer to connect to %s ...\n", TRAIN_FIFO_PATH);
    fifo = fopen(TRAIN_FIFO_PATH, "wb"); /* blocks until a reader opens the other end */
    if (!fifo) {
        fprintf(stderr, "train_worker: could not open fifo for writing: %s\n", strerror(errno));
        mnist_free(&train);
        return 1;
    }

    /* Second FIFO, viewer->worker - see the comment on TRAIN_CMD_FIFO_PATH in
     * train_record.h. open() here blocks until the viewer opens its write
     * end (same handshake pattern as the data FIFO above), and the first
     * read() in the loop below then blocks until the user actually clicks
     * Start - this fd only becomes O_NONBLOCK once a training run begins, so
     * the per-batch poll for a Restart never stalls the training loop. */
    if (mkfifo(TRAIN_CMD_FIFO_PATH, 0600) != 0 && errno != EEXIST) {
        fprintf(stderr, "train_worker: mkfifo(%s) failed: %s\n", TRAIN_CMD_FIFO_PATH, strerror(errno));
        fclose(fifo);
        mnist_free(&train);
        return 1;
    }
    cmd_fd = open(TRAIN_CMD_FIFO_PATH, O_RDONLY);
    if (cmd_fd < 0) {
        fprintf(stderr, "train_worker: could not open command fifo for reading: %s\n", strerror(errno));
        fclose(fifo);
        mnist_free(&train);
        return 1;
    }
    fprintf(stderr, "train_worker: viewer connected. Waiting for Start...\n");

    float *host_logits = (float *)malloc(sizeof(float) * batch_size * NUM_CLASSES);
    float *grad_output = (float *)malloc(sizeof(float) * batch_size * NUM_CLASSES);
    /* Reused scratch for pulling conv2-4's full (out,in,k,k) weights and
     * fc1's full (in,out) weight matrix to host before reduce_filter_magnitude()
     * / reduce_fc1_viz() collapse them; sized for the largest consumer
     * (fc1: 1568*128, bigger than any conv layer here). */
    float *raw_scratch = (float *)malloc(sizeof(float) * FC1_IN * FC1_OUT);

    srand(42);

    /* Outer loop: each pass is one full (re)start. A restart clicked mid-run
     * sets have_pending_start so the next pass skips straight back into
     * training instead of blocking on another read() for a command that was
     * already consumed. */
    int have_pending_start = 0;
    for (;;) {
        if (!have_pending_start) {
            unsigned char cmd;
            ssize_t n = read(cmd_fd, &cmd, 1); /* blocking: cmd_fd is back in blocking mode by the end of each pass */
            if (n <= 0) break;                 /* viewer closed the command fifo - nothing left to serve */
            if (cmd != TRAIN_CMD_START) continue;
        }
        have_pending_start = 0;

        int flags = fcntl(cmd_fd, F_GETFL, 0);
        fcntl(cmd_fd, F_SETFL, flags | O_NONBLOCK);
        drain_pending_commands(cmd_fd);

        fprintf(stderr, "train_worker: training started.\n");
        MnistModel model;
        model_init(&model);
        clock_t t_start = clock();
        int interrupt = 0; /* 0 = ran to completion, 1 = restart requested, 2 = viewer gone */

        for (int epoch = 0; epoch < epochs && !interrupt; ++epoch) {
            int epoch_correct = 0;
            float epoch_loss = 0.0f;

            for (int i = 0; i < train.count && !interrupt; i += batch_size) {
                int curr_batch = batch_size;
                float batch_loss = 0.0f;
                int batch_correct = 0;
                int x_shape[4];
                Tensor *grad_tensor;
                ForwardCache cache;

                if (i + curr_batch > train.count) curr_batch = train.count - i;

                x_shape[0] = curr_batch; x_shape[1] = 1; x_shape[2] = train.rows; x_shape[3] = train.cols;
                cache = model_forward(&model, tensor_from_data(train.images + (size_t)i * train.rows * train.cols, 4, x_shape));
                tensor_to_host(cache.logits, host_logits);

                int sample_pred = 0;
                for (int j = 0; j < curr_batch; ++j) {
                    float probs[NUM_CLASSES];
                    int k, label = train.labels[i + j];
                    softmax(host_logits + j * NUM_CLASSES, probs, NUM_CLASSES);
                    batch_loss += -logf(probs[label] + 1e-7f);
                    for (k = 0; k < NUM_CLASSES; ++k) grad_output[j * NUM_CLASSES + k] = probs[k] - ((k == label) ? 1.0f : 0.0f);
                    if (argmax(probs, NUM_CLASSES) == label) batch_correct++;
                    if (j == 0) sample_pred = argmax(probs, NUM_CLASSES);
                }

                model_zero_grad(&model);
                { int grad_shape[4] = {curr_batch, NUM_CLASSES, 1, 1}; grad_tensor = tensor_from_data(grad_output, 4, grad_shape); }
                tensor_backward(cache.logits, grad_tensor);
                tensor_free(grad_tensor);
                model_update(&model, lr);

                epoch_loss += batch_loss;
                epoch_correct += batch_correct;

                TrainRecord rec;
                rec.epoch = epoch;
                rec.epochs = epochs;
                rec.batch = i / batch_size;
                rec.batch_count = (train.count + batch_size - 1) / batch_size;
                rec.loss = batch_loss / curr_batch;
                rec.acc = (float)batch_correct / curr_batch;
                rec.true_label = train.labels[i];
                rec.pred_label = sample_pred;
                rec.elapsed = (float)(clock() - t_start) / CLOCKS_PER_SEC;
                for (int p = 0; p < TRAIN_IMG_SIZE * TRAIN_IMG_SIZE; ++p)
                    rec.pixels[p] = (unsigned char)(train.images[(size_t)i * train.rows * train.cols + p] * 255.0f);
                tensor_to_host(conv2d_tensor(model.conv1), rec.conv1_weights);
                tensor_to_host(conv2d_tensor(model.conv2), raw_scratch);
                reduce_filter_magnitude(raw_scratch, CONV2_OUT, CONV2_IN, CONV2_K, rec.conv2_filters);
                tensor_to_host(conv2d_tensor(model.conv3), raw_scratch);
                reduce_filter_magnitude(raw_scratch, CONV3_OUT, CONV3_IN, CONV3_K, rec.conv3_filters);
                tensor_to_host(conv2d_tensor(model.conv4), raw_scratch);
                reduce_filter_magnitude(raw_scratch, CONV4_OUT, CONV4_IN, CONV4_K, rec.conv4_filters);
                tensor_to_host(linear_tensor(model.fc1), raw_scratch);
                reduce_fc1_viz(raw_scratch, FC1_IN, FC1_OUT, FC1_POOL, rec.fc1_viz);
                tensor_to_host(linear_tensor(model.fc2), rec.fc2_weights);

                if (fwrite(&rec, sizeof(rec), 1, fifo) != 1) {
                    fprintf(stderr, "train_worker: viewer disconnected, stopping.\n");
                    free_cache(&cache);
                    interrupt = 2;
                    break;
                }
                fflush(fifo);
                free_cache(&cache);

                unsigned char poll_cmd;
                ssize_t pn = read(cmd_fd, &poll_cmd, 1);
                if (pn == 1) interrupt = 1;        /* restart requested mid-run */
                else if (pn == 0) interrupt = 2;   /* viewer closed the command fifo */
                /* pn < 0 (EAGAIN/EWOULDBLOCK): nothing pending, keep training */
            }

            if (!interrupt) {
                printf("=== epoch %d done: avg loss %.4f, acc %.4f ===\n", epoch, epoch_loss / train.count,
                       (float)epoch_correct / train.count);
            }
        }

        model_free(&model);
        fcntl(cmd_fd, F_SETFL, flags); /* back to blocking for the next outer-loop wait */

        if (interrupt == 2) break; /* viewer gone - nothing left to serve */
        if (interrupt == 1) {
            have_pending_start = 1; /* go again immediately, the restart byte already arrived */
        } else {
            fprintf(stderr, "train_worker: training finished. Waiting for Restart...\n");
        }
    }

    fclose(fifo);
    close(cmd_fd);
    mnist_free(&train);
    free(host_logits);
    free(grad_output);
    free(raw_scratch);
    return 0;
}
