/* ConvNet trainer, written to be the CUDA half of the live-training viewer.
 * It does NOT link SDL2. Every training parameter (dataset, epochs, lr, the
 * conv/fc layer stack, ...) arrives from the viewer's setup form as a
 * TrainConfig over the command FIFO - nothing about the model is hardcoded.
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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "batchnorm.cuh"
#include "conv2d.cuh"
#include "linear.cuh"
#include "mnist.h"
#include "pool.cuh"
#include "tensor.cuh"
#include "tensorops.cuh"
#include "train_record.h"

/* fc weights up to this many floats are streamed in full; bigger ones are
 * average-pooled over input rows (TrainConfig::fc_pool) so a heatmap update
 * stays cheap to send every batch. FC_VIZ_MAX_FLOATS caps the pooled result
 * no matter what fc_pool was asked for. */
#define FC_FULL_DETAIL_MAX 4096
#define FC_VIZ_MAX_FLOATS 65536

/* One entry per block; conv[i] / lin[i] / bn[i] are only set for the blocks of that type. */
typedef struct Net {
    int n_blocks;
    int has_bn;
    Block blocks[MAX_BLOCKS];
    Conv2D *conv[MAX_BLOCKS];
    Linear *lin[MAX_BLOCKS];
    BatchNorm *bn[MAX_BLOCKS];
} Net;

/* Every tensor created by one forward pass, in creation order, so freeing in
 * reverse order releases views (flatten) before the tensors they alias. */
#define MAX_CACHE (MAX_BLOCKS + 2)
typedef struct ForwardCache {
    Tensor *owned[MAX_CACHE];
    int n;
    Tensor *logits;
} ForwardCache;

static Tensor *cache_add(ForwardCache *c, Tensor *t) {
    c->owned[c->n++] = t;
    return t;
}

static void free_cache(ForwardCache *c) {
    for (int i = c->n - 1; i >= 0; --i) tensor_free(c->owned[i]);
    c->n = 0;
}

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

static void net_init(Net *n, const TrainConfig *c, const NetShapes *s) {
    memset(n, 0, sizeof(*n));
    n->n_blocks = c->n_blocks;
    for (int i = 0; i < c->n_blocks; ++i) {
        const Block *b = &c->blocks[i];
        LayerShape in = train_block_input(s, b);
        n->blocks[i] = *b;
        if (b->type == BLK_CONV) n->conv[i] = conv2d_new(in.c, b->a, b->b, b->c, b->d, 1);
        else if (b->type == BLK_LINEAR) n->lin[i] = linear_new(in.c, b->a);
        else if (b->type == BLK_BATCHNORM) {
            n->bn[i] = batchnorm_new(in.c);
            n->has_bn = 1;
        }
    }
}

static void net_free(Net *n) {
    for (int i = 0; i < n->n_blocks; ++i) {
        if (n->conv[i]) conv2d_free(n->conv[i]);
        if (n->lin[i]) linear_free(n->lin[i]);
        if (n->bn[i]) batchnorm_free(n->bn[i]);
    }
}

/* Evaluates the block graph in order (every block only reads earlier tensors) and leaves the last
 * tensor in c->logits. Takes ownership of x via the cache. `training` selects batch statistics
 * (1) or running statistics (0) in batch-norm blocks. */
static void net_forward(Net *n, Tensor *x, ForwardCache *c, int training) {
    Tensor *val[MAX_BLOCKS + 1];
    c->n = 0;
    val[0] = cache_add(c, x);
    for (int i = 0; i < n->n_blocks; ++i) {
        const Block *b = &n->blocks[i];
        Tensor *in = val[b->in[0]], *out = NULL;
        switch (b->type) {
        case BLK_CONV: out = conv2d_forward(n->conv[i], in); break;
        case BLK_RELU: out = tensor_relu(in); break;
        case BLK_FLATTEN: out = tensor_flatten(in); break;
        case BLK_LINEAR: out = linear_forward(n->lin[i], in); break;
        case BLK_ADD: out = tensor_add(in, val[b->in[1]]); break;
        case BLK_BATCHNORM: out = batchnorm_forward(n->bn[i], in, training); break;
        case BLK_MAXPOOL: out = maxpool2d_forward(in, b->a, b->b); break;
        case BLK_GAP: out = global_avg_pool_forward(in); break;
        }
        if (!out) {
            /* train_cfg_derive() accepted these shapes, so this is a bug, not bad input. */
            fprintf(stderr, "train_worker: block %d (%s) failed to run\n", i + 1, train_block_type_name(b->type));
            exit(1);
        }
        val[i + 1] = cache_add(c, out);
    }
    c->logits = val[n->n_blocks];
}

static void net_zero_grad(Net *n) {
    for (int i = 0; i < n->n_blocks; ++i) {
        if (n->conv[i]) {
            tensor_zero_grad(conv2d_tensor(n->conv[i]));
            tensor_zero_grad(n->conv[i]->bias);
        }
        if (n->lin[i]) {
            tensor_zero_grad(linear_tensor(n->lin[i]));
            tensor_zero_grad(n->lin[i]->bias);
        }
        if (n->bn[i]) {
            tensor_zero_grad(n->bn[i]->gamma);
            tensor_zero_grad(n->bn[i]->beta);
        }
    }
}

static void net_update(Net *n, float lr, float momentum) {
    for (int i = 0; i < n->n_blocks; ++i) {
        if (n->conv[i]) {
            tensor_update_momentum(conv2d_tensor(n->conv[i]), lr, momentum);
            tensor_update_momentum(n->conv[i]->bias, lr, momentum);
        }
        if (n->lin[i]) {
            tensor_update_momentum(linear_tensor(n->lin[i]), lr, momentum);
            tensor_update_momentum(n->lin[i]->bias, lr, momentum);
        }
        if (n->bn[i]) {
            tensor_update_momentum(n->bn[i]->gamma, lr, momentum);
            tensor_update_momentum(n->bn[i]->beta, lr, momentum);
        }
    }
}

/* conv kernels with several input channels have no single in-channel slice
 * that reads as "the filter" the way a 1-channel first layer's does. Instead,
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

/* An fc weight is (in_f, out_f) row-major - average-pools every `pool`
 * consecutive input rows into one. Unlike reduce_filter_magnitude this
 * doesn't take |.|: a weight matrix's sign is part of what's interesting to
 * look at (excitatory vs inhibitory), there's no channel-mixing to collapse
 * away here. Trailing rows that don't fill a whole pool group are dropped. */
static void reduce_fc_viz(const float *raw, int in_f, int out_f, int pool, float *out) {
    int viz_rows = in_f / pool;
    for (int r = 0; r < viz_rows; ++r) {
        for (int c = 0; c < out_f; ++c) {
            float sum = 0.0f;
            for (int p = 0; p < pool; ++p) sum += raw[(size_t)(r * pool + p) * out_f + c];
            out[r * out_f + c] = sum / pool;
        }
    }
}

static int grid_cols(int count) { return count <= 4 ? count : (count <= 16 ? 4 : 8); }

static int effective_pool(int in_dim, int out_dim, int requested) {
    if ((long)in_dim * out_dim <= FC_FULL_DETAIL_MAX) return 1;
    int pool = requested < in_dim ? requested : in_dim;
    while ((long)(in_dim / pool) * out_dim > FC_VIZ_MAX_FLOATS) pool++;
    return pool;
}

/* One LayerViz per block that has weights (conv / linear), in block order,
 * named conv1.. and fc1.. by how many of each came before. */
static void build_model_info(const TrainConfig *c, const NetShapes *s, ModelInfo *info) {
    memset(info, 0, sizeof(*info));
    info->cfg = *c;
    int offset = 0, v = 0, n_conv = 0, n_lin = 0;
    for (int i = 0; i < c->n_blocks; ++i) {
        const Block *b = &c->blocks[i];
        LayerShape in = train_block_input(s, b);
        if (b->type == BLK_CONV) {
            LayerViz *lv = &info->layers[v++];
            lv->kind = VIZ_CONV_GRID;
            lv->count = b->a;
            lv->cols = grid_cols(lv->count);
            lv->ksize = b->b;
            lv->full_detail = (in.c == 1);
            lv->offset = offset;
            snprintf(lv->name, sizeof(lv->name), "conv%d", ++n_conv);
            offset += lv->count * lv->ksize * lv->ksize;
        } else if (b->type == BLK_LINEAR) {
            LayerViz *lv = &info->layers[v++];
            lv->kind = VIZ_MATRIX;
            lv->in_dim = in.c;
            lv->out_dim = b->a;
            lv->pool = effective_pool(lv->in_dim, lv->out_dim, c->fc_pool);
            lv->count = lv->in_dim / lv->pool;
            lv->cols = lv->out_dim;
            lv->full_detail = (lv->pool == 1);
            lv->offset = offset;
            snprintf(lv->name, sizeof(lv->name), "fc%d", ++n_lin);
            offset += lv->count * lv->cols;
        }
    }
    info->n_viz = v;
    info->total_floats = offset;
}

/* Pulls every layer's current weights to the host and reduces each into the
 * flat layout described by `info`. `scratch` must hold the largest raw
 * weight tensor. */
static void gather_viz(const Net *n, const ModelInfo *info, float *out, float *scratch) {
    int v = 0;
    for (int i = 0; i < n->n_blocks; ++i) {
        if (n->conv[i]) {
            const LayerViz *lv = &info->layers[v++];
            const Conv2D *cv = n->conv[i];
            if (lv->full_detail) {
                tensor_to_host(conv2d_tensor(n->conv[i]), out + lv->offset);
            } else {
                tensor_to_host(conv2d_tensor(n->conv[i]), scratch);
                reduce_filter_magnitude(scratch, cv->out_channels, cv->in_channels, cv->kernel_size,
                                        out + lv->offset);
            }
        } else if (n->lin[i]) {
            const LayerViz *lv = &info->layers[v++];
            if (lv->full_detail) {
                tensor_to_host(linear_tensor(n->lin[i]), out + lv->offset);
            } else {
                tensor_to_host(linear_tensor(n->lin[i]), scratch);
                reduce_fc_viz(scratch, lv->in_dim, lv->out_dim, lv->pool, out + lv->offset);
            }
        }
    }
}

static size_t max_weight_floats(const Net *n) {
    size_t m = 1;
    for (int i = 0; i < n->n_blocks; ++i) {
        size_t sz = 0;
        if (n->conv[i]) {
            const Conv2D *c = n->conv[i];
            sz = (size_t)c->out_channels * c->in_channels * c->kernel_size * c->kernel_size;
        } else if (n->lin[i]) {
            sz = (size_t)n->lin[i]->in_features * n->lin[i]->out_features;
        }
        if (sz > m) m = sz;
    }
    return m;
}

/* -- framed output ------------------------------------------------------- */

/* Header + any number of payload pieces, written as one message. */
static int send_msgv(FILE *f, uint32_t type, const void *const *parts, const size_t *sizes, int n) {
    size_t total = 0;
    for (int i = 0; i < n; ++i) total += sizes[i];
    MsgHeader h = {TRAIN_MAGIC, type, (uint32_t)total};
    if (fwrite(&h, sizeof(h), 1, f) != 1) return -1;
    for (int i = 0; i < n; ++i)
        if (sizes[i] && fwrite(parts[i], 1, sizes[i], f) != sizes[i]) return -1;
    return fflush(f) == 0 ? 0 : -1;
}

static int send_msg(FILE *f, uint32_t type, const void *p1, size_t n1, const void *p2, size_t n2) {
    const void *parts[2] = {p1, p2};
    size_t sizes[2] = {n1, n2};
    return send_msgv(f, type, parts, sizes, 2);
}

static int send_error(FILE *f, const char *fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "train_worker: %s\n", buf);
    return send_msg(f, MSG_ERROR, buf, strlen(buf) + 1, NULL, 0);
}

/* -- command input ------------------------------------------------------- */

/* Reads one whole TrainCmd. 1 = got one, 0 = the viewer closed its end,
 * -1 = nothing available right now (non-blocking fds only). Commands are far
 * below PIPE_BUF so they arrive atomically; the partial-read loop is just a
 * guard. */
static int read_cmd(int fd, TrainCmd *cmd) {
    size_t have = 0;
    while (have < sizeof(*cmd)) {
        ssize_t n = read(fd, (char *)cmd + have, sizeof(*cmd) - have);
        if (n > 0) { have += (size_t)n; continue; }
        if (n == 0) return 0;
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            if (have == 0) return -1;
            usleep(1000);
            continue;
        }
        return 0;
    }
    return 1;
}

/* Drains every command currently queued on a non-blocking fd, keeping the
 * newest valid one. Returns how many valid commands were read, or -1 if the
 * viewer hung up. */
static int take_latest_cmd(int fd, TrainCmd *latest) {
    int got = 0;
    for (;;) {
        TrainCmd c;
        int r = read_cmd(fd, &c);
        if (r == 0) return -1;
        if (r < 0) return got;
        if (c.magic == TRAIN_MAGIC && c.type == TRAIN_CMD_START) {
            *latest = c;
            got++;
        }
    }
}

/* -- dataset cache ------------------------------------------------------- */

static MnistData g_data, g_test;
static char g_data_name[sizeof(((TrainConfig *)0)->dataset)];
static int g_data_loaded = 0, g_test_loaded = 0;

/* Pixel statistics of the training split, and whether the cached images are
 * currently standardized with them. The cache is flipped in place when a run
 * asks for the other setting, so a Restart with a different `normalize` never
 * needs a reload. */
static float g_pix_mean[MAX_IMG_CH] = {0.0f, 0.0f, 0.0f}, g_pix_std[MAX_IMG_CH] = {1.0f, 1.0f, 1.0f};
static int g_norm_on = 0;

static void dataset_release(void) {
    if (g_data_loaded) mnist_free(&g_data);
    if (g_test_loaded) mnist_free(&g_test);
    g_data_loaded = g_test_loaded = 0;
    g_data_name[0] = '\0';
    g_norm_on = 0;
}

/* One mean / standard deviation per channel (images are stored channel by channel). */
static void compute_pixel_stats(const MnistData *d) {
    size_t plane = (size_t)d->rows * d->cols;
    for (int ch = 0; ch < MAX_IMG_CH; ++ch) {
        g_pix_mean[ch] = 0.0f;
        g_pix_std[ch] = 1.0f;
    }
    for (int ch = 0; ch < d->channels; ++ch) {
        double sum = 0.0, sumsq = 0.0;
        for (int k = 0; k < d->count; ++k) {
            const float *p = d->images + ((size_t)k * d->channels + ch) * plane;
            for (size_t i = 0; i < plane; ++i) {
                sum += p[i];
                sumsq += (double)p[i] * p[i];
            }
        }
        size_t n = plane * (size_t)d->count;
        double mean = n ? sum / (double)n : 0.0;
        double var = n ? sumsq / (double)n - mean * mean : 1.0;
        g_pix_mean[ch] = (float)mean;
        g_pix_std[ch] = var > 1e-12 ? (float)sqrt(var) : 1.0f;
    }
}

static void transform_pixels(MnistData *d, int standardize) {
    size_t plane = (size_t)d->rows * d->cols;
    for (int k = 0; k < d->count; ++k) {
        for (int ch = 0; ch < d->channels; ++ch) {
            float *p = d->images + ((size_t)k * d->channels + ch) * plane;
            float m = g_pix_mean[ch], sd = g_pix_std[ch];
            for (size_t i = 0; i < plane; ++i) p[i] = standardize ? (p[i] - m) / sd : p[i] * sd + m;
        }
    }
}

static void dataset_set_normalized(int on) {
    if (on == g_norm_on) return;
    transform_pixels(&g_data, on);
    if (g_test_loaded) transform_pixels(&g_test, on);
    g_norm_on = on;
}

static int labels_in_range(const MnistData *d, int num_classes) {
    for (int i = 0; i < d->count; ++i)
        if (d->labels[i] < 0 || d->labels[i] >= num_classes) return 0;
    return 1;
}

/* Loads data/<name>/train-* (required) and t10k-* (optional: without it no
 * test accuracy is reported). The name has already passed train_cfg_derive()'s
 * character check, so it is safe to splice into a path. Reloading is skipped
 * when the same dataset is asked for again (every Restart), but the shape and
 * label checks run every time since the config may differ. */
static int dataset_ensure(const TrainConfig *cfg, char *err, size_t errsz) {
    const char *name = cfg->dataset;
    if (!(g_data_loaded && strcmp(g_data_name, name) == 0)) {
        dataset_release();
        char images[128], labels[128];
        snprintf(images, sizeof(images), "data/%s/train-images-idx3-ubyte", name);
        snprintf(labels, sizeof(labels), "data/%s/train-labels-idx1-ubyte", name);
        if (mnist_load(images, labels, &g_data) != 0) {
            snprintf(err, errsz, "could not load dataset '%s' (expected IDX files under data/%s/ - see README)", name, name);
            return -1;
        }
        g_data_loaded = 1;
        snprintf(g_data_name, sizeof(g_data_name), "%s", name);
        snprintf(images, sizeof(images), "data/%s/t10k-images-idx3-ubyte", name);
        snprintf(labels, sizeof(labels), "data/%s/t10k-labels-idx1-ubyte", name);
        g_test_loaded = mnist_load(images, labels, &g_test) == 0;
        compute_pixel_stats(&g_data);
        g_norm_on = 0;
    }
    if (g_data.rows != cfg->img_h || g_data.cols != cfg->img_w || g_data.channels != cfg->img_c) {
        snprintf(err, errsz, "dataset '%s' is %dx%dx%d but the settings say %dx%dx%d", name, g_data.rows, g_data.cols,
                 g_data.channels, cfg->img_h, cfg->img_w, cfg->img_c);
        return -1;
    }
    if (!labels_in_range(&g_data, cfg->num_classes)) {
        snprintf(err, errsz, "dataset '%s' has a label outside 0-%d", name, cfg->num_classes - 1);
        return -1;
    }
    if (g_test_loaded && (g_test.rows != g_data.rows || g_test.cols != g_data.cols || g_test.channels != g_data.channels ||
                          !labels_in_range(&g_test, cfg->num_classes))) {
        fprintf(stderr, "train_worker: ignoring the test split of '%s' (different image size or labels)\n", name);
        mnist_free(&g_test);
        g_test_loaded = 0;
    }
    dataset_set_normalized(cfg->normalize);
    return 0;
}

/* Forward-only pass over the whole test split. */
static float eval_accuracy(Net *net, const MnistData *test, int num_classes, int batch, float *host_logits) {
    int correct = 0;
    for (int i = 0; i < test->count; i += batch) {
        int cb = batch < test->count - i ? batch : test->count - i;
        int shape[4] = {cb, test->channels, test->rows, test->cols};
        ForwardCache cache;
        net_forward(net, tensor_from_data(test->images + (size_t)i * test->channels * test->rows * test->cols, 4, shape),
                    &cache, 0);
        tensor_to_host(cache.logits, host_logits);
        for (int j = 0; j < cb; ++j)
            if (argmax(host_logits + j * num_classes, num_classes) == test->labels[i + j]) correct++;
        free_cache(&cache);
    }
    return test->count > 0 ? (float)correct / (float)test->count : -1.0f;
}

/* Wall-clock seconds (clock() would count CPU time, which is about half of what the user waits). */
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* -- one training run ---------------------------------------------------- */

#define RUN_DONE 0
#define RUN_RESTART 1 /* *cfg already holds the config to start next */
#define RUN_GONE 2    /* the viewer disconnected */
#define RUN_REJECTED 3

static int run_training(TrainConfig *cfg, FILE *fifo, int cmd_fd) {
    char err[256];
    NetShapes shapes;
    if (train_cfg_derive(cfg, &shapes, err, sizeof(err)) != 0) {
        return send_error(fifo, "invalid settings: %s", err) == 0 ? RUN_REJECTED : RUN_GONE;
    }
    if (dataset_ensure(cfg, err, sizeof(err)) != 0) {
        return send_error(fifo, "%s", err) == 0 ? RUN_REJECTED : RUN_GONE;
    }
    const MnistData *train = &g_data;
    const int batch_size = cfg->batch_size;
    const int epochs = cfg->epochs;
    const int nc = cfg->num_classes;
    const int img_px = train->channels * train->rows * train->cols;

    srand((unsigned)cfg->seed);
    Net net;
    net_init(&net, cfg, &shapes);

    ModelInfo info;
    build_model_info(cfg, &shapes, &info);
    if (send_msg(fifo, MSG_MODEL, &info, sizeof(info), NULL, 0) != 0) {
        net_free(&net);
        fprintf(stderr, "train_worker: viewer disconnected, stopping.\n");
        return RUN_GONE;
    }

    /* A second click that arrived while the dataset was loading and the model
     * was being built is either a duplicate of this very config (ignore it)
     * or a newer config the user typed in the meantime (switch to it). */
    TrainCmd queued;
    int queued_n = take_latest_cmd(cmd_fd, &queued);
    if (queued_n < 0) { net_free(&net); return RUN_GONE; }
    if (queued_n > 0 && memcmp(&queued.cfg, cfg, sizeof(*cfg)) != 0) {
        *cfg = queued.cfg;
        net_free(&net);
        return RUN_RESTART;
    }

    float *host_logits = (float *)malloc(sizeof(float) * batch_size * nc);
    float *grad_output = (float *)malloc(sizeof(float) * batch_size * nc);
    float *viz = (float *)malloc(sizeof(float) * (info.total_floats > 0 ? info.total_floats : 1));
    float *scratch = (float *)malloc(sizeof(float) * max_weight_floats(&net));
    unsigned char *pixels = (unsigned char *)malloc((size_t)img_px);
    /* Samples are visited in a fresh random order every epoch: datasets are
     * often stored sorted by class (NEU-CLS: 240 of one defect, then the next),
     * and batches of a single class make plain SGD swing between classes. */
    int *perm = (int *)malloc(sizeof(int) * (size_t)train->count);
    float *batch_x = (float *)malloc(sizeof(float) * (size_t)batch_size * img_px);
    int *batch_y = (int *)malloc(sizeof(int) * (size_t)batch_size);

    double t_start = now_seconds();
    int result = RUN_DONE;
    int batch_no = 0;
    float test_acc = -1.0f;

    for (int epoch = 0; epoch < epochs && result == RUN_DONE; ++epoch) {
        int epoch_correct = 0;
        float epoch_loss = 0.0f;
        /* Batch norm can't take statistics from a single sample, so if the last batch would hold just
         * one, this epoch leaves one (random, the order is shuffled) image out. */
        int n_epoch = (net.has_bn && train->count % batch_size == 1) ? train->count - 1 : train->count;
        for (int i = 0; i < train->count; ++i) perm[i] = i;
        for (int i = train->count - 1; i > 0; --i) {
            int j = rand() % (i + 1), t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }

        for (int i = 0; i < n_epoch && result == RUN_DONE; i += batch_size, ++batch_no) {
            int curr_batch = batch_size;
            float batch_loss = 0.0f;
            int batch_correct = 0;
            int x_shape[4];
            Tensor *grad_tensor;
            ForwardCache cache;

            if (i + curr_batch > n_epoch) curr_batch = n_epoch - i;

            for (int j = 0; j < curr_batch; ++j) {
                const float *src = train->images + (size_t)perm[i + j] * img_px;
                float *dst = batch_x + (size_t)j * img_px;
                if (cfg->augment) {
                    /* mode 1: flip both ways (textures have no "up"); mode 2: flip left-right and
                     * shift by up to 4 px, the usual recipe for photos. Pixels shifted in from
                     * outside are 0 (the dataset mean once standardized). */
                    int flip_h = rand() & 1, flip_v = cfg->augment == 1 ? (rand() & 1) : 0;
                    int dy = 0, dx = 0;
                    if (cfg->augment == 2) {
                        dy = rand() % 9 - 4;
                        dx = rand() % 9 - 4;
                    }
                    for (int ch = 0; ch < train->channels; ++ch) {
                        const float *sp = src + (size_t)ch * train->rows * train->cols;
                        float *dp = dst + (size_t)ch * train->rows * train->cols;
                        for (int y = 0; y < train->rows; ++y) {
                            for (int x = 0; x < train->cols; ++x) {
                                int sy = (flip_v ? train->rows - 1 - y : y) + dy;
                                int sx = (flip_h ? train->cols - 1 - x : x) + dx;
                                int inside = sy >= 0 && sy < train->rows && sx >= 0 && sx < train->cols;
                                dp[y * train->cols + x] = inside ? sp[sy * train->cols + sx] : 0.0f;
                            }
                        }
                    }
                } else {
                    memcpy(dst, src, sizeof(float) * (size_t)img_px);
                }
                batch_y[j] = train->labels[perm[i + j]];
            }
            x_shape[0] = curr_batch; x_shape[1] = train->channels; x_shape[2] = train->rows; x_shape[3] = train->cols;
            net_forward(&net, tensor_from_data(batch_x, 4, x_shape), &cache, 1);
            tensor_to_host(cache.logits, host_logits);

            int sample_pred = 0;
            for (int j = 0; j < curr_batch; ++j) {
                float probs[MAX_CLASSES];
                int k, label = batch_y[j];
                softmax(host_logits + j * nc, probs, nc);
                batch_loss += -logf(probs[label] + 1e-7f);
                for (k = 0; k < nc; ++k) grad_output[j * nc + k] = probs[k] - ((k == label) ? 1.0f : 0.0f);
                if (argmax(probs, nc) == label) batch_correct++;
                if (j == 0) sample_pred = argmax(probs, nc);
            }

            net_zero_grad(&net);
            { int grad_shape[4] = {curr_batch, nc, 1, 1}; grad_tensor = tensor_from_data(grad_output, 4, grad_shape); }
            tensor_backward(cache.logits, grad_tensor);
            tensor_free(grad_tensor);
            net_update(&net, cfg->lr, cfg->momentum);
            free_cache(&cache);

            epoch_loss += batch_loss;
            epoch_correct += batch_correct;

            int last_batch_of_epoch = i + batch_size >= n_epoch;
            if (last_batch_of_epoch && g_test_loaded) test_acc = eval_accuracy(&net, &g_test, nc, batch_size, host_logits);

            Progress prog;
            memset(&prog, 0, sizeof(prog));
            prog.epoch = epoch;
            prog.epochs = epochs;
            prog.batch = i / batch_size;
            prog.batch_count = (n_epoch + batch_size - 1) / batch_size;
            prog.loss = batch_loss / curr_batch;
            prog.acc = (float)batch_correct / curr_batch;
            prog.true_label = batch_y[0];
            prog.pred_label = sample_pred;
            prog.elapsed = (float)(now_seconds() - t_start);
            prog.has_weights = (batch_no % cfg->viz_every == 0) && info.total_floats > 0;
            prog.test_acc = test_acc;
            for (int p = 0; p < img_px; ++p) {
                float v = batch_x[p];
                int ch = p / (train->rows * train->cols);
                if (g_norm_on) v = v * g_pix_std[ch] + g_pix_mean[ch]; /* back to the raw 0..1 pixel for display */
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                pixels[p] = (unsigned char)(v * 255.0f);
            }
            if (prog.has_weights) gather_viz(&net, &info, viz, scratch);

            const void *parts[3] = {&prog, pixels, viz};
            size_t sizes[3] = {sizeof(prog), (size_t)img_px, prog.has_weights ? sizeof(float) * (size_t)info.total_floats : 0};
            if (send_msgv(fifo, MSG_PROGRESS, parts, sizes, 3) != 0) {
                fprintf(stderr, "train_worker: viewer disconnected, stopping.\n");
                result = RUN_GONE;
                break;
            }

            /* Any command arriving mid-run (the Restart button re-sends the
             * current config, Start-from-settings sends a new one) restarts
             * training from scratch with the newest config received. */
            TrainCmd next;
            int pn = take_latest_cmd(cmd_fd, &next);
            if (pn < 0) result = RUN_GONE;
            else if (pn > 0) { *cfg = next.cfg; result = RUN_RESTART; }
        }

        if (result == RUN_DONE) {
            printf("=== epoch %d done: avg loss %.4f, train acc %.4f, test acc %.4f ===\n", epoch,
                   epoch_loss / n_epoch, (float)epoch_correct / n_epoch, test_acc);
        }
    }

    net_free(&net);
    free(host_logits);
    free(grad_output);
    free(viz);
    free(scratch);
    free(pixels);
    free(perm);
    free(batch_x);
    free(batch_y);
    tensor_pool_release(); /* the next run may have different tensor sizes */
    return result;
}

int main(void) {
    FILE *fifo;
    int cmd_fd;

    /* Without this, writing to the data FIFO after the viewer has exited
     * (closing its read end) raises SIGPIPE, whose default disposition kills
     * the process outright - before the send_msg() error path ever gets a
     * chance to shut down cleanly. Ignoring it makes write() return -1/EPIPE
     * instead, which fwrite() surfaces as a short item count. */
    signal(SIGPIPE, SIG_IGN);

    if (mkfifo(TRAIN_FIFO_PATH, 0600) != 0 && errno != EEXIST) {
        fprintf(stderr, "train_worker: mkfifo(%s) failed: %s\n", TRAIN_FIFO_PATH, strerror(errno));
        return 1;
    }
    fprintf(stderr, "train_worker: waiting for train_viewer to connect to %s ...\n", TRAIN_FIFO_PATH);
    fifo = fopen(TRAIN_FIFO_PATH, "wb"); /* blocks until a reader opens the other end */
    if (!fifo) {
        fprintf(stderr, "train_worker: could not open fifo for writing: %s\n", strerror(errno));
        return 1;
    }

    /* Second FIFO, viewer->worker. open() blocks until the viewer opens its
     * write end (same handshake as the data FIFO above). While idle, the
     * read() below blocks until the user clicks Start; the fd only becomes
     * O_NONBLOCK while a run is in progress, so the per-batch poll for a
     * restart never stalls training. */
    if (mkfifo(TRAIN_CMD_FIFO_PATH, 0600) != 0 && errno != EEXIST) {
        fprintf(stderr, "train_worker: mkfifo(%s) failed: %s\n", TRAIN_CMD_FIFO_PATH, strerror(errno));
        fclose(fifo);
        return 1;
    }
    cmd_fd = open(TRAIN_CMD_FIFO_PATH, O_RDONLY);
    if (cmd_fd < 0) {
        fprintf(stderr, "train_worker: could not open command fifo for reading: %s\n", strerror(errno));
        fclose(fifo);
        return 1;
    }
    fprintf(stderr, "train_worker: viewer connected. Waiting for Start...\n");

    /* Each pass is one full (re)start. A restart received mid-run leaves its
     * config in `cfg` and sets have_pending_start so the next pass goes
     * straight back into training instead of blocking on another read(). */
    TrainConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    int have_pending_start = 0;
    for (;;) {
        if (!have_pending_start) {
            TrainCmd cmd;
            int r = read_cmd(cmd_fd, &cmd); /* blocking: cmd_fd is back in blocking mode by the end of each pass */
            if (r <= 0) break;              /* viewer closed the command fifo - nothing left to serve */
            if (cmd.magic != TRAIN_MAGIC || cmd.type != TRAIN_CMD_START) continue;
            cfg = cmd.cfg;
        }
        have_pending_start = 0;

        int flags = fcntl(cmd_fd, F_GETFL, 0);
        fcntl(cmd_fd, F_SETFL, flags | O_NONBLOCK);
        fprintf(stderr, "train_worker: training started.\n");
        int result = run_training(&cfg, fifo, cmd_fd);
        fcntl(cmd_fd, F_SETFL, flags);

        if (result == RUN_GONE) break;
        if (result == RUN_RESTART) {
            have_pending_start = 1;
        } else if (result == RUN_DONE) {
            fprintf(stderr, "train_worker: training finished. Waiting for Restart...\n");
        }
    }

    fclose(fifo);
    close(cmd_fd);
    dataset_release();
    return 0;
}
