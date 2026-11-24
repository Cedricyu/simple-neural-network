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
#include "seg_loss.h"
#include "tensor.cuh"
#include "tensorops.cuh"
#include "train_record.h"
#include "vae.cuh"

/* fc weights up to this many floats are streamed in full; bigger ones are
 * average-pooled over input rows (TrainConfig::fc_pool) so a heatmap update
 * stays cheap to send every batch. FC_VIZ_MAX_FLOATS caps the pooled result
 * no matter what fc_pool was asked for. */
#define FC_FULL_DETAIL_MAX 32768 /* the network view draws every connection of a layer that is sent in full */
#define FC_VIZ_MAX_FLOATS 65536

/* One entry per block; conv[i] / lin[i] / bn[i] are only set for the blocks of that type. */
typedef struct Net {
    int n_blocks;
    int has_bn;
    Block blocks[MAX_BLOCKS];
    Conv2D *conv[MAX_BLOCKS];
    Linear *lin[MAX_BLOCKS];
    BatchNorm *bn[MAX_BLOCKS];
    int rep_block; /* index of the VAE sampling block, or -1 */
} Net;

/* Every tensor created by one forward pass, in creation order, so freeing in
 * reverse order releases views (flatten) before the tensors they alias. */
#define MAX_CACHE (MAX_BLOCKS + 4)
typedef struct ForwardCache {
    Tensor *owned[MAX_CACHE];
    int n;
    Tensor *logits;
    Tensor *latent_in; /* VAE: the (B, 2Z) mean / log-variance tensor that fed the sampling block */
    Tensor *val[MAX_BLOCKS + 1]; /* every tensor of the pass by id (0 = input, i+1 = block i's output), until free_cache */
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
    n->rep_block = -1;
    for (int i = 0; i < c->n_blocks; ++i) {
        const Block *b = &c->blocks[i];
        LayerShape in = train_block_input(s, b);
        n->blocks[i] = *b;
        if (b->type == BLK_REPARAM) n->rep_block = i;
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

/* Standard-normal noise from the C library's generator (Box-Muller), so a run is reproducible from its seed. */
static void fill_normal_rand(float *dst, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0), u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double r = sqrt(-2.0 * log(u1));
        dst[i] = (float)(r * cos(6.283185307179586 * u2));
        if (i + 1 < n) dst[i + 1] = (float)(r * sin(6.283185307179586 * u2));
    }
}

/* Evaluates the block graph in order (every block only reads earlier tensors) and leaves the last
 * tensor in c->logits. Takes ownership of x via the cache. `training` selects batch statistics
 * (1) or running statistics (0) in batch-norm blocks, and random noise (1) or none (0, z = mu) in the
 * VAE's sampling block. With z_override (a VAE only), the encoder is skipped: the blocks after the
 * sampling block run on that latent instead, which is how new images are generated. */
static void net_forward_ex(Net *n, Tensor *x, ForwardCache *c, int training, Tensor *z_override) {
    Tensor *val[MAX_BLOCKS + 1] = {0};
    int first = 0;
    c->n = 0;
    c->latent_in = NULL;
    if (z_override) {
        first = n->rep_block + 1;
        val[first] = cache_add(c, z_override);
    } else {
        val[0] = cache_add(c, x);
    }
    for (int i = first; i < n->n_blocks; ++i) {
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
        case BLK_UPSAMPLE: out = upsample_nearest_forward(in, b->a); break;
        case BLK_CONCAT: out = tensor_concat_channels(in, val[b->in[1]]); break;
        case BLK_RESHAPE: out = tensor_reshape4(in, b->a, b->b, b->c); break;
        case BLK_REPARAM: {
            int B = in->shape[0], Z = in->shape[1] / 2;
            float *e = (float *)calloc((size_t)B * Z, sizeof(float));
            if (training) fill_normal_rand(e, (size_t)B * Z);
            int es[2] = {B, Z};
            Tensor *eps = cache_add(c, tensor_from_data(e, 2, es)); /* must outlive the backward pass */
            free(e);
            c->latent_in = in;
            out = reparam_forward(in, eps, b->f);
            break;
        }
        }
        if (!out) {
            /* train_cfg_derive() accepted these shapes, so this is a bug, not bad input. */
            fprintf(stderr, "train_worker: block %d (%s) failed to run\n", i + 1, train_block_type_name(b->type));
            exit(1);
        }
        val[i + 1] = cache_add(c, out);
    }
    c->logits = val[n->n_blocks];
    memcpy(c->val, val, sizeof(val));
}

static void net_forward(Net *n, Tensor *x, ForwardCache *c, int training) { net_forward_ex(n, x, c, training, NULL); }

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

/* One optimizer step on every parameter. `step` counts steps from 1 (Adam's bias correction needs it). */
static void update_param(Tensor *t, const TrainConfig *cfg, int step) {
    if (cfg->optimizer == OPT_ADAM) tensor_update_adam(t, cfg->lr, 0.9f, 0.999f, 1e-8f, step);
    else tensor_update_momentum(t, cfg->lr, cfg->momentum);
}

static void net_update(Net *n, const TrainConfig *cfg, int step) {
    for (int i = 0; i < n->n_blocks; ++i) {
        if (n->conv[i]) {
            update_param(conv2d_tensor(n->conv[i]), cfg, step);
            update_param(n->conv[i]->bias, cfg, step);
        }
        if (n->lin[i]) {
            update_param(linear_tensor(n->lin[i]), cfg, step);
            update_param(n->lin[i]->bias, cfg, step);
        }
        if (n->bn[i]) {
            update_param(n->bn[i]->gamma, cfg, step);
            update_param(n->bn[i]->beta, cfg, step);
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
    int offset = 0, v = 0, n_conv = 0, n_lin = 0, act_of_block[MAX_BLOCKS];
    for (int i = 0; i < c->n_blocks; ++i) act_of_block[i] = -1;
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

            /* the live activations of this layer: its own output, or the relu right after it */
            ActLayer *al = &info->act[info->n_act];
            act_of_block[i] = info->n_act++;
            al->dim = b->a;
            al->in_dim = in.c;
            al->viz_index = v - 1;
            al->offset = info->total_act_floats;
            al->tid = i + 1;
            for (int j = i + 1; j < c->n_blocks; ++j)
                if (c->blocks[j].type == BLK_RELU && c->blocks[j].in[0] == i + 1) {
                    al->tid = j + 1;
                    al->relu = 1;
                    break;
                }
            info->total_act_floats += al->dim;
            /* what feeds it: walk back through relus to the block that produced its input */
            int id = b->in[0];
            while (id > 0 && c->blocks[id - 1].type == BLK_RELU) id = c->blocks[id - 1].in[0];
            al->prev = -1;
            if (id > 0 && c->blocks[id - 1].type == BLK_LINEAR) al->prev = act_of_block[id - 1];
            else if (id == 0) snprintf(al->src, sizeof(al->src), "input");
            else snprintf(al->src, sizeof(al->src), "%s", c->blocks[id - 1].type == BLK_REPARAM ? "latent z" : train_block_type_name(c->blocks[id - 1].type));
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

/* The shown sample's row of every fc layer's output (or relu output), after the forward pass of this batch. */
static void gather_activations(const ModelInfo *info, const ForwardCache *c, int shown, float *out, float *tmp) {
    for (int k = 0; k < info->n_act; ++k) {
        const ActLayer *al = &info->act[k];
        const Tensor *t = c->val[al->tid];
        if (!t || t->ndim != 2 || t->shape[1] != al->dim || shown >= t->shape[0]) {
            memset(out + al->offset, 0, sizeof(float) * (size_t)al->dim);
            continue;
        }
        tensor_to_host((Tensor *)t, tmp);
        memcpy(out + al->offset, tmp + (size_t)shown * al->dim, sizeof(float) * (size_t)al->dim);
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
        /* a masks file next to the images makes this a segmentation dataset; the test masks are optional */
        snprintf(images, sizeof(images), "data/%s/train-masks-idx3-ubyte", name);
        mnist_load_masks(images, &g_data);
        if (g_test_loaded && g_data.masks) {
            snprintf(images, sizeof(images), "data/%s/t10k-masks-idx3-ubyte", name);
            if (mnist_load_masks(images, &g_test) != 0) {
                mnist_free(&g_test);
                g_test_loaded = 0;
            }
        }
        compute_pixel_stats(&g_data);
        g_norm_on = 0;
    }
    if (g_data.rows != cfg->img_h || g_data.cols != cfg->img_w || g_data.channels != cfg->img_c) {
        snprintf(err, errsz, "dataset '%s' is %dx%dx%d but the settings say %dx%dx%d", name, g_data.rows, g_data.cols,
                 g_data.channels, cfg->img_h, cfg->img_w, cfg->img_c);
        return -1;
    }
    if (cfg->task == TASK_SEGMENT) {
        if (!g_data.masks) {
            snprintf(err, errsz, "dataset '%s' has no train-masks-idx3-ubyte, so it cannot be used for segmentation", name);
            return -1;
        }
        size_t n = (size_t)g_data.count * g_data.rows * g_data.cols;
        for (size_t i = 0; i < n; ++i)
            if (g_data.masks[i] >= cfg->num_classes) {
                snprintf(err, errsz, "dataset '%s' has a mask value outside 0-%d", name, cfg->num_classes - 1);
                return -1;
            }
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

/* Forward-only pass over the test split for segmentation: foreground dice and IoU, accumulated over every
 * pixel of every image (not averaged per image), averaged over the foreground classes. */
static void eval_segmentation(Net *net, const MnistData *test, int nc, int batch, float *host_logits, float *dice,
                              float *iou) {
    size_t px = (size_t)test->rows * test->cols;
    double tp[MAX_CLASSES] = {0}, fp[MAX_CLASSES] = {0}, fn[MAX_CLASSES] = {0};
    for (int i = 0; i < test->count; i += batch) {
        int cb = batch < test->count - i ? batch : test->count - i;
        int shape[4] = {cb, test->channels, test->rows, test->cols};
        ForwardCache cache;
        net_forward(net, tensor_from_data(test->images + (size_t)i * test->channels * px, 4, shape), &cache, 0);
        tensor_to_host(cache.logits, host_logits);
        for (int j = 0; j < cb; ++j) {
            const float *lg = host_logits + (size_t)j * nc * px;
            const unsigned char *gt = test->masks + (size_t)(i + j) * px;
            for (size_t k = 0; k < px; ++k) {
                int best = 0;
                for (int c = 1; c < nc; ++c)
                    if (lg[c * px + k] > lg[best * px + k]) best = c;
                for (int c = 1; c < nc; ++c) {
                    int pred = best == c, truth = gt[k] == c;
                    tp[c] += pred && truth;
                    fp[c] += pred && !truth;
                    fn[c] += !pred && truth;
                }
            }
        }
        free_cache(&cache);
    }
    double d = 0.0, u = 0.0;
    for (int c = 1; c < nc; ++c) {
        d += (2 * tp[c] + 1e-9) / (2 * tp[c] + fp[c] + fn[c] + 1e-9);
        u += (tp[c] + 1e-9) / (tp[c] + fp[c] + fn[c] + 1e-9);
    }
    *dice = (float)(d / (nc - 1));
    *iou = (float)(u / (nc - 1));
}

/* -- predictions on a fixed set of samples --------------------------------- */

/* Which samples to show: spread evenly over the split. For segmentation two thirds are samples that
 * contain foreground (there is little to look at on the rest), mixed in 2:1 so every page of the viewer
 * has both. Returns how many were picked. */
static int pick_pred_indices(const MnistData *d, int seg, int max_n, int *idx) {
    int n = 0;
    if (!seg) {
        int want = d->count < max_n ? d->count : max_n;
        for (int i = 0; i < want; ++i) idx[n++] = (int)((long)i * d->count / want);
        return n;
    }
    size_t px = (size_t)d->rows * d->cols;
    int *fg = (int *)malloc(sizeof(int) * (size_t)d->count), *bg = (int *)malloc(sizeof(int) * (size_t)d->count);
    int nf = 0, nb = 0;
    for (int i = 0; i < d->count; ++i) {
        int has = 0;
        for (size_t k = 0; k < px && !has; ++k) has = d->masks[(size_t)i * px + k] != 0;
        if (has) fg[nf++] = i;
        else bg[nb++] = i;
    }
    int want_fg = max_n * 2 / 3 < nf ? max_n * 2 / 3 : nf;
    int want_bg = max_n - want_fg < nb ? max_n - want_fg : nb;
    int a = 0, b = 0;
    while (a < want_fg || b < want_bg) {
        for (int r = 0; r < 2 && a < want_fg; ++r, ++a) idx[n++] = fg[(long)a * nf / want_fg];
        if (b < want_bg) { idx[n++] = bg[(long)b * nb / want_bg]; ++b; }
    }
    free(fg);
    free(bg);
    return n;
}

/* Predicts the picked samples with the current weights (eval mode) and sends them. Returns 0, or -1 if
 * the viewer is gone. */
static int send_predictions(FILE *fifo, const TrainConfig *cfg, Net *net, const MnistData *d, int from_test, int epoch) {
    int idx[PRED_MAX_SAMPLES];
    const int seg = cfg->task == TASK_SEGMENT, nc = cfg->num_classes;
    int n = pick_pred_indices(d, seg, seg ? 36 : PRED_MAX_SAMPLES, idx);
    if (n == 0) return 0;
    const size_t px = (size_t)d->rows * d->cols, img_px = px * (size_t)d->channels;
    const size_t sample_bytes = train_sample_bytes(cfg);

    float *x = (float *)malloc(sizeof(float) * (size_t)n * img_px);
    for (int j = 0; j < n; ++j) memcpy(x + (size_t)j * img_px, d->images + (size_t)idx[j] * img_px, sizeof(float) * img_px);
    int shape[4] = {n, d->channels, d->rows, d->cols};
    ForwardCache cache;
    net_forward(net, tensor_from_data(x, 4, shape), &cache, 0);
    size_t out_per = seg ? (size_t)nc * px : (size_t)nc;
    float *logits = (float *)malloc(sizeof(float) * (size_t)n * out_per);
    tensor_to_host(cache.logits, logits);
    free_cache(&cache);

    size_t record = sizeof(PredSample) + sample_bytes;
    size_t total = sizeof(PredHeader) + (size_t)n * record;
    unsigned char *buf = (unsigned char *)calloc(1, total);
    PredHeader *h = (PredHeader *)buf;
    h->n = n;
    h->epoch = epoch;
    h->from_test = from_test;
    for (int j = 0; j < n; ++j) {
        unsigned char *rec = buf + sizeof(PredHeader) + (size_t)j * record;
        PredSample *ps = (PredSample *)rec;
        unsigned char *pix = rec + sizeof(PredSample);
        for (size_t p = 0; p < img_px; ++p) {
            float v = x[(size_t)j * img_px + p];
            int ch = (int)(p / px);
            if (g_norm_on) v = v * g_pix_std[ch] + g_pix_mean[ch];
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            pix[p] = (unsigned char)(v * 255.0f);
        }
        if (seg) {
            const float *lg = logits + (size_t)j * nc * px;
            const unsigned char *gt = d->masks + (size_t)idx[j] * px;
            unsigned char *gt_out = pix + img_px, *pred_out = pix + img_px + px;
            long tp = 0, fp = 0, fn = 0;
            for (size_t k = 0; k < px; ++k) {
                int best = 0;
                for (int c = 1; c < nc; ++c)
                    if (lg[c * px + k] > lg[best * px + k]) best = c;
                gt_out[k] = gt[k];
                pred_out[k] = (unsigned char)best;
                int a = gt[k] != 0, b = best != 0;
                tp += a && b; fp += !a && b; fn += a && !b;
            }
            ps->a = (int)(tp + fn);
            ps->b = (int)(tp + fp);
            ps->f = (tp + fp + fn) > 0 ? (float)(2.0 * tp / (2.0 * tp + fp + fn)) : -1.0f;
        } else {
            float probs[MAX_CLASSES];
            softmax(logits + (size_t)j * nc, probs, nc);
            ps->a = d->labels[idx[j]];
            ps->b = argmax(probs, nc);
            ps->f = probs[ps->b];
        }
    }
    free(x);
    free(logits);
    int rc = send_msg(fifo, MSG_PREDICTIONS, buf, total, NULL, 0);
    free(buf);
    return rc;
}

/* -- VAE ------------------------------------------------------------------- */

/* Pixels back to what the decoder is trained to produce: the raw 0..1 image, whether or not the encoder's
 * input was standardized. */
static void raw_targets(const float *x, float *target, int B, int C, size_t px) {
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (size_t p = 0; p < px; ++p) {
                size_t at = ((size_t)b * C + c) * px + p;
                float v = g_norm_on ? x[at] * g_pix_std[c] + g_pix_mean[c] : x[at];
                target[at] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            }
}

/* Reconstruction loss: binary cross entropy of sigmoid(logits) against the raw pixels (each pixel read as a
 * probability), summed over the pixels of an image and averaged over the batch. grad (optional) receives
 * d(loss)/d(logits) = (sigmoid(logit) - pixel) / B. */
static double bce_logits_grad(const float *logits, const float *target, size_t per_sample, int B, float *grad) {
    double total = 0.0;
    for (size_t i = 0; i < per_sample * (size_t)B; ++i) {
        float z = logits[i], x = target[i];
        total += (double)(fmaxf(z, 0.0f) - x * z + log1pf(expf(-fabsf(z))));
        if (grad) grad[i] = (1.0f / (1.0f + expf(-z)) - x) / (float)B;
    }
    return total / B;
}

/* KL(N(mu, exp(logvar)) || N(0, 1)) in nats, averaged over the batch; lat is (B, 2Z): mu then logvar. */
static double kl_mean(const float *lat, int B, int Z) {
    double total = 0.0;
    for (int b = 0; b < B; ++b)
        for (int k = 0; k < Z; ++k) {
            double mu = lat[(size_t)b * 2 * Z + k], lv = lat[(size_t)b * 2 * Z + Z + k];
            lv = lv < -10.0 ? -10.0 : (lv > 10.0 ? 10.0 : lv);
            total += -0.5 * (1.0 + lv - mu * mu - exp(lv));
        }
    return total / B;
}

/* Forward-only pass over the test split with z = mu: the reconstruction loss per pixel. */
static float eval_recon(Net *net, const MnistData *test, int batch, float *host_logits, float *target) {
    size_t px = (size_t)test->rows * test->cols, img_px = px * (size_t)test->channels;
    double total = 0.0;
    for (int i = 0; i < test->count; i += batch) {
        int cb = batch < test->count - i ? batch : test->count - i;
        int shape[4] = {cb, test->channels, test->rows, test->cols};
        ForwardCache cache;
        net_forward(net, tensor_from_data(test->images + (size_t)i * img_px, 4, shape), &cache, 0);
        tensor_to_host(cache.logits, host_logits);
        raw_targets(test->images + (size_t)i * img_px, target, cb, test->channels, px);
        total += bce_logits_grad(host_logits, target, img_px, cb, NULL) * cb;
        free_cache(&cache);
    }
    return test->count > 0 ? (float)(total / test->count / (double)img_px) : -1.0f;
}

static unsigned char to_byte(float v) {
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return (unsigned char)(v * 255.0f);
}

/* Everything the generative page shows (see GenHeader): reconstructions, samples decoded from fixed random
 * latents, and interpolations between the latents of two test images. Returns 0, or -1 if the viewer is gone. */
static int send_generated(FILE *fifo, const TrainConfig *cfg, Net *net, const MnistData *d, int epoch, int zdim) {
    const size_t px = (size_t)d->rows * d->cols, img_px = px * (size_t)d->channels;
    const int n_rec = GEN_RECON < d->count ? GEN_RECON : d->count, n_smp = GEN_SAMPLES, rows = GEN_ROWS, steps = GEN_STEPS;
    (void)cfg;

    /* 1. reconstructions (z = mu) of images spread over the split */
    float *x = (float *)malloc(sizeof(float) * (size_t)n_rec * img_px);
    for (int j = 0; j < n_rec; ++j)
        memcpy(x + (size_t)j * img_px, d->images + (size_t)((long)j * d->count / n_rec) * img_px, sizeof(float) * img_px);
    int shape[4] = {n_rec, d->channels, d->rows, d->cols};
    ForwardCache cache;
    net_forward(net, tensor_from_data(x, 4, shape), &cache, 0);
    float *rec = (float *)malloc(sizeof(float) * (size_t)n_rec * img_px);
    tensor_to_host(cache.logits, rec);
    free_cache(&cache);
    float *target = (float *)malloc(sizeof(float) * (size_t)n_rec * img_px);
    raw_targets(x, target, n_rec, d->channels, px);

    /* 2. the latents (mu) of four images, for the two interpolation rows */
    int pair_idx[4] = {d->count / 7, d->count / 3, d->count / 2, d->count * 3 / 4};
    float *xp = (float *)malloc(sizeof(float) * 4 * img_px);
    for (int j = 0; j < 4; ++j) memcpy(xp + (size_t)j * img_px, d->images + (size_t)pair_idx[j] * img_px, sizeof(float) * img_px);
    int shape4[4] = {4, d->channels, d->rows, d->cols};
    net_forward(net, tensor_from_data(xp, 4, shape4), &cache, 0);
    float *lat = (float *)malloc(sizeof(float) * 4 * 2 * (size_t)zdim);
    tensor_to_host(cache.latent_in, lat);
    free_cache(&cache);
    free(xp);

    /* 3. decode: n_smp fixed random latents, then rows * steps interpolated ones, in one batch */
    int n_dec = n_smp + rows * steps;
    float *z = (float *)malloc(sizeof(float) * (size_t)n_dec * zdim);
    unsigned rng = 12345u; /* fixed: the same z every epoch, so the samples visibly sharpen */
    for (int i = 0; i < n_smp * zdim; i += 2) {
        rng = rng * 1664525u + 1013904223u;
        double u1 = ((rng >> 8) + 0.5) / 16777216.0;
        rng = rng * 1664525u + 1013904223u;
        double u2 = ((rng >> 8) + 0.5) / 16777216.0;
        double r = sqrt(-2.0 * log(u1));
        z[i] = (float)(r * cos(6.283185307179586 * u2));
        if (i + 1 < n_smp * zdim) z[i + 1] = (float)(r * sin(6.283185307179586 * u2));
    }
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < steps; ++k) {
            float t = steps > 1 ? (float)k / (float)(steps - 1) : 0.0f;
            const float *a = lat + (size_t)(2 * r) * 2 * zdim, *b = lat + (size_t)(2 * r + 1) * 2 * zdim; /* mu = first zdim */
            float *dst = z + (size_t)(n_smp + r * steps + k) * zdim;
            for (int q = 0; q < zdim; ++q) dst[q] = (1.0f - t) * a[q] + t * b[q];
        }
    int zshape[2] = {n_dec, zdim};
    net_forward_ex(net, NULL, &cache, 0, tensor_from_data(z, 2, zshape));
    float *dec = (float *)malloc(sizeof(float) * (size_t)n_dec * img_px);
    tensor_to_host(cache.logits, dec);
    free_cache(&cache);

    size_t n_images = 2 * (size_t)n_rec + (size_t)n_dec;
    size_t total = sizeof(GenHeader) + n_images * img_px;
    unsigned char *buf = (unsigned char *)calloc(1, total);
    GenHeader *h = (GenHeader *)buf;
    h->epoch = epoch;
    h->n_recon = n_rec;
    h->n_samples = n_smp;
    h->n_rows = rows;
    h->n_steps = steps;
    h->z_dim = zdim;
    unsigned char *w = buf + sizeof(GenHeader);
    for (int j = 0; j < n_rec; ++j) {
        for (size_t p = 0; p < img_px; ++p) *w++ = to_byte(target[(size_t)j * img_px + p]);
        for (size_t p = 0; p < img_px; ++p) *w++ = to_byte(1.0f / (1.0f + expf(-rec[(size_t)j * img_px + p])));
    }
    for (size_t k = 0; k < (size_t)n_dec * img_px; ++k) *w++ = to_byte(1.0f / (1.0f + expf(-dec[k])));
    int rc = send_msg(fifo, MSG_GENERATED, buf, total, NULL, 0);
    free(buf); free(x); free(rec); free(target); free(lat); free(z); free(dec);
    return rc;
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
    const int seg = cfg->task == TASK_SEGMENT;
    const int gen = cfg->task == TASK_GENERATE;
    const size_t map_px = (size_t)train->rows * train->cols;
    const size_t out_per_sample = seg ? (size_t)nc * map_px : (gen ? (size_t)img_px : (size_t)nc); /* logits per sample */
    const size_t sample_bytes = train_sample_bytes(cfg);

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

    float *host_logits = (float *)malloc(sizeof(float) * (size_t)batch_size * out_per_sample);
    float *grad_output = (float *)malloc(sizeof(float) * (size_t)batch_size * out_per_sample);
    /* VAE: the latent size, the KL weight, and scratch for the (B, 2Z) mean / log-variance and the raw-pixel targets */
    const int zdim = gen ? shapes.out[net.rep_block].c : 0;
    const float beta = gen ? net.blocks[net.rep_block].f : 0.0f;
    float *lat_buf = gen ? (float *)malloc(sizeof(float) * (size_t)batch_size * 2 * zdim) : NULL;
    float *target_buf = gen ? (float *)malloc(sizeof(float) * (size_t)batch_size * img_px) : NULL;
    unsigned char *batch_mask = seg ? (unsigned char *)malloc((size_t)batch_size * map_px) : NULL;
    float *viz = (float *)malloc(sizeof(float) * (info.total_floats > 0 ? info.total_floats : 1));
    float *acts = (float *)calloc((size_t)(info.total_act_floats > 0 ? info.total_act_floats : 1), sizeof(float));
    size_t max_act_tmp = 1;
    for (int k = 0; k < info.n_act; ++k)
        if ((size_t)batch_size * info.act[k].dim > max_act_tmp) max_act_tmp = (size_t)batch_size * info.act[k].dim;
    float *act_tmp = (float *)malloc(sizeof(float) * max_act_tmp);
    float *scratch = (float *)malloc(sizeof(float) * max_weight_floats(&net));
    unsigned char *pixels = (unsigned char *)malloc(sample_bytes);
    /* Samples are visited in a fresh random order every epoch: datasets are
     * often stored sorted by class (NEU-CLS: 240 of one defect, then the next),
     * and batches of a single class make plain SGD swing between classes. */
    int *perm = (int *)malloc(sizeof(int) * (size_t)train->count);
    float *batch_x = (float *)malloc(sizeof(float) * (size_t)batch_size * img_px);
    int *batch_y = (int *)malloc(sizeof(int) * (size_t)batch_size);

    double t_start = now_seconds();
    int result = RUN_DONE;
    int batch_no = 0;
    float test_acc = -1.0f, test_iou = -1.0f;

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
                    const unsigned char *msrc = seg ? train->masks + (size_t)perm[i + j] * map_px : NULL;
                    for (int ch = 0; ch < train->channels; ++ch) {
                        const float *sp = src + (size_t)ch * train->rows * train->cols;
                        float *dp = dst + (size_t)ch * train->rows * train->cols;
                        for (int y = 0; y < train->rows; ++y) {
                            for (int x = 0; x < train->cols; ++x) {
                                int sy = (flip_v ? train->rows - 1 - y : y) + dy;
                                int sx = (flip_h ? train->cols - 1 - x : x) + dx;
                                int inside = sy >= 0 && sy < train->rows && sx >= 0 && sx < train->cols;
                                dp[y * train->cols + x] = inside ? sp[sy * train->cols + sx] : 0.0f;
                                /* the mask moves exactly like the image (shifted-in pixels are background) */
                                if (ch == 0 && msrc)
                                    batch_mask[(size_t)j * map_px + (size_t)y * train->cols + x] =
                                        inside ? msrc[sy * train->cols + sx] : 0;
                            }
                        }
                    }
                } else {
                    memcpy(dst, src, sizeof(float) * (size_t)img_px);
                    if (seg) memcpy(batch_mask + (size_t)j * map_px, train->masks + (size_t)perm[i + j] * map_px, map_px);
                }
                batch_y[j] = train->labels[perm[i + j]];
            }
            x_shape[0] = curr_batch; x_shape[1] = train->channels; x_shape[2] = train->rows; x_shape[3] = train->cols;
            net_forward(&net, tensor_from_data(batch_x, 4, x_shape), &cache, 1);
            tensor_to_host(cache.logits, host_logits);
            if (gen) tensor_to_host(cache.latent_in, lat_buf);

            int sample_pred = 0;
            int shown = 0; /* which sample of the batch the dashboard shows */
            float batch_dice = 0.0f, batch_recon_px = 0.0f, batch_kl = 0.0f;
            if (gen) {
                raw_targets(batch_x, target_buf, curr_batch, train->channels, map_px);
                double recon = bce_logits_grad(host_logits, target_buf, img_px, curr_batch, grad_output);
                double kl = kl_mean(lat_buf, curr_batch, zdim);
                /* the logged loss is per pixel so it is comparable across image sizes; the KL gradient itself
                 * is added by the reparam block's backward */
                batch_loss = (float)((recon + (double)beta * kl) / (double)img_px) * (float)curr_batch;
                batch_recon_px = (float)(recon / (double)img_px);
                batch_kl = (float)kl;
            } else if (seg) {
                float l = seg_loss_grad(host_logits, batch_mask, curr_batch, nc, train->rows, train->cols, 1.0f, grad_output);
                if (l < 0.0f) { fprintf(stderr, "train_worker: out of memory in the loss\n"); exit(1); }
                batch_loss = l * (float)curr_batch;
                /* the sample with the most foreground (a tumour you can look at) is the one shown, and
                 * the batch's foreground dice stands in for "accuracy" */
                long best_fg = -1;
                double tp = 0, fp = 0, fn = 0;
                for (int j = 0; j < curr_batch; ++j) {
                    const float *lg = host_logits + (size_t)j * nc * map_px;
                    const unsigned char *gt = batch_mask + (size_t)j * map_px;
                    long fg = 0;
                    for (size_t k = 0; k < map_px; ++k) {
                        int best = 0;
                        for (int c = 1; c < nc; ++c)
                            if (lg[c * map_px + k] > lg[best * map_px + k]) best = c;
                        int pred = best != 0, truth = gt[k] != 0;
                        tp += pred && truth;
                        fp += pred && !truth;
                        fn += !pred && truth;
                        fg += truth;
                    }
                    if (fg > best_fg) { best_fg = fg; shown = j; }
                }
                batch_dice = (tp + fp + fn) > 0 ? (float)(2 * tp / (2 * tp + fp + fn)) : 1.0f;
            } else {
                for (int j = 0; j < curr_batch; ++j) {
                    float probs[MAX_CLASSES];
                    int k, label = batch_y[j];
                    softmax(host_logits + j * nc, probs, nc);
                    batch_loss += -logf(probs[label] + 1e-7f);
                    for (k = 0; k < nc; ++k) grad_output[j * nc + k] = (probs[k] - ((k == label) ? 1.0f : 0.0f)) / (float)curr_batch; /* the loss is the batch mean */
                    if (argmax(probs, nc) == label) batch_correct++;
                    if (j == 0) sample_pred = argmax(probs, nc);
                }
            }

            net_zero_grad(&net);
            {
                int grad_shape[4] = {curr_batch, gen ? train->channels : nc, seg || gen ? train->rows : 1,
                                     seg || gen ? train->cols : 1};
                grad_tensor = tensor_from_data(grad_output, 4, grad_shape);
            }
            tensor_backward(cache.logits, grad_tensor);
            tensor_free(grad_tensor);
            net_update(&net, cfg, batch_no + 1);
            gather_activations(&info, &cache, shown, acts, act_tmp); /* before the pass's tensors are freed */
            free_cache(&cache);

            epoch_loss += batch_loss;
            epoch_correct += batch_correct;

            int last_batch_of_epoch = i + batch_size >= n_epoch;
            if (last_batch_of_epoch && g_test_loaded) {
                if (gen) test_acc = eval_recon(&net, &g_test, batch_size, host_logits, target_buf);
                else if (seg) eval_segmentation(&net, &g_test, nc, batch_size, host_logits, &test_acc, &test_iou);
                else test_acc = eval_accuracy(&net, &g_test, nc, batch_size, host_logits);
            }
            if (last_batch_of_epoch && gen) { /* the viewer's generative page */
                if (send_generated(fifo, cfg, &net, g_test_loaded ? &g_test : train, epoch, zdim) != 0) {
                    fprintf(stderr, "train_worker: viewer disconnected, stopping.\n");
                    result = RUN_GONE;
                    free_cache(&cache);
                    break;
                }
            }
            if (last_batch_of_epoch && !gen) { /* the viewer's "predictions" page */
                if (send_predictions(fifo, cfg, &net, g_test_loaded ? &g_test : train, g_test_loaded, epoch) != 0) {
                    fprintf(stderr, "train_worker: viewer disconnected, stopping.\n");
                    result = RUN_GONE;
                    free_cache(&cache);
                    break;
                }
            }

            Progress prog;
            memset(&prog, 0, sizeof(prog));
            prog.epoch = epoch;
            prog.epochs = epochs;
            prog.batch = i / batch_size;
            prog.batch_count = (n_epoch + batch_size - 1) / batch_size;
            prog.loss = batch_loss / curr_batch;
            prog.acc = gen ? batch_recon_px : (seg ? batch_dice : (float)batch_correct / curr_batch);
            prog.true_label = batch_y[0];
            prog.pred_label = sample_pred;
            prog.elapsed = (float)(now_seconds() - t_start);
            prog.has_weights = (batch_no % cfg->viz_every == 0) && info.total_floats > 0;
            prog.test_acc = test_acc;
            prog.test_iou = gen ? batch_kl : test_iou; /* generative: the batch's KL in nats per image */
            for (int p = 0; p < img_px; ++p) {
                float v = batch_x[(size_t)shown * img_px + p];
                int ch = p / (train->rows * train->cols);
                if (g_norm_on) v = v * g_pix_std[ch] + g_pix_mean[ch]; /* back to the raw 0..1 pixel for display */
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                pixels[p] = (unsigned char)(v * 255.0f);
            }
            if (gen) { /* then the reconstruction of that same sample */
                for (int p = 0; p < img_px; ++p) pixels[img_px + p] = to_byte(1.0f / (1.0f + expf(-host_logits[(size_t)shown * img_px + p])));
            }
            if (seg) {
                /* then the true mask and the predicted mask of the shown sample, one class index per pixel */
                const unsigned char *gt = batch_mask + (size_t)shown * map_px;
                const float *lg = host_logits + (size_t)shown * nc * map_px;
                memcpy(pixels + img_px, gt, map_px);
                for (size_t k = 0; k < map_px; ++k) {
                    int best = 0;
                    for (int c = 1; c < nc; ++c)
                        if (lg[c * map_px + k] > lg[best * map_px + k]) best = c;
                    pixels[img_px + map_px + k] = (unsigned char)best;
                }
            }
            if (prog.has_weights) gather_viz(&net, &info, viz, scratch);

            const void *parts[4] = {&prog, pixels, acts, viz};
            size_t sizes[4] = {sizeof(prog), sample_bytes, sizeof(float) * (size_t)info.total_act_floats,
                               prog.has_weights ? sizeof(float) * (size_t)info.total_floats : 0};
            if (send_msgv(fifo, MSG_PROGRESS, parts, sizes, 4) != 0) {
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
            if (gen)
                printf("=== epoch %d done: avg loss/pixel %.4f, test recon/pixel %.4f ===\n", epoch, epoch_loss / n_epoch,
                       test_acc);
            else if (seg)
                printf("=== epoch %d done: avg loss %.4f, test dice %.4f, test IoU %.4f ===\n", epoch, epoch_loss / n_epoch,
                       test_acc, test_iou);
            else
                printf("=== epoch %d done: avg loss %.4f, train acc %.4f, test acc %.4f ===\n", epoch,
                       epoch_loss / n_epoch, (float)epoch_correct / n_epoch, test_acc);
        }
    }

    net_free(&net);
    free(host_logits);
    free(grad_output);
    free(batch_mask);
    free(lat_buf);
    free(target_buf);
    free(viz);
    free(acts);
    free(act_tmp);
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
