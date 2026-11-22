#ifndef TRAIN_RECORD_H
#define TRAIN_RECORD_H

/* Shared between train_worker (pure CUDA, no SDL2) and train_viewer (pure
 * SDL2, no CUDA) - see the comment at the top of train_worker.c for why these
 * are two separate processes. Everything here is plain C so both can include
 * it:
 *   - TrainConfig: every parameter the user fills in on the viewer's setup
 *     form (dataset, epochs, learning rate, conv/fc layer stack, ...).
 *   - train_cfg_derive(): the single source of truth for "is this config
 *     valid, and what are the resulting layer shapes". The viewer calls it
 *     for live form validation, the worker calls it again on whatever bytes
 *     arrive over the command FIFO.
 *   - The framed messages the worker streams back (model description, one
 *     progress message per batch, errors). */

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TRAIN_FIFO_PATH "/tmp/tinydl_train.fifo"
#define TRAIN_CMD_FIFO_PATH "/tmp/tinydl_train_cmd.fifo"
#define TRAIN_MAGIC 0x54494E59u /* "TINY" */

/* A dataset decides its own image size and class count (the viewer reads them
 * off the IDX header / label file / classes.txt and puts them in the config);
 * images have 1 (grey) or 3 (colour) channels. These are just upper bounds. */
#define MAX_IMG_SIDE 128
#define MAX_IMG_CH 3
#define MAX_CLASSES 16
#define CLASS_NAME_LEN 24

/* Limits enforced by train_cfg_derive(). The hard caps exist because the
 * framework's CUDA_CHECK() exits the whole process on any CUDA failure
 * (including out-of-memory), so a config that can't fit must be rejected up
 * front with a readable message rather than discovered mid-run. */
#define MAX_BLOCKS 32
#define MAX_VIZ_LAYERS 16 /* conv + linear blocks: the layers that have weights to show */
#define MAX_CONV_CHANNELS 64
#define MAX_KERNEL 7
#define MAX_STRIDE 4
#define MAX_PADDING 3
#define MAX_POOL 4 /* max-pool kernel and stride */
#define MAX_FC_WIDTH 1024
#define MAX_BATCH 512
#define MAX_EPOCHS 1000
#define MAX_FC_WEIGHTS 4194304         /* in*out floats per Linear (16MB) */
#define MAX_ACTIVATION_BYTES 1500000000.0 /* rough forward+backward working set */

/* The model is a graph of blocks, drawn in the viewer as a node graph. Tensor ids: 0 is the input image
 * (img_c channels, img_h x img_w) and id i+1 is the output of block i. A block reads the tensor(s) named by
 * its `in` wires - always an earlier id, so the block order is already a valid evaluation order and
 * there can be no cycles - and the last block's output is the model's output: a flat tensor of
 * num_classes values. A tensor may feed several blocks (that is what a skip connection is), and every
 * block output except the last must be used by some block. */
#define BLK_CONV 1    /* a = out channels, b = kernel, c = stride, d = padding; needs an image (not flat) */
#define BLK_RELU 2    /* no parameters */
#define BLK_FLATTEN 3 /* (C,H,W) -> C*H*W values; needs an image */
#define BLK_LINEAR 4  /* a = out features; needs a flat input */
#define BLK_ADD 5     /* in[0] + in[1], elementwise: both must have the same shape (the residual connection) */
#define BLK_BATCHNORM 6 /* per-channel normalization with learned scale and shift; any shape */
#define BLK_MAXPOOL 7   /* a = window, b = stride; needs an image */
#define BLK_GAP 8       /* global average pool: (C,H,W) -> C values (already flat); needs an image */

typedef struct Block {
    int type;
    int a, b, c, d;
    int in[2]; /* tensor ids this block reads; in[1] is only used by BLK_ADD (else -1) */
} Block;

typedef struct TrainConfig {
    char dataset[32]; /* directory under data/, e.g. "fashion-mnist" */
    int epochs;
    int batch_size;
    float lr;
    int seed;
    int fc_pool;   /* viz only: average-pool this many input rows per pixel of big fc weight heatmaps */
    int viz_every; /* viz only: send the weights every N batches */
    int img_h, img_w; /* the dataset's image size */
    int img_c;        /* ... and channel count (1 = grey, 3 = colour) */
    int num_classes;  /* the dataset's class count: the model must end in this many outputs */
    float momentum;   /* SGD momentum, 0 = plain SGD */
    int normalize;    /* 1 = standardize pixels to zero mean / unit variance (dataset statistics) */
    int augment;      /* 0 = none, 1 = random horizontal + vertical flips, 2 = random horizontal flip + shift by up to 4 px */
    int n_blocks;
    Block blocks[MAX_BLOCKS];
} TrainConfig;

/* Viewer -> worker. Kept far below PIPE_BUF so a command is written and read
 * atomically (never split across two read()s). */
#define TRAIN_CMD_START 1 /* (re)initialize with cfg and train from scratch */
typedef struct TrainCmd {
    uint32_t magic;
    uint32_t type;
    TrainConfig cfg;
} TrainCmd;

/* A tensor shape: an image (c,h,w) or, once flat, c values. */
typedef struct LayerShape {
    int c, h, w;
    int flat;
} LayerShape;

typedef struct NetShapes {
    LayerShape input;
    LayerShape out[MAX_BLOCKS]; /* shape after block i; valid for i < n_ok */
    int n_ok;
    int err_block; /* index of the block train_cfg_derive() rejected, or -1 */
} NetShapes;

/* Shape of tensor `id` (0 = the input image, i+1 = the output of block i). */
static inline LayerShape train_node_shape(const NetShapes *s, int id) { return id == 0 ? s->input : s->out[id - 1]; }
/* Shape block i reads through its first input wire. */
static inline LayerShape train_block_input(const NetShapes *s, const Block *b) { return train_node_shape(s, b->in[0]); }

static inline const char *train_block_type_name(int type) {
    switch (type) {
    case BLK_CONV: return "conv";
    case BLK_RELU: return "relu";
    case BLK_FLATTEN: return "flatten";
    case BLK_LINEAR: return "fc";
    case BLK_ADD: return "add";
    case BLK_BATCHNORM: return "batchnorm";
    case BLK_MAXPOOL: return "maxpool";
    case BLK_GAP: return "gap";
    default: return "?";
    }
}

static inline int train_shape_equal(LayerShape a, LayerShape b) {
    return a.flat == b.flat && a.c == b.c && a.h == b.h && a.w == b.w;
}

static inline int train_cfg_fail(char *err, size_t errsz, const char *fmt, int a, int b) {
    if (err && errsz) snprintf(err, errsz, fmt, a, b);
    return -1;
}

/* Returns 0 and fills *s if cfg is runnable, else -1 with a one-line reason in
 * err (and s->err_block naming the offending block when there is one). */
static inline int train_cfg_derive(const TrainConfig *c, NetShapes *s, char *err, size_t errsz) {
    memset(s, 0, sizeof(*s));
    s->err_block = -1;
    s->input.c = c->img_c;
    s->input.h = c->img_h;
    s->input.w = c->img_w;

    size_t name_len = strnlen(c->dataset, sizeof(c->dataset));
    if (name_len == 0 || name_len >= sizeof(c->dataset)) {
        return train_cfg_fail(err, errsz, "dataset name must be 1-%d characters", (int)sizeof(c->dataset) - 1, 0);
    }
    for (size_t i = 0; i < name_len; ++i) {
        char ch = c->dataset[i];
        int ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' ||
                 ch == '_';
        if (!ok) return train_cfg_fail(err, errsz, "dataset name may only contain letters, digits, - and _", 0, 0);
    }
    if (c->epochs < 1 || c->epochs > MAX_EPOCHS)
        return train_cfg_fail(err, errsz, "epochs must be 1-%d", MAX_EPOCHS, 0);
    if (c->batch_size < 1 || c->batch_size > MAX_BATCH)
        return train_cfg_fail(err, errsz, "batch size must be 1-%d", MAX_BATCH, 0);
    if (!(c->lr > 0.0f) || c->lr > 10.0f || !isfinite(c->lr))
        return train_cfg_fail(err, errsz, "learning rate must be a number in (0, 10]", 0, 0);
    if (c->seed < 0) return train_cfg_fail(err, errsz, "seed must be >= 0", 0, 0);
    if (c->fc_pool < 1 || c->fc_pool > 4096)
        return train_cfg_fail(err, errsz, "fc viz pool must be 1-%d", 4096, 0);
    if (c->viz_every < 1 || c->viz_every > 100000)
        return train_cfg_fail(err, errsz, "weights update interval must be 1-%d", 100000, 0);
    if (c->img_h < 1 || c->img_h > MAX_IMG_SIDE || c->img_w < 1 || c->img_w > MAX_IMG_SIDE)
        return train_cfg_fail(err, errsz, "image size must be 1-%d pixels per side", MAX_IMG_SIDE, 0);
    if (c->num_classes < 2 || c->num_classes > MAX_CLASSES)
        return train_cfg_fail(err, errsz, "the dataset must have 2-%d classes", MAX_CLASSES, 0);
    if (!(c->momentum >= 0.0f) || c->momentum > 0.99f || !isfinite(c->momentum))
        return train_cfg_fail(err, errsz, "momentum must be a number in [0, 0.99]", 0, 0);
    if (c->normalize != 0 && c->normalize != 1)
        return train_cfg_fail(err, errsz, "normalize must be 0 or 1", 0, 0);
    if (c->augment < 0 || c->augment > 2)
        return train_cfg_fail(err, errsz, "augment must be 0, 1 or 2", 0, 0);
    if (c->img_c < 1 || c->img_c > MAX_IMG_CH)
        return train_cfg_fail(err, errsz, "image channels must be 1-%d", MAX_IMG_CH, 0);
    if (c->batch_size < 2) {
        for (int i = 0; i < c->n_blocks && i < MAX_BLOCKS; ++i)
            if (c->blocks[i].type == BLK_BATCHNORM)
                return train_cfg_fail(err, errsz, "batchnorm needs a batch size of at least %d", 2, 0);
    }
    if (c->n_blocks < 0 || c->n_blocks > MAX_BLOCKS)
        return train_cfg_fail(err, errsz, "the model can have at most %d blocks", MAX_BLOCKS, 0);

    double act_floats = (double)c->img_h * c->img_w * c->img_c;
    LayerShape cur = s->input;
    int n_conv = 0, n_lin = 0, n_bn = 0, n_add = 0, n_pool = 0;
    int uses[MAX_BLOCKS + 1] = {0};
    for (int i = 0; i < c->n_blocks; ++i) {
        const Block *b = &c->blocks[i];
        s->err_block = i;
        if (b->in[0] < 0 || b->in[0] > i)
            return train_cfg_fail(err, errsz, "block %d has no valid input wire", i + 1, 0);
        cur = train_node_shape(s, b->in[0]);
        uses[b->in[0]]++;
        switch (b->type) {
        case BLK_CONV: {
            n_conv++;
            if (cur.flat) return train_cfg_fail(err, errsz, "conv%d: needs an image input, but its input is already flat", n_conv, 0);
            if (b->a < 1 || b->a > MAX_CONV_CHANNELS)
                return train_cfg_fail(err, errsz, "conv%d: out channels must be 1-%d", n_conv, MAX_CONV_CHANNELS);
            if (b->b < 1 || b->b > MAX_KERNEL)
                return train_cfg_fail(err, errsz, "conv%d: kernel must be 1-%d", n_conv, MAX_KERNEL);
            if (b->c < 1 || b->c > MAX_STRIDE)
                return train_cfg_fail(err, errsz, "conv%d: stride must be 1-%d", n_conv, MAX_STRIDE);
            if (b->d < 0 || b->d > MAX_PADDING)
                return train_cfg_fail(err, errsz, "conv%d: padding must be 0-%d", n_conv, MAX_PADDING);
            int ho = (cur.h + 2 * b->d - b->b) / b->c + 1;
            int wo = (cur.w + 2 * b->d - b->b) / b->c + 1;
            if (cur.h + 2 * b->d < b->b || cur.w + 2 * b->d < b->b || ho < 1 || wo < 1) {
                if (err && errsz)
                    snprintf(err, errsz, "conv%d: kernel %d is larger than its %dx%d padded input", n_conv, b->b,
                             cur.h + 2 * b->d, cur.w + 2 * b->d);
                return -1;
            }
            cur.c = b->a;
            cur.h = ho;
            cur.w = wo;
            act_floats += (double)cur.c * cur.h * cur.w;
            break;
        }
        case BLK_RELU:
            act_floats += (double)cur.c * cur.h * cur.w;
            break;
        case BLK_FLATTEN:
            if (cur.flat) return train_cfg_fail(err, errsz, "flatten: the input is already flat", 0, 0);
            cur.c = cur.c * cur.h * cur.w;
            cur.h = cur.w = 1;
            cur.flat = 1;
            break;
        case BLK_LINEAR: {
            n_lin++;
            if (!cur.flat)
                return train_cfg_fail(err, errsz, "fc%d: needs a flat input - put a Flatten or Global Avg Pool block before it", n_lin, 0);
            if (b->a < 1 || b->a > MAX_FC_WIDTH)
                return train_cfg_fail(err, errsz, "fc%d: out features must be 1-%d", n_lin, MAX_FC_WIDTH);
            if ((double)cur.c * b->a > MAX_FC_WEIGHTS) {
                if (err && errsz)
                    snprintf(err, errsz, "fc%d: %d x %d weights is too large (limit %d) - add stride or fewer channels",
                             n_lin, cur.c, b->a, MAX_FC_WEIGHTS);
                return -1;
            }
            cur.c = b->a;
            act_floats += (double)b->a;
            break;
        }
        case BLK_ADD: {
            n_add++;
            if (b->in[1] < 0 || b->in[1] > i)
                return train_cfg_fail(err, errsz, "add%d: wire both inputs", n_add, 0);
            if (b->in[1] == b->in[0])
                return train_cfg_fail(err, errsz, "add%d: both inputs are the same tensor", n_add, 0);
            uses[b->in[1]]++;
            LayerShape other = train_node_shape(s, b->in[1]);
            if (!train_shape_equal(cur, other)) {
                if (err && errsz) {
                    char sa[24], sb[24];
                    if (cur.flat) snprintf(sa, sizeof(sa), "%d", cur.c);
                    else snprintf(sa, sizeof(sa), "%dx%dx%d", cur.h, cur.w, cur.c);
                    if (other.flat) snprintf(sb, sizeof(sb), "%d", other.c);
                    else snprintf(sb, sizeof(sb), "%dx%dx%d", other.h, other.w, other.c);
                    snprintf(err, errsz, "add%d: inputs differ in shape (%s vs %s) - a 1x1 conv on the shortcut can match them", n_add, sa, sb);
                }
                return -1;
            }
            act_floats += (double)cur.c * cur.h * cur.w;
            break;
        }
        case BLK_BATCHNORM:
            n_bn++;
            act_floats += (double)cur.c * cur.h * cur.w;
            break;
        case BLK_MAXPOOL: {
            n_pool++;
            if (cur.flat) return train_cfg_fail(err, errsz, "pool%d: needs an image input", n_pool, 0);
            if (b->a < 1 || b->a > MAX_POOL)
                return train_cfg_fail(err, errsz, "pool%d: window must be 1-%d", n_pool, MAX_POOL);
            if (b->b < 1 || b->b > MAX_POOL)
                return train_cfg_fail(err, errsz, "pool%d: stride must be 1-%d", n_pool, MAX_POOL);
            if (cur.h < b->a || cur.w < b->a) {
                if (err && errsz)
                    snprintf(err, errsz, "pool%d: window %d is larger than its %dx%d input", n_pool, b->a, cur.h, cur.w);
                return -1;
            }
            cur.h = (cur.h - b->a) / b->b + 1;
            cur.w = (cur.w - b->a) / b->b + 1;
            act_floats += (double)cur.c * cur.h * cur.w;
            break;
        }
        case BLK_GAP:
            if (cur.flat) return train_cfg_fail(err, errsz, "gap: needs an image input", 0, 0);
            cur.h = cur.w = 1;
            cur.flat = 1;
            act_floats += (double)cur.c;
            break;
        default:
            return train_cfg_fail(err, errsz, "block %d has an unknown type", i + 1, 0);
        }
        if (n_conv + n_lin > MAX_VIZ_LAYERS)
            return train_cfg_fail(err, errsz, "at most %d conv + linear layers", MAX_VIZ_LAYERS, 0);
        s->out[i] = cur;
        s->n_ok = i + 1;
    }
    for (int i = 0; i + 1 < c->n_blocks; ++i) {
        if (uses[i + 1] == 0) {
            s->err_block = i;
            if (err && errsz)
                snprintf(err, errsz, "block %d (%s): its output is not used by anything - wire it in or delete it", i + 1,
                         train_block_type_name(c->blocks[i].type));
            return -1;
        }
    }
    s->err_block = c->n_blocks > 0 ? c->n_blocks - 1 : -1;
    if (c->n_blocks == 0 || !cur.flat || cur.c != c->num_classes)
        return train_cfg_fail(err, errsz, "the model must end with a Linear block with out=%d (one value per class)",
                              c->num_classes, 0);
    s->err_block = -1;

    double act_bytes = act_floats * c->batch_size * 4.0 * 3.0; /* activations + grads + scratch */
    if (act_bytes > MAX_ACTIVATION_BYTES) {
        if (err && errsz)
            snprintf(err, errsz, "needs ~%.1f GB of activations at batch %d - lower batch size or channels",
                     act_bytes / 1e9, c->batch_size);
        return -1;
    }
    return 0;
}

/* Worker -> viewer: framed messages. Every message is MsgHeader + `bytes` of
 * payload; the viewer re-assembles across partial non-blocking reads. */
typedef struct MsgHeader {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} MsgHeader;

#define MSG_MODEL 1    /* ModelInfo - sent once at the start of every run */
#define MSG_PROGRESS 2 /* Progress, then img_h*img_w pixel bytes, then (if has_weights) info.total_floats floats */
#define MSG_ERROR 3    /* char[]: a NUL-terminated human-readable reason */
#define MSG_MAX_BYTES (16u * 1024u * 1024u)

/* How the weights viewer draws one layer. conv layers with a single input
 * channel are sent in full; conv layers mixing several input channels send
 * mean(|w|) over the input-channel axis (one k*k map per output channel);
 * fc layers are one (rows x cols) matrix, average-pooled over input rows when
 * they're too big to stream every batch. `offset` is in floats, into the
 * weights payload that follows a Progress. */
#define VIZ_CONV_GRID 0
#define VIZ_MATRIX 1
typedef struct LayerViz {
    int kind;
    int count;       /* CONV_GRID: filters | MATRIX: rows */
    int cols;        /* CONV_GRID: grid columns | MATRIX: cols */
    int ksize;       /* CONV_GRID: kernel size */
    int full_detail; /* CONV_GRID: in_channels == 1 | MATRIX: pool == 1 */
    int in_dim;      /* MATRIX: real (unpooled) input rows */
    int out_dim;
    int pool; /* MATRIX: input rows averaged per displayed row */
    int offset;
    char name[16];
} LayerViz;

typedef struct ModelInfo {
    TrainConfig cfg;
    int n_viz;
    LayerViz layers[MAX_VIZ_LAYERS];
    int total_floats;
} ModelInfo;

typedef struct Progress {
    int epoch;
    int epochs;
    int batch;
    int batch_count;
    float loss;
    float acc;
    int true_label;
    int pred_label;
    float elapsed;
    int has_weights;
    float test_acc; /* accuracy on the held-out test split after the latest finished epoch, or -1 if not evaluated yet */
} Progress;

#endif
