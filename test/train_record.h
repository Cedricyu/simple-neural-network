#ifndef TRAIN_RECORD_H
#define TRAIN_RECORD_H

/* Shared wire format between train_worker (pure CUDA, no SDL2) and
 * train_viewer (pure SDL2, no CUDA) - see the comment at the top of
 * train_worker.c for why these are two separate processes instead of one. */

#define TRAIN_FIFO_PATH "/tmp/tinydl_train.fifo"
#define TRAIN_IMG_SIZE 28

/* Second, reverse-direction FIFO: train_data.fifo only ever flows
 * worker->viewer (training progress), so starting/restarting training needs
 * its own channel the other way. The viewer writes a single byte
 * (TRAIN_CMD_START) when the user clicks Start or Restart - both do the same
 * thing on the worker side (reinitialize the model and begin a fresh run),
 * the only difference is which button the user sees depending on whether a
 * run has happened yet. See the big comment above main() in train_worker.c
 * for how the worker interleaves reading this with training. */
#define TRAIN_CMD_FIFO_PATH "/tmp/tinydl_train_cmd.fifo"
#define TRAIN_CMD_START 1

/* Must match model_init() in train_worker.c:
 *   conv1(1->8,3x3,s1) -> conv2(8->16,3x3,s2) -> conv3(16->32,3x3,s2)
 *   -> conv4(32->32,3x3,s1) -> flatten -> fc1(1568->128) -> fc2(128->10, logits)
 * conv1 has in_channels=1, so its kernels are directly viewable as images
 * and sent in full. conv2-4 mix multiple input channels per kernel, so
 * there's no single in-channel grayscale patch to show for those - the
 * worker instead sends, per output channel, the mean(|weight|) over input
 * channels (see reduce_filter_magnitude in train_worker.c): a
 * same-size-as-the-kernel "how much does this output channel care about
 * each kernel position" summary, not the literal weights.
 *
 * fc1/fc2 aren't spatial filters - their weight is just one big (in,out)
 * matrix - so the viewer renders those as a single heatmap image instead of
 * a grid of square thumbnails. fc1's weight is (1568,128): too large to
 * stream raw every batch (~800KB) and too tall to read as a heatmap anyway,
 * so the worker average-pools every FC1_POOL consecutive input rows into one
 * before sending (see reduce_fc1_viz) - cheap, and the point is to see the
 * overall structure, not individual weights. fc2's weight is small enough
 * (128,10) to send as-is. */
#define CONV1_OUT 8
#define CONV1_IN 1
#define CONV1_K 3
#define CONV2_OUT 16
#define CONV2_IN 8
#define CONV2_K 3
#define CONV3_OUT 32
#define CONV3_IN 16
#define CONV3_K 3
#define CONV4_OUT 32
#define CONV4_IN 32
#define CONV4_K 3

#define FC1_POOL 8
#define FC1_VIZ_ROWS 196 /* 1568 / FC1_POOL */
#define FC1_VIZ_COLS 128
#define FC2_ROWS 128
#define FC2_COLS 10

typedef struct TrainRecord {
    int epoch;
    int epochs;
    int batch;
    int batch_count;
    float loss;
    float acc;
    int true_label;
    int pred_label;
    float elapsed;
    unsigned char pixels[TRAIN_IMG_SIZE * TRAIN_IMG_SIZE]; /* 0-255 grayscale */
    float conv1_weights[CONV1_OUT * CONV1_K * CONV1_K];  /* full detail, row-major (8,1,3,3) */
    float conv2_filters[CONV2_OUT * CONV2_K * CONV2_K];  /* mean(|w|) over in-channels, per (out,ky,kx) */
    float conv3_filters[CONV3_OUT * CONV3_K * CONV3_K];
    float conv4_filters[CONV4_OUT * CONV4_K * CONV4_K];
    float fc1_viz[FC1_VIZ_ROWS * FC1_VIZ_COLS]; /* average-pooled, row-major (98,128) */
    float fc2_weights[FC2_ROWS * FC2_COLS];     /* full detail, row-major (128,10) */
} TrainRecord;

#endif
