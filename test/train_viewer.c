/* SDL2 viewer half of the live-training UI - reads TrainRecord structs from
 * a FIFO and draws them. Deliberately has no CUDA dependency at all (see the
 * big comment at the top of train_worker.c for why this is a separate
 * process from the trainer). Run both together with `make train_gui`. */
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <SDL.h>
#include <SDL_ttf.h>

#include "train_record.h"

#define WIN_W 1150
#define WIN_H 700
#define HISTORY_LEN 600
#define LOSS_CLIP 2.5f
#define MAX_LABEL_W 440
#define MAX_LABEL_H 32

/* Shared between render_frame() (draws it) and pump_events() (hit-tests
 * clicks against it) so the two can never drift apart. */
static const SDL_Rect WEIGHTS_BUTTON = {830, 370, 300, 150};

/* Training doesn't begin until Start is clicked - centered in the chart area,
 * which has nothing to show yet anyway before the first record arrives.
 * Restart is a small persistent control once a run exists (whether still in
 * progress or finished), tucked into the bottom of the info box in place of
 * the old static hint line. */
static const SDL_Rect START_BUTTON = {265, 360, 300, 90};
static const SDL_Rect RESTART_BUTTON = {830, 644, 300, 36};

/* Static (compile-time-known) description of the ConvNet train_worker.c
 * builds - purely informational, doesn't need anything from the FIFO. Two
 * lines per box: name, then shape + kernel/stride. Keep in sync with
 * model_init() in train_worker.c if that ever changes. */
typedef struct ArchBox {
    const char *name;
    const char *info;
} ArchBox;
static const ArchBox ARCH_BOXES[] = {
    {"input", "28x28x1"},
    {"conv1", "28x28x8 3x3 s1"},
    {"conv2", "14x14x16 3x3 s2"},
    {"conv3", "7x7x32 3x3 s2"},
    {"conv4", "7x7x32 3x3 s1"},
    {"fc1", "1568 -> 128"},
    {"fc2", "128 -> 10"},
};
#define NUM_ARCH_BOXES (int)(sizeof(ARCH_BOXES) / sizeof(ARCH_BOXES[0]))

/* Fashion-MNIST's label order is fixed by the dataset itself (not something
 * train_worker.c computes or sends) - purely a display lookup. */
static const char *CLASS_NAMES[10] = {"T-shirt/top", "Trouser", "Pullover", "Dress",  "Coat",
                                       "Sandal",      "Shirt",   "Sneaker",  "Bag",    "Ankle boot"};

enum {
    LBL_CHART_CAPTION,
    LBL_DIGIT_CAPTION,
    LBL_EPOCH,
    LBL_BATCH,
    LBL_LOSS,
    LBL_ELAPSED,
    LBL_WEIGHTS_BTN_TITLE,
    LBL_WEIGHTS_BTN_SUB,
    LBL_START_TITLE,
    LBL_RESTART,
    LBL_ARCH_CAPTION,
    LBL_ARCH_BASE /* NUM_ARCH_BOXES * 2 slots follow (name line, info line) */
};
#define NUM_LABEL_SLOTS (LBL_ARCH_BASE + NUM_ARCH_BOXES * 2)

/* Text inside the weights window - separate slot space from the main
 * window's since they belong to a different renderer. One slot per tab
 * button plus one for the caption above the active layer's grid (only one
 * layer is ever shown at a time - see Gui::weights_tab). */
enum {
    WLBL_TAB0,
    WLBL_TAB1,
    WLBL_TAB2,
    WLBL_TAB3,
    WLBL_TAB4,
    WLBL_TAB5,
    WLBL_CAPTION,
    WLBL_STATS,
    WLBL_ZOOM,
    NUM_WEIGHTS_LABEL_SLOTS
};

/* One tab per conv layer. conv1 has in_channels=1 so its kernels are full
 * detail; conv2-4 mix multiple input channels per kernel, so those show
 * mean(|weight|) over the input-channel axis instead (see
 * reduce_filter_magnitude in train_worker.c) - a same-size "how much this
 * output channel weighs each kernel position overall" map, not the literal
 * weights. fc1/fc2 (the MLP head after conv4) aren't shown here - they're
 * not spatial filters, so there's no grid of thumbnails for them the way
 * there is for the conv layers. `cols` is just a layout choice picked per
 * layer to keep cells roughly square given how many filters there are. */
#define TABKIND_CONV_GRID 0 /* grid of square (ksize,ksize) thumbnails, one per filter */
#define TABKIND_FC_MATRIX 1 /* a single (rows,cols) heatmap - reuses count=rows, cols=cols */
typedef struct WeightsTab {
    const char *tab_label;
    const char *caption;
    int kind;
    int count; /* CONV_GRID: filter count | FC_MATRIX: matrix rows */
    int cols;  /* CONV_GRID: grid columns | FC_MATRIX: matrix cols */
    int ksize; /* CONV_GRID only: kernel size (square) */
} WeightsTab;
static const WeightsTab WEIGHTS_TABS[6] = {
    {"conv1", "conv1 - 8 filters (3x3, full detail)", TABKIND_CONV_GRID, CONV1_OUT, 4, CONV1_K},
    {"conv2", "conv2 - 16 maps (3x3, mean |w| over in-ch)", TABKIND_CONV_GRID, CONV2_OUT, 4, CONV2_K},
    {"conv3", "conv3 - 32 maps (3x3, mean |w| over in-ch)", TABKIND_CONV_GRID, CONV3_OUT, 8, CONV3_K},
    {"conv4", "conv4 - 32 maps (3x3, mean |w| over in-ch)", TABKIND_CONV_GRID, CONV4_OUT, 8, CONV4_K},
    {"fc1", NULL /* built from FC1_POOL at render time, see render_weights_window */, TABKIND_FC_MATRIX,
     FC1_VIZ_ROWS, FC1_VIZ_COLS, 0},
    {"fc2", "fc2 weight (128->10, full detail)", TABKIND_FC_MATRIX, FC2_ROWS, FC2_COLS, 0},
};
#define NUM_WEIGHTS_TABS 6

#define COL_BG 13, 15, 20, 255
#define COL_SURFACE 22, 25, 34, 255
#define COL_BORDER 42, 47, 59, 255
#define COL_GRID 30, 34, 44, 255
#define COL_TEXT 238, 240, 244, 255
#define COL_TEXT_DIM 170, 176, 189, 255
#define COL_LOSS 57, 135, 229, 255
#define COL_GOOD 12, 163, 12, 255
#define COL_BAD 230, 103, 103, 255

typedef struct Gui {
    SDL_Window *win;
    SDL_Renderer *ren;
    TTF_Font *font;
    TTF_Font *font_big;
    SDL_Texture *digit_tex;
    SDL_Texture *label_tex[NUM_LABEL_SLOTS];
    float loss_hist[HISTORY_LEN];
    int hist_count;
    int hist_head;
    int mouse_x, mouse_y;

    int cmd_fd;      /* write end of the viewer->worker command FIFO */
    int started;     /* Start (or Restart) has been clicked at least once */
    int has_record;  /* at least one TrainRecord has arrived since the last (re)start */

    /* Opened on demand by clicking the "view live weights" button. Textures
     * and label caches belong to weights_ren specifically (SDL textures
     * aren't shared across renderers), so they get their own arrays instead
     * of reusing anything from the main window. All NULL when closed. */
    SDL_Window *weights_win;
    SDL_Renderer *weights_ren;
    int weights_tab; /* which of WEIGHTS_TABS is shown; persists across close/reopen */
    SDL_Texture *weights_label_tex[NUM_WEIGHTS_LABEL_SLOTS];
    SDL_Texture *tex_conv1[CONV1_OUT];
    SDL_Texture *tex_conv2[CONV2_OUT];
    SDL_Texture *tex_conv3[CONV3_OUT];
    SDL_Texture *tex_conv4[CONV4_OUT];
    SDL_Texture *tex_fc1;
    SDL_Texture *tex_fc2;

    /* Zoom into whatever the active tab is showing: the tab's content (filter
     * grid or fc heatmap) is drawn once into this fixed-size offscreen
     * target, then a centered crop of it is scaled up into the visible box -
     * draw_filter_section/draw_fc_matrix don't need to know zoom exists at
     * all, they just always draw at native size into the content texture. */
    SDL_Texture *weights_content_tex;
    float weights_zoom; /* 1.0 = fully zoomed out (the whole content visible) */
} Gui;

static int gui_init(Gui *g) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return -1;
    }
    if (TTF_Init() != 0) {
        fprintf(stderr, "TTF_Init failed: %s\n", TTF_GetError());
        return -1;
    }
    g->win = SDL_CreateWindow("tinyDL - Fashion-MNIST training", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIN_W,
                               WIN_H, SDL_WINDOW_SHOWN);
    if (!g->win) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return -1;
    }
    g->ren = SDL_CreateRenderer(g->win, -1, SDL_RENDERER_SOFTWARE);
    SDL_SetRenderDrawBlendMode(g->ren, SDL_BLENDMODE_BLEND);

    const char *font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf";
    g->font = TTF_OpenFont(font_path, 14);
    g->font_big = TTF_OpenFont(font_path, 20);
    if (!g->font || !g->font_big) {
        fprintf(stderr, "TTF_OpenFont failed (%s): %s\n", font_path, TTF_GetError());
        return -1;
    }

    g->digit_tex =
        SDL_CreateTexture(g->ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, TRAIN_IMG_SIZE, TRAIN_IMG_SIZE);
    if (g->digit_tex) SDL_SetTextureScaleMode(g->digit_tex, SDL_ScaleModeNearest);

    g->hist_count = 0;
    g->hist_head = 0;
    return 0;
}

static void close_weights_window(Gui *g);

static void gui_shutdown(Gui *g) {
    close_weights_window(g);
    if (g->digit_tex) SDL_DestroyTexture(g->digit_tex);
    for (int i = 0; i < NUM_LABEL_SLOTS; ++i)
        if (g->label_tex[i]) SDL_DestroyTexture(g->label_tex[i]);
    if (g->font) TTF_CloseFont(g->font);
    if (g->font_big) TTF_CloseFont(g->font_big);
    if (g->ren) SDL_DestroyRenderer(g->ren);
    if (g->win) SDL_DestroyWindow(g->win);
    TTF_Quit();
    SDL_Quit();
}

static void gui_push_point(Gui *g, float loss) {
    g->loss_hist[g->hist_head] = loss;
    g->hist_head = (g->hist_head + 1) % HISTORY_LEN;
    if (g->hist_count < HISTORY_LEN) g->hist_count++;
}

/* `slot` is the address of one persistent per-label SDL_Texture* (either a
 * slot in Gui::label_tex for the main window, or in Gui::weights_label_tex
 * for the weights window) so re-rendering the same line of text every frame
 * updates that texture's pixels in place instead of doing a fresh
 * SDL_CreateTexture/DestroyTexture - see the comment on Gui::digit_tex's
 * origin story for why that churn matters on this renderer/driver. Takes the
 * renderer explicitly (rather than always using a Gui's main `ren`) so the
 * same function draws into either window. */
static void draw_text(SDL_Renderer *ren, SDL_Texture **slot, TTF_Font *font, const char *s, int x, int y,
                       SDL_Color color) {
    SDL_Surface *raw = TTF_RenderText_Blended(font, s, color);
    if (!raw) return;
    SDL_Surface *surf = SDL_ConvertSurfaceFormat(raw, SDL_PIXELFORMAT_RGBA32, 0);
    SDL_FreeSurface(raw);
    if (!surf) return;

    if (!*slot) {
        *slot = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, MAX_LABEL_W, MAX_LABEL_H);
        if (*slot) SDL_SetTextureBlendMode(*slot, SDL_BLENDMODE_BLEND);
    }
    SDL_Texture *tex = *slot;
    if (!tex) { SDL_FreeSurface(surf); return; }

    int w = surf->w < MAX_LABEL_W ? surf->w : MAX_LABEL_W;
    int h = surf->h < MAX_LABEL_H ? surf->h : MAX_LABEL_H;
    void *pixels;
    int pitch;
    if (SDL_LockTexture(tex, NULL, &pixels, &pitch) == 0) {
        for (int row = 0; row < h; ++row) {
            memcpy((Uint8 *)pixels + row * pitch, (Uint8 *)surf->pixels + row * surf->pitch, (size_t)w * 4);
        }
        SDL_UnlockTexture(tex);
        SDL_Rect src = {0, 0, w, h};
        SDL_Rect dst = {x, y, w, h};
        SDL_RenderCopy(ren, tex, &src, &dst);
    } else {
        fprintf(stderr, "SDL_LockTexture failed: %s\n", SDL_GetError());
    }
    SDL_FreeSurface(surf);
}

static void draw_panel(SDL_Renderer *ren, SDL_Rect r) {
    SDL_SetRenderDrawColor(ren, COL_SURFACE);
    SDL_RenderFillRect(ren, &r);
    SDL_SetRenderDrawColor(ren, COL_BORDER);
    SDL_RenderDrawRect(ren, &r);
}

static void draw_chart(Gui *g, SDL_Rect r) {
    draw_panel(g->ren, r);
    draw_text(g->ren, &g->label_tex[LBL_CHART_CAPTION], g->font, "loss (0-2.5)", r.x + 10, r.y + 8,
              (SDL_Color){COL_TEXT_DIM});

    int pad_top = 32, pad_bottom = 10, pad_x = 10;
    int px0 = r.x + pad_x, px1 = r.x + r.w - pad_x;
    int py0 = r.y + pad_top, py1 = r.y + r.h - pad_bottom;
    int plot_w = px1 - px0, plot_h = py1 - py0;

    SDL_SetRenderDrawColor(g->ren, COL_GRID);
    for (int i = 0; i <= 4; ++i) {
        int y = py0 + plot_h * i / 4;
        SDL_RenderDrawLine(g->ren, px0, y, px1, y);
    }

    if (g->hist_count < 2) return;
    int n = g->hist_count;
    int start = (g->hist_head - n + HISTORY_LEN) % HISTORY_LEN;

    SDL_SetRenderDrawColor(g->ren, COL_LOSS);
    for (int i = 1; i < n; ++i) {
        float v0 = fminf(g->loss_hist[(start + i - 1) % HISTORY_LEN], LOSS_CLIP) / LOSS_CLIP;
        float v1 = fminf(g->loss_hist[(start + i) % HISTORY_LEN], LOSS_CLIP) / LOSS_CLIP;
        int x0 = px0 + plot_w * (i - 1) / (HISTORY_LEN - 1);
        int x1 = px0 + plot_w * i / (HISTORY_LEN - 1);
        int y0 = py1 - (int)(plot_h * v0);
        int y1 = py1 - (int)(plot_h * v1);
        SDL_RenderDrawLine(g->ren, x0, y0, x1, y1);
    }
}

static void draw_architecture(Gui *g, SDL_Rect r) {
    draw_panel(g->ren, r);
    draw_text(g->ren, &g->label_tex[LBL_ARCH_CAPTION], g->font, "model architecture", r.x + 10, r.y + 6,
              (SDL_Color){COL_TEXT_DIM});

    int n = NUM_ARCH_BOXES;
    int box_w = 140, box_h = 50;
    int gap = (r.w - 20 - box_w * n) / (n - 1);
    int y = r.y + 28 + (r.h - 28 - box_h) / 2;

    for (int i = 0; i < n; ++i) {
        int x = r.x + 10 + i * (box_w + gap);
        SDL_Rect box = {x, y, box_w, box_h};
        draw_panel(g->ren, box);
        draw_text(g->ren, &g->label_tex[LBL_ARCH_BASE + i * 2], g->font, ARCH_BOXES[i].name, box.x + 8, box.y + 6,
                  (SDL_Color){COL_TEXT});
        draw_text(g->ren, &g->label_tex[LBL_ARCH_BASE + i * 2 + 1], g->font, ARCH_BOXES[i].info, box.x + 8,
                  box.y + 26, (SDL_Color){COL_TEXT_DIM});

        if (i < n - 1) {
            int ax0 = box.x + box_w, ax1 = ax0 + gap;
            int ay = y + box_h / 2;
            SDL_SetRenderDrawColor(g->ren, COL_TEXT_DIM);
            SDL_RenderDrawLine(g->ren, ax0, ay, ax1 - 6, ay);
            SDL_RenderDrawLine(g->ren, ax1 - 6, ay, ax1 - 12, ay - 4);
            SDL_RenderDrawLine(g->ren, ax1 - 6, ay, ax1 - 12, ay + 4);
        }
    }
}

/* Normalizes one kernel/filter-map/weight-matrix to its own min/max - the
 * raw values are small floats clustered near 0 (conv1, fc1, fc2) or
 * magnitudes that shrink with depth (the reduced conv2-4 maps), so a shared
 * scale would just show a stack of near-identical gray squares. Takes a bare
 * texture + dims (not Gui) so one function serves every layer and either
 * renderer - SDL_LockTexture only needs the texture itself. rows/cols need
 * not be equal (conv kernels are square; fc weight matrices aren't).
 * out_min/out_max are optional (pass NULL to ignore) - callers that want to
 * show the actual numbers next to the auto-contrasted image (the picture
 * alone can't tell you whether 0.009 drifted to 0.011 since every frame
 * re-stretches its own min..max to black..white) can read them back here
 * instead of re-scanning the data themselves. */
static void fill_filter_texture(SDL_Texture *tex, const float *data, int rows, int cols, float *out_min,
                                 float *out_max) {
    int n = rows * cols;
    float dmin = data[0], dmax = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] < dmin) dmin = data[i];
        if (data[i] > dmax) dmax = data[i];
    }
    if (out_min) *out_min = dmin;
    if (out_max) *out_max = dmax;
    float range = dmax - dmin;
    if (range < 1e-6f) range = 1e-6f;

    void *pixels;
    int pitch;
    if (SDL_LockTexture(tex, NULL, &pixels, &pitch) != 0) return;
    for (int y = 0; y < rows; ++y) {
        Uint32 *row_px = (Uint32 *)((Uint8 *)pixels + y * pitch);
        for (int x = 0; x < cols; ++x) {
            float val = data[y * cols + x];
            Uint8 v = (Uint8)(((val - dmin) / range) * 255.0f);
            row_px[x] = (0xFFu << 24) | (v << 16) | (v << 8) | v;
        }
    }
    SDL_UnlockTexture(tex);
}

/* Clickable panel in the main window that replaces the old inline filter
 * grid; clicking it pops the live weights out into their own window (see
 * open_weights_window below) instead of eating space in the main layout. */
static void draw_weights_button(Gui *g, SDL_Rect box, int hovered) {
    SDL_SetRenderDrawColor(g->ren, COL_SURFACE);
    SDL_RenderFillRect(g->ren, &box);
    SDL_SetRenderDrawColor(g->ren, COL_LOSS);
    SDL_RenderDrawRect(g->ren, &box);
    if (hovered) {
        SDL_Rect inset = {box.x + 2, box.y + 2, box.w - 4, box.h - 4};
        SDL_RenderDrawRect(g->ren, &inset);
    }
    draw_text(g->ren, &g->label_tex[LBL_WEIGHTS_BTN_TITLE], g->font_big, "view layer weights", box.x + 18,
              box.y + box.h / 2 - 26, (SDL_Color){COL_LOSS});
    draw_text(g->ren, &g->label_tex[LBL_WEIGHTS_BTN_SUB], g->font, "opens a live weights window", box.x + 18,
              box.y + box.h / 2 + 4, (SDL_Color){COL_TEXT_DIM});
}

/* Start and Restart both just write this one byte - the worker treats them
 * identically (reinitialize the model, train from scratch); which button the
 * user sees is purely a viewer-side UI state (Gui::started). Ignores write()
 * failures: if the worker's end is gone there's nothing useful to do about
 * it here, and the next frame's non-blocking read on the data FIFO will
 * notice the disconnect on its own. */
static void send_train_command(Gui *g) {
    unsigned char cmd = TRAIN_CMD_START;
    ssize_t written = write(g->cmd_fd, &cmd, 1);
    (void)written;
}

/* Shown in the chart area until the first record arrives. Before Start is
 * clicked (!g->started) this is the actual clickable button; once clicked,
 * g->started flips true immediately but the worker still needs a moment to
 * initialize the model and run its first batch - rendering a plain "waiting"
 * message for that gap (rather than leaving the button up) avoids it looking
 * like the click didn't register and inviting a confusing double-click. */
static void draw_start_panel(Gui *g, SDL_Rect chart_box, int hovered) {
    draw_panel(g->ren, chart_box);
    draw_text(g->ren, &g->label_tex[LBL_CHART_CAPTION], g->font, "loss (0-2.5)", chart_box.x + 10, chart_box.y + 8,
              (SDL_Color){COL_TEXT_DIM});

    if (g->started) {
        draw_text(g->ren, &g->label_tex[LBL_START_TITLE], g->font_big, "waiting for training to start...",
                  START_BUTTON.x - 20, START_BUTTON.y + 34, (SDL_Color){COL_TEXT_DIM});
        return;
    }

    SDL_SetRenderDrawColor(g->ren, COL_SURFACE);
    SDL_RenderFillRect(g->ren, &START_BUTTON);
    SDL_SetRenderDrawColor(g->ren, COL_LOSS);
    SDL_RenderDrawRect(g->ren, &START_BUTTON);
    if (hovered) {
        SDL_Rect inset = {START_BUTTON.x + 2, START_BUTTON.y + 2, START_BUTTON.w - 4, START_BUTTON.h - 4};
        SDL_RenderDrawRect(g->ren, &inset);
    }
    draw_text(g->ren, &g->label_tex[LBL_START_TITLE], g->font_big, "> start training", START_BUTTON.x + 62,
              START_BUTTON.y + 34, (SDL_Color){COL_LOSS});
}

static void draw_restart_button(Gui *g, int hovered) {
    SDL_SetRenderDrawColor(g->ren, COL_SURFACE);
    SDL_RenderFillRect(g->ren, &RESTART_BUTTON);
    SDL_SetRenderDrawColor(g->ren, COL_LOSS);
    SDL_RenderDrawRect(g->ren, &RESTART_BUTTON);
    if (hovered) {
        SDL_Rect inset = {RESTART_BUTTON.x + 2, RESTART_BUTTON.y + 2, RESTART_BUTTON.w - 4, RESTART_BUTTON.h - 4};
        SDL_RenderDrawRect(g->ren, &inset);
    }
    draw_text(g->ren, &g->label_tex[LBL_RESTART], g->font, "restart training", RESTART_BUTTON.x + 92,
              RESTART_BUTTON.y + 9, (SDL_Color){COL_LOSS});
}

#define WEIGHTS_WIN_W 700
#define WEIGHTS_WIN_H 600
#define WEIGHTS_TAB_Y 20
#define WEIGHTS_TAB_H 44
#define WEIGHTS_CONTENT_W (WEIGHTS_WIN_W - 40)
#define WEIGHTS_CONTENT_H (WEIGHTS_WIN_H - (WEIGHTS_TAB_Y + WEIGHTS_TAB_H + 16) - 20)
#define WEIGHTS_ZOOM_MIN 1.0f
#define WEIGHTS_ZOOM_MAX 8.0f

/* Pure function of i (no Gui) so render_weights_window (draws tabs) and
 * pump_events (hit-tests clicks on them) use the exact same rects. */
static SDL_Rect weights_tab_rect(int i) {
    int bar_w = WEIGHTS_WIN_W - 40;
    int tab_w = bar_w / NUM_WEIGHTS_TABS;
    SDL_Rect r = {20 + i * tab_w, WEIGHTS_TAB_Y, tab_w - 4, WEIGHTS_TAB_H};
    return r;
}

static void close_weights_window(Gui *g) {
    if (!g->weights_win) return;
    for (int i = 0; i < NUM_WEIGHTS_LABEL_SLOTS; ++i) {
        if (g->weights_label_tex[i]) SDL_DestroyTexture(g->weights_label_tex[i]);
        g->weights_label_tex[i] = NULL;
    }
#define DESTROY_ARR(arr, n)                                       \
    do {                                                          \
        for (int _i = 0; _i < (n); ++_i) {                        \
            if (g->arr[_i]) SDL_DestroyTexture(g->arr[_i]);        \
            g->arr[_i] = NULL;                                    \
        }                                                         \
    } while (0)
    DESTROY_ARR(tex_conv1, CONV1_OUT);
    DESTROY_ARR(tex_conv2, CONV2_OUT);
    DESTROY_ARR(tex_conv3, CONV3_OUT);
    DESTROY_ARR(tex_conv4, CONV4_OUT);
#undef DESTROY_ARR
    if (g->tex_fc1) SDL_DestroyTexture(g->tex_fc1);
    g->tex_fc1 = NULL;
    if (g->tex_fc2) SDL_DestroyTexture(g->tex_fc2);
    g->tex_fc2 = NULL;
    if (g->weights_content_tex) SDL_DestroyTexture(g->weights_content_tex);
    g->weights_content_tex = NULL;
    if (g->weights_ren) SDL_DestroyRenderer(g->weights_ren);
    g->weights_ren = NULL;
    SDL_DestroyWindow(g->weights_win);
    g->weights_win = NULL;
}

static void create_filter_tex_array(SDL_Renderer *ren, SDL_Texture **arr, int count, int ksize) {
    for (int f = 0; f < count; ++f) {
        arr[f] = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, ksize, ksize);
        if (arr[f]) SDL_SetTextureScaleMode(arr[f], SDL_ScaleModeNearest);
    }
}

static void open_weights_window(Gui *g) {
    if (g->weights_win) { SDL_RaiseWindow(g->weights_win); return; }

    g->weights_win = SDL_CreateWindow("tinyDL - layer weights (live)", SDL_WINDOWPOS_UNDEFINED,
                                       SDL_WINDOWPOS_UNDEFINED, WEIGHTS_WIN_W, WEIGHTS_WIN_H, SDL_WINDOW_SHOWN);
    if (!g->weights_win) {
        fprintf(stderr, "SDL_CreateWindow (weights) failed: %s\n", SDL_GetError());
        return;
    }
    g->weights_ren = SDL_CreateRenderer(g->weights_win, -1, SDL_RENDERER_SOFTWARE);
    if (!g->weights_ren) {
        fprintf(stderr, "SDL_CreateRenderer (weights) failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(g->weights_win);
        g->weights_win = NULL;
        return;
    }
    SDL_SetRenderDrawBlendMode(g->weights_ren, SDL_BLENDMODE_BLEND);
    create_filter_tex_array(g->weights_ren, g->tex_conv1, CONV1_OUT, CONV1_K);
    create_filter_tex_array(g->weights_ren, g->tex_conv2, CONV2_OUT, CONV2_K);
    create_filter_tex_array(g->weights_ren, g->tex_conv3, CONV3_OUT, CONV3_K);
    create_filter_tex_array(g->weights_ren, g->tex_conv4, CONV4_OUT, CONV4_K);

    g->tex_fc1 = SDL_CreateTexture(g->weights_ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING,
                                    FC1_VIZ_COLS, FC1_VIZ_ROWS);
    if (g->tex_fc1) SDL_SetTextureScaleMode(g->tex_fc1, SDL_ScaleModeNearest);
    g->tex_fc2 = SDL_CreateTexture(g->weights_ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, FC2_COLS,
                                    FC2_ROWS);
    if (g->tex_fc2) SDL_SetTextureScaleMode(g->tex_fc2, SDL_ScaleModeNearest);

    g->weights_content_tex = SDL_CreateTexture(g->weights_ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_TARGET,
                                                WEIGHTS_CONTENT_W, WEIGHTS_CONTENT_H);
    if (g->weights_content_tex) {
        SDL_SetTextureScaleMode(g->weights_content_tex, SDL_ScaleModeNearest);
    } else {
        fprintf(stderr, "SDL_CreateTexture (weights zoom target) failed: %s - zoom will be unavailable\n",
                SDL_GetError());
    }
    g->weights_zoom = WEIGHTS_ZOOM_MIN;
}

/* One quadrant of the weights window: a caption plus a grid of normalized
 * filter/filter-map thumbnails for one conv layer. */
static void draw_filter_section(Gui *g, SDL_Rect box, SDL_Texture **label_slot, const char *caption,
                                 SDL_Texture **tex_array, int count, int cols, int ksize, const float *weights) {
    draw_panel(g->weights_ren, box);
    draw_text(g->weights_ren, label_slot, g->font, caption, box.x + 10, box.y + 8, (SDL_Color){COL_TEXT});

    int rows = (count + cols - 1) / cols;
    int pad = 10, gap = 6;
    int cell_w = (box.w - 2 * pad - (cols - 1) * gap) / cols;
    int cell_h = (box.h - 30 - pad - (rows - 1) * gap) / rows;
    int cell = cell_w < cell_h ? cell_w : cell_h;

    for (int f = 0; f < count; ++f) {
        int col = f % cols, row = f / cols;
        int x = box.x + pad + col * (cell + gap);
        int y = box.y + 30 + row * (cell + gap);

        fill_filter_texture(tex_array[f], weights + (size_t)f * ksize * ksize, ksize, ksize, NULL, NULL);

        SDL_Rect dst = {x, y, cell, cell};
        SDL_RenderCopy(g->weights_ren, tex_array[f], NULL, &dst);
        SDL_SetRenderDrawColor(g->weights_ren, COL_BORDER);
        SDL_RenderDrawRect(g->weights_ren, &dst);
    }
}

static void compute_mean_rms(const float *data, int n, float *out_mean, float *out_rms) {
    double sum = 0.0, sumsq = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
        sumsq += (double)data[i] * (double)data[i];
    }
    *out_mean = (float)(sum / n);
    *out_rms = (float)sqrt(sumsq / n);
}

/* fc1/fc2's weight is one (rows,cols) matrix, not a set of square filters -
 * rendered as a single heatmap scaled to fit the box while keeping its
 * aspect ratio (a weight matrix stretched to fill a differently-shaped box
 * would visually distort which rows/cols are near each other).
 *
 * Also prints min/max/mean/rms as text: fc1 in particular has no spatial
 * structure (its rows are pooled groups of flattened conv-output features,
 * its columns are output neurons, with no proximity relationship like an
 * image's pixels have), so a genuinely-updating heatmap of it still looks
 * like plain static to the eye - the per-frame min..max auto-contrast also
 * means the picture alone can't show the numbers actually drifting. The
 * numbers can. */
static void draw_fc_matrix(Gui *g, SDL_Rect box, SDL_Texture **label_slot, SDL_Texture **stats_slot,
                            const char *caption, SDL_Texture *tex, int rows, int cols, const float *data) {
    draw_panel(g->weights_ren, box);
    draw_text(g->weights_ren, label_slot, g->font, caption, box.x + 10, box.y + 8, (SDL_Color){COL_TEXT});

    float vmin, vmax, vmean, vrms;
    fill_filter_texture(tex, data, rows, cols, &vmin, &vmax);
    compute_mean_rms(data, rows * cols, &vmean, &vrms);
    char stats[96];
    snprintf(stats, sizeof(stats), "min % .5f  max % .5f  mean % .5f  rms %.5f", vmin, vmax, vmean, vrms);
    draw_text(g->weights_ren, stats_slot, g->font, stats, box.x + 10, box.y + box.h - 22, (SDL_Color){COL_TEXT_DIM});

    int pad = 10;
    int avail_w = box.w - 2 * pad;
    int avail_h = box.h - 56 - pad;
    float scale_w = (float)avail_w / cols;
    float scale_h = (float)avail_h / rows;
    float scale = scale_w < scale_h ? scale_w : scale_h;
    int disp_w = (int)(cols * scale);
    int disp_h = (int)(rows * scale);

    SDL_Rect dst = {box.x + pad + (avail_w - disp_w) / 2, box.y + 30 + (avail_h - disp_h) / 2, disp_w, disp_h};
    SDL_RenderCopy(g->weights_ren, tex, NULL, &dst);
    SDL_SetRenderDrawColor(g->weights_ren, COL_BORDER);
    SDL_RenderDrawRect(g->weights_ren, &dst);
}

static void render_weights_window(Gui *g, const TrainRecord *rec) {
    if (!g->weights_win) return;

    SDL_SetRenderDrawColor(g->weights_ren, COL_BG);
    SDL_RenderClear(g->weights_ren);

    for (int i = 0; i < NUM_WEIGHTS_TABS; ++i) {
        SDL_Rect tab = weights_tab_rect(i);
        int active = (g->weights_tab == i);
        if (active) {
            SDL_SetRenderDrawColor(g->weights_ren, COL_LOSS);
        } else {
            SDL_SetRenderDrawColor(g->weights_ren, COL_SURFACE);
        }
        SDL_RenderFillRect(g->weights_ren, &tab);
        SDL_SetRenderDrawColor(g->weights_ren, COL_BORDER);
        SDL_RenderDrawRect(g->weights_ren, &tab);
        SDL_Color label_col = active ? (SDL_Color){COL_TEXT} : (SDL_Color){COL_TEXT_DIM};
        draw_text(g->weights_ren, &g->weights_label_tex[WLBL_TAB0 + i], g->font, WEIGHTS_TABS[i].tab_label,
                  tab.x + 16, tab.y + 13, label_col);
    }

    const WeightsTab *t = &WEIGHTS_TABS[g->weights_tab];
    SDL_Rect grid_box = {20, WEIGHTS_TAB_Y + WEIGHTS_TAB_H + 16, WEIGHTS_CONTENT_W, WEIGHTS_CONTENT_H};

    /* Draw the tab's content at native size into the offscreen target - a
     * (0,0)-based box the same size as grid_box, since the content doesn't
     * need to know it's about to be cropped/zoomed rather than shown
     * directly. Falls back to drawing straight into the window (no zoom
     * available) if the target texture failed to create. */
    SDL_Texture *content = g->weights_content_tex;
    SDL_Rect content_box = {0, 0, WEIGHTS_CONTENT_W, WEIGHTS_CONTENT_H};
    SDL_Rect draw_box = content ? content_box : grid_box;
    if (content) SDL_SetRenderTarget(g->weights_ren, content);

    if (t->kind == TABKIND_CONV_GRID) {
        SDL_Texture **tex_arrays[4] = {g->tex_conv1, g->tex_conv2, g->tex_conv3, g->tex_conv4};
        const float *weights_by_tab[4] = {rec->conv1_weights, rec->conv2_filters, rec->conv3_filters,
                                           rec->conv4_filters};
        draw_filter_section(g, draw_box, &g->weights_label_tex[WLBL_CAPTION], t->caption,
                             tex_arrays[g->weights_tab], t->count, t->cols, t->ksize,
                             weights_by_tab[g->weights_tab]);
    } else {
        SDL_Texture *tex = (g->weights_tab == 4) ? g->tex_fc1 : g->tex_fc2;
        const float *data = (g->weights_tab == 4) ? rec->fc1_viz : rec->fc2_weights;
        char caption_buf[64];
        const char *caption = t->caption;
        if (g->weights_tab == 4) {
            /* Built from FC1_POOL rather than baked into the WEIGHTS_TABS
             * literal, so it can't drift out of sync with the actual pooling
             * the worker does the way a hardcoded string could. */
            snprintf(caption_buf, sizeof(caption_buf), "fc1 weight (1568->128, pooled %d rows/px)", FC1_POOL);
            caption = caption_buf;
        }
        draw_fc_matrix(g, draw_box, &g->weights_label_tex[WLBL_CAPTION], &g->weights_label_tex[WLBL_STATS], caption,
                       tex, t->count, t->cols, data);
    }

    if (content) {
        SDL_SetRenderTarget(g->weights_ren, NULL);

        /* Centered crop: at zoom 1 this is the whole texture (no-op); above
         * that it's a shrinking, centered window into it, scaled back up to
         * fill grid_box - the crop shrinks, not the box, so this is a
         * magnifying glass, not a shrink-then-stretch. */
        int crop_w = (int)(WEIGHTS_CONTENT_W / g->weights_zoom);
        int crop_h = (int)(WEIGHTS_CONTENT_H / g->weights_zoom);
        if (crop_w < 1) crop_w = 1;
        if (crop_h < 1) crop_h = 1;
        SDL_Rect src = {(WEIGHTS_CONTENT_W - crop_w) / 2, (WEIGHTS_CONTENT_H - crop_h) / 2, crop_w, crop_h};
        SDL_RenderCopy(g->weights_ren, content, &src, &grid_box);

        /* Top-right corner, clear of both the caption (top-left, inside the
         * zoomed content) and the fc stats line (bottom-left, also inside
         * the zoomed content and liable to scroll out of view once zoomed) -
         * this is drawn straight onto the window, after the crop blit, so it
         * always stays put regardless of zoom/pan. */
        char zoom_buf[48];
        int zoomed = g->weights_zoom > WEIGHTS_ZOOM_MIN + 0.01f;
        if (zoomed) {
            snprintf(zoom_buf, sizeof(zoom_buf), "zoom %.1fx (scroll=zoom, 0=reset)", g->weights_zoom);
        } else {
            snprintf(zoom_buf, sizeof(zoom_buf), "scroll to zoom in");
        }
        /* Backing rect so this stays legible regardless of what the zoomed
         * image happens to look like underneath - there's no letterboxed
         * margin to put chrome in since the content fills the whole box. */
        SDL_Rect zoom_bg = {grid_box.x + grid_box.w - 304, grid_box.y + 4, 294, 22};
        SDL_SetRenderDrawColor(g->weights_ren, 13, 15, 20, 215);
        SDL_RenderFillRect(g->weights_ren, &zoom_bg);
        draw_text(g->weights_ren, &g->weights_label_tex[WLBL_ZOOM], g->font, zoom_buf, zoom_bg.x + 6, zoom_bg.y + 4,
                  zoomed ? (SDL_Color){COL_LOSS} : (SDL_Color){COL_TEXT_DIM});
    }

    SDL_RenderPresent(g->weights_ren);
}

static void draw_digit(Gui *g, SDL_Rect box, const unsigned char *pixels255, int true_label, int pred_label) {
    int correct = true_label == pred_label;
    draw_panel(g->ren, box);

    void *pixels;
    int pitch;
    SDL_LockTexture(g->digit_tex, NULL, &pixels, &pitch);
    for (int y = 0; y < TRAIN_IMG_SIZE; ++y) {
        Uint32 *row = (Uint32 *)((Uint8 *)pixels + y * pitch);
        for (int x = 0; x < TRAIN_IMG_SIZE; ++x) {
            Uint8 v = pixels255[y * TRAIN_IMG_SIZE + x];
            row[x] = (0xFFu << 24) | (v << 16) | (v << 8) | v;
        }
    }
    SDL_UnlockTexture(g->digit_tex);

    int size = box.h - 70;
    SDL_Rect dst = {box.x + (box.w - size) / 2, box.y + 16, size, size};
    SDL_SetRenderDrawColor(g->ren, correct ? COL_GOOD : COL_BAD);
    SDL_Rect ring = {dst.x - 4, dst.y - 4, dst.w + 8, dst.h + 8};
    SDL_RenderFillRect(g->ren, &ring);
    SDL_RenderCopy(g->ren, g->digit_tex, NULL, &dst);

    /* Class names run a lot longer than MNIST's bare digits ("Ankle boot" vs
     * "9"), so this is deliberately terser than the old "true %d pred %d"
     * form - it has to fit the same ~300px-wide panel. */
    char buf[64];
    snprintf(buf, sizeof(buf), "%s -> %s  %s", CLASS_NAMES[true_label % 10], CLASS_NAMES[pred_label % 10],
              correct ? "ok" : "WRONG");
    draw_text(g->ren, &g->label_tex[LBL_DIGIT_CAPTION], g->font, buf, box.x + 14, dst.y + dst.h + 16,
              (SDL_Color){correct ? COL_GOOD : COL_BAD});
}

static int pump_events(Gui *g) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_QUIT:
            return 0;
        case SDL_KEYDOWN:
            if (e.key.keysym.sym == SDLK_ESCAPE) return 0;
            if (g->weights_win && e.key.windowID == SDL_GetWindowID(g->weights_win) &&
                e.key.keysym.sym == SDLK_0) {
                g->weights_zoom = WEIGHTS_ZOOM_MIN;
            }
            break;
        case SDL_MOUSEWHEEL:
            if (g->weights_win && e.wheel.windowID == SDL_GetWindowID(g->weights_win)) {
                float factor = e.wheel.y > 0 ? 1.15f : e.wheel.y < 0 ? (1.0f / 1.15f) : 1.0f;
                g->weights_zoom *= factor;
                if (g->weights_zoom < WEIGHTS_ZOOM_MIN) g->weights_zoom = WEIGHTS_ZOOM_MIN;
                if (g->weights_zoom > WEIGHTS_ZOOM_MAX) g->weights_zoom = WEIGHTS_ZOOM_MAX;
            }
            break;
        case SDL_MOUSEMOTION:
            if (e.motion.windowID == SDL_GetWindowID(g->win)) {
                g->mouse_x = e.motion.x;
                g->mouse_y = e.motion.y;
            }
            break;
        case SDL_MOUSEBUTTONDOWN:
            if (e.button.windowID == SDL_GetWindowID(g->win)) {
                SDL_Point p = {e.button.x, e.button.y};
                if (SDL_PointInRect(&p, &WEIGHTS_BUTTON)) {
                    open_weights_window(g);
                } else if (!g->started && SDL_PointInRect(&p, &START_BUTTON)) {
                    send_train_command(g);
                    g->started = 1;
                } else if (g->started && SDL_PointInRect(&p, &RESTART_BUTTON)) {
                    send_train_command(g);
                    /* Clear the chart and drop the last record so the UI
                     * reads as "waiting for the new run" rather than
                     * silently continuing to show the old one's data until
                     * the worker's first fresh record arrives. */
                    g->has_record = 0;
                    g->hist_count = 0;
                    g->hist_head = 0;
                }
            } else if (g->weights_win && e.button.windowID == SDL_GetWindowID(g->weights_win)) {
                SDL_Point p = {e.button.x, e.button.y};
                for (int i = 0; i < NUM_WEIGHTS_TABS; ++i) {
                    SDL_Rect tab = weights_tab_rect(i);
                    if (SDL_PointInRect(&p, &tab)) {
                        if (g->weights_tab != i) g->weights_zoom = WEIGHTS_ZOOM_MIN; /* don't carry zoom to a different layer */
                        g->weights_tab = i;
                        break;
                    }
                }
            }
            break;
        case SDL_WINDOWEVENT:
            if (e.window.event == SDL_WINDOWEVENT_CLOSE) {
                if (g->weights_win && e.window.windowID == SDL_GetWindowID(g->weights_win)) {
                    close_weights_window(g);
                } else if (e.window.windowID == SDL_GetWindowID(g->win)) {
                    return 0;
                }
            }
            break;
        }
    }
    return 1;
}

static void render_frame(Gui *g, const TrainRecord *rec) {
    SDL_SetRenderDrawColor(g->ren, COL_BG);
    SDL_RenderClear(g->ren);

    SDL_Rect arch_box = {20, 20, 1110, 90};
    draw_architecture(g, arch_box);

    SDL_Point mouse = {g->mouse_x, g->mouse_y};
    SDL_Rect chart_box = {20, 130, 790, 550};
    if (g->has_record) {
        draw_chart(g, chart_box);
    } else {
        draw_start_panel(g, chart_box, SDL_PointInRect(&mouse, &START_BUTTON));
    }

    SDL_Rect digit_box = {830, 130, 300, 230};
    if (g->has_record) {
        draw_digit(g, digit_box, rec->pixels, rec->true_label, rec->pred_label);
    } else {
        draw_panel(g->ren, digit_box);
    }

    draw_weights_button(g, WEIGHTS_BUTTON, SDL_PointInRect(&mouse, &WEIGHTS_BUTTON));
    if (g->has_record) render_weights_window(g, rec);

    SDL_Rect info_box = {830, 530, 300, 150};
    draw_panel(g->ren, info_box);
    char line[96];
    if (g->has_record) {
        snprintf(line, sizeof(line), "epoch   %d / %d", rec->epoch, rec->epochs);
        draw_text(g->ren, &g->label_tex[LBL_EPOCH], g->font_big, line, info_box.x + 14, info_box.y + 12,
                  (SDL_Color){COL_TEXT});
        snprintf(line, sizeof(line), "batch   %d / %d", rec->batch, rec->batch_count);
        draw_text(g->ren, &g->label_tex[LBL_BATCH], g->font, line, info_box.x + 14, info_box.y + 46,
                  (SDL_Color){COL_TEXT_DIM});
        snprintf(line, sizeof(line), "loss %.4f   acc %.1f%%", rec->loss, rec->acc * 100.0f);
        draw_text(g->ren, &g->label_tex[LBL_LOSS], g->font, line, info_box.x + 14, info_box.y + 68,
                  (SDL_Color){COL_LOSS});
        snprintf(line, sizeof(line), "elapsed %.1fs", rec->elapsed);
        draw_text(g->ren, &g->label_tex[LBL_ELAPSED], g->font, line, info_box.x + 14, info_box.y + 96,
                  (SDL_Color){COL_TEXT_DIM});
    } else {
        draw_text(g->ren, &g->label_tex[LBL_EPOCH], g->font_big, "not started", info_box.x + 14, info_box.y + 12,
                  (SDL_Color){COL_TEXT_DIM});
    }
    if (g->started) draw_restart_button(g, SDL_PointInRect(&mouse, &RESTART_BUTTON));

    SDL_RenderPresent(g->ren);
}

int main(void) {
    Gui gui = {0}; /* label_tex[]/digit_tex etc. must start NULL - see the lazy-create checks in draw_text() */
    if (gui_init(&gui) != 0) return 1;

    if (mkfifo(TRAIN_FIFO_PATH, 0600) != 0 && errno != EEXIST) {
        fprintf(stderr, "train_viewer: mkfifo(%s) failed: %s\n", TRAIN_FIFO_PATH, strerror(errno));
        gui_shutdown(&gui);
        return 1;
    }
    fprintf(stderr, "train_viewer: waiting for train_worker to connect to %s ...\n", TRAIN_FIFO_PATH);
    FILE *fifo = fopen(TRAIN_FIFO_PATH, "rb"); /* blocks until the worker opens it for writing */
    if (!fifo) {
        fprintf(stderr, "train_viewer: could not open fifo for reading: %s\n", strerror(errno));
        gui_shutdown(&gui);
        return 1;
    }
    int data_fd = fileno(fifo);
    /* Training doesn't start until the user clicks Start, so there's nothing
     * to read yet - non-blocking lets the loop below keep pumping events and
     * rendering the idle UI instead of stalling on the first record. */
    fcntl(data_fd, F_SETFL, O_NONBLOCK);

    /* Second FIFO, this direction viewer->worker - see the comment on
     * TRAIN_CMD_FIFO_PATH in train_record.h. open() blocks until the
     * worker's read end is open, which by this point it already is (the
     * worker opens it right after the data FIFO handshake above, before
     * doing anything else). */
    if (mkfifo(TRAIN_CMD_FIFO_PATH, 0600) != 0 && errno != EEXIST) {
        fprintf(stderr, "train_viewer: mkfifo(%s) failed: %s\n", TRAIN_CMD_FIFO_PATH, strerror(errno));
        fclose(fifo);
        gui_shutdown(&gui);
        return 1;
    }
    gui.cmd_fd = open(TRAIN_CMD_FIFO_PATH, O_WRONLY);
    if (gui.cmd_fd < 0) {
        fprintf(stderr, "train_viewer: could not open command fifo for writing: %s\n", strerror(errno));
        fclose(fifo);
        gui_shutdown(&gui);
        return 1;
    }
    fprintf(stderr, "train_viewer: worker connected. Click Start to begin training.\n");

    TrainRecord rec;
    unsigned char assemble[sizeof(TrainRecord)];
    size_t have = 0;
    int running = 1;

    while (running) {
        if (!pump_events(&gui)) break;

        /* A TrainRecord (now over 100KB with the weight-visualization
         * fields) doesn't necessarily arrive in one read() even though the
         * worker writes it in one fwrite() - the pipe can hand it over in
         * pieces - so this accumulates into `assemble` across frames until a
         * full record's worth of bytes has shown up. */
        ssize_t n = read(data_fd, assemble + have, sizeof(TrainRecord) - have);
        if (n > 0) {
            have += (size_t)n;
            if (have == sizeof(TrainRecord)) {
                memcpy(&rec, assemble, sizeof(TrainRecord));
                have = 0;
                gui.has_record = 1;
                gui_push_point(&gui, rec.loss);
            }
        }
        /* n == 0 (worker closed its end) or n < 0 with EAGAIN (nothing
         * available yet) both just mean "no new record this frame" - either
         * way the last-known `rec` (if any) is still fine to keep showing. */

        render_frame(&gui, &rec);
        SDL_Delay(8); /* avoid busy-spinning the CPU now that nothing blocks this loop */
    }

    fclose(fifo);
    if (gui.cmd_fd >= 0) close(gui.cmd_fd);
    gui_shutdown(&gui);
    return 0;
}
