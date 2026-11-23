/* SDL2 viewer half of the live-training UI. Two screens:
 *   - setup: a form where every training parameter (dataset, epochs, batch
 *     size, learning rate, seed, the conv/fc layer stack, ...) is typed in.
 *     Start sends the validated TrainConfig to train_worker.
 *   - training: the live dashboard (loss curve, current sample, architecture,
 *     per-layer weights window) driven by the messages the worker streams back.
 * Deliberately has no CUDA dependency at all (see the big comment at the top
 * of train_worker.c for why this is a separate process from the trainer).
 * Run both together with `make train_gui`. */
#include <ctype.h>
#include <dirent.h>
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

#include <SDL.h>
#include <SDL_ttf.h>

#include "train_record.h"

#define WIN_W 1150
#define WIN_H 700
#define HISTORY_LEN 600
#define LOSS_CLIP 2.5f
#define MAX_LABEL_W 440
#define MAX_LABEL_H 32
#define NUM_LABEL_SLOTS 800  /* main window text, one slot per draw call per frame */
#define NUM_WLABEL_SLOTS 64  /* weights window text */

#define SCREEN_SETUP 0
#define SCREEN_RUN 1
#define SCREEN_PRED 2 /* the predictions page */

/* Hit-tested by pump_events() and drawn by the render functions from the same
 * constants so the two can never drift apart. */
static const SDL_Rect WEIGHTS_BUTTON = {830, 370, 300, 70};
static const SDL_Rect PRED_BUTTON = {830, 450, 300, 70};
static const SDL_Rect BTN_PRED_BACK = {20, 20, 110, 36};
static const SDL_Rect BTN_PRED_PREV = {1018, 20, 54, 36};
static const SDL_Rect BTN_PRED_NEXT = {1076, 20, 54, 36};
static const SDL_Rect BTN_VIEW_PRED = {648, 646, 200, 34}; /* settings screen: jump to the predictions page */
static const SDL_Rect RESTART_BUTTON = {830, 644, 146, 36};
static const SDL_Rect SETTINGS_BUTTON = {984, 644, 146, 36};

static const SDL_Rect PANEL_TRAIN = {20, 20, 370, 660};
static const SDL_Rect PANEL_GRAPH = {410, 20, 720, 546};
static const SDL_Rect GRAPH_VIEW = {411, 66, 718, 499}; /* the node canvas, below the palette row */
static const SDL_Rect BTN_ARRANGE = {976, 28, 64, 30};
static const SDL_Rect BTN_PRESETS = {1046, 28, 76, 30}; /* opens the preset menu below it */
/* The problem comes first, and is always the user's choice: every decision below it (the task, the blocks on
 * offer, the presets, the datasets that fit, the training defaults) follows from it. */
#define PROBLEM_CLASSIFY 0 /* one class per image */
#define PROBLEM_SEGMENT 1  /* one class per pixel: needs a dataset with masks */
#define PROBLEM_GENERATE 2 /* a VAE: reconstruct and generate images; needs no labels */
#define NUM_PRESETS 4      /* ids: 0 cnn, 1 resnet (classification) | 2 unet (segmentation) | 3 vae (generation) */
static const char *PRESET_NAMES[NUM_PRESETS] = {"cnn", "resnet", "unet", "vae"};
static const SDL_Rect PROBLEM_BTN[3] = {{36, 21, 107, 31}, {149, 21, 107, 31}, {262, 21, 104, 31}};
static SDL_Rect preset_rect(int i) {
    SDL_Rect r = {BTN_PRESETS.x - 54, BTN_PRESETS.y + BTN_PRESETS.h + 4 + i * 30, BTN_PRESETS.w + 54, 28};
    return r;
}
/* Blocks offered per problem: the VAE's view / vae blocks make no sense for classification, and a generator
 * has no use for global average pooling. */
static const int PALETTE_CLS[8] = {BLK_CONV, BLK_RELU, BLK_FLATTEN, BLK_LINEAR, BLK_ADD, BLK_BATCHNORM, BLK_MAXPOOL, BLK_GAP};
static const int PALETTE_SEG[7] = {BLK_CONV, BLK_RELU, BLK_ADD, BLK_BATCHNORM, BLK_MAXPOOL, BLK_UPSAMPLE, BLK_CONCAT};
static const int PALETTE_GEN[11] = {BLK_CONV,    BLK_RELU,   BLK_FLATTEN,  BLK_LINEAR,   BLK_ADD,     BLK_BATCHNORM,
                                    BLK_MAXPOOL, BLK_UPSAMPLE, BLK_CONCAT, BLK_RESHAPE, BLK_REPARAM};

static SDL_Rect palette_rect(int i) {
    SDL_Rect r = {418 + i * 46, 28, 43, 30};
    return r;
}
static const SDL_Rect BTN_START = {880, 582, 250, 98};
static const SDL_Rect BTN_BACK = {410, 646, 230, 34};

static const char *FASHION_NAMES[10] = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                         "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot"};

#define COL_BG 11, 13, 19, 255
#define COL_SURFACE 20, 23, 32, 255
#define COL_SURFACE2 28, 32, 44, 255 /* raised: buttons, chips, cards' bodies */
#define COL_BORDER 40, 45, 60, 255
#define COL_BORDER_HI 62, 70, 94, 255
#define COL_GRID 28, 32, 44, 255
#define COL_TEXT 232, 236, 244, 255
#define COL_TEXT_DIM 143, 151, 170, 255
#define COL_LOSS 96, 140, 255, 255 /* the accent colour */
#define COL_GOOD 62, 200, 128, 255
#define COL_BAD 245, 101, 112, 255

/* -- form fields --------------------------------------------------------- */

enum { FK_INT, FK_FLOAT, FK_NAME, FK_CHOICE /* picked from a dropdown, never typed */ };

typedef struct Field {
    char text[40];
    int kind;
    SDL_Rect rect;
    int visible;
} Field;

enum { F_DATASET, F_EPOCHS, F_BATCH, F_LR, F_OPT, F_MOMENTUM, F_NORMALIZE, F_AUGMENT, F_SEED, F_POOL, F_VIZ, NUM_GENERAL_FIELDS };
#define FIELD_PITCH 36 /* vertical distance between the general fields */

/* The model editor: every node may carry up to 4 parameter fields, which live
 * in the same fields[] array as the general ones (index F_NODE(node, j)). A
 * conv node uses j = 0 out channels, 1 kernel, 2 stride, 3 padding; fc: j = 0 out;
 * maxpool: j = 0 window, 1 stride. */
#define MAX_NODES 56
#define F_NODE(i, j) (NUM_GENERAL_FIELDS + (i)*4 + (j))
#define NUM_FIELDS (NUM_GENERAL_FIELDS + MAX_NODES * 4)
#define NODE_INPUT 0
#define NODE_OUTPUT 15 /* node types are the BLK_* values plus these two */
#define NODE_W 112
#define NODE_GAP 24 /* horizontal space left between auto-placed nodes */
#define DRAG_NONE 0
#define DRAG_NODE 1 /* moving a node */
#define DRAG_PAN 2  /* panning the canvas */
#define DRAG_LINK 3 /* pulling a wire out of a node's output dot */
#define DRAG_PAL 4  /* carrying a palette block */

/* A node reads the outputs of the nodes its input wires point at (in[0], and for an add node also
 * in[1]); an output dot may feed any number of nodes. That is the whole graph. */
typedef struct Node {
    int used;
    int type;
    int x, y;    /* canvas coordinates (before panning) */
    int in[2];   /* source node of each input port, or -1 */
} Node;
#define FIELD_MAX_CHARS 31
#define MAX_DATASETS 12

/* What the viewer learns about a data/<name>/ folder before any training:
 * image size and sample count from the IDX header, class count from the label
 * file (or classes.txt, which also names the classes), and optionally the
 * settings that work well for it from defaults.txt ("key value" lines:
 * epochs, batch_size, lr, momentum, normalize, augment). */
typedef struct DatasetInfo {
    char name[32]; /* must stay first: the list is sorted with a plain strcmp */
    int count, rows, cols, channels, n_classes;
    int task; /* TASK_SEGMENT if the folder has train-masks-idx3-ubyte, else TASK_CLASSIFY */
    int has_names;
    char class_names[MAX_CLASSES][CLASS_NAME_LEN];
    int def_epochs, def_batch, def_normalize, def_augment; /* -1 = not given */
    float def_lr, def_momentum;                           /* < 0 = not given */
} DatasetInfo;

typedef struct Gui {
    SDL_Window *win;
    SDL_Renderer *ren;
    SDL_Texture *frame_tex; /* the whole UI is drawn here at WIN_W x WIN_H, then scaled once to the window */
    TTF_Font *font;
    TTF_Font *font_big;
    TTF_Font *font_med; /* same size as font, heavier weight: titles */
    TTF_Font *font_small;
    SDL_Texture *digit_tex; /* the current training sample, digit_w x digit_h */
    int digit_w, digit_h, digit_c;
    SDL_Texture *mask_tex[2]; /* segmentation: true / predicted mask of the shown sample, RGBA, digit_w x digit_h */
    int preset_open;          /* the preset menu under the presets button is showing */
    int problem;              /* PROBLEM_CLASSIFY or PROBLEM_GENERATE: chosen first, decides the task, palette and presets */

    /* the latest MSG_PREDICTIONS: a header followed by n records, kept so the page can be opened any time */
    PredHeader pred_hdr;
    unsigned char *pred_data;
    int pred_page;
    int pred_return; /* screen the predictions page's back button goes to */
    SDL_Texture *pred_tex[96];
    GenHeader gen_hdr; /* the latest MSG_GENERATED (a VAE run): n_recon == 0 until the first epoch ends */
    unsigned char *gen_data;
    int pred_tex_w, pred_tex_h;
    SDL_Texture *label_tex[NUM_LABEL_SLOTS];
    int slot; /* next free label_tex slot this frame */
    float loss_hist[HISTORY_LEN];
    int hist_count;
    int hist_head;
    int mouse_x, mouse_y;

    int cmd_fd; /* write end of the viewer->worker command FIFO */
    int screen;
    int run_active;  /* a Start was sent and not rejected since */
    int worker_gone; /* data FIFO hit EOF / protocol error */
    char worker_msg[256];

    /* setup form */
    Field fields[NUM_FIELDS];
    int focus; /* index into fields, or -1 */
    Node nodes[MAX_NODES];
    /* The node canvas has its own coordinate space ("view space"): view = (screen - canvas origin) / zoom, and a
     * node at (x, y) sits at (pan + x, pan + y) in it. The canvas is drawn at native size into canvas_tex and then
     * scaled to the screen, so zooming scales text, wires and fields together. */
    float pan_x, pan_y, zoom;
    float drag_fx, drag_fy; /* sub-pixel remainders while dragging at a zoom other than 1 */
    SDL_Texture *canvas_tex;
    int drag, drag_node, drag_type, drag_moved, press_x, press_y;
    int chain[MAX_BLOCKS];     /* the node behind each block of form_cfg, in block (evaluation) order */
    int chain_len;
    int node_block[MAX_NODES]; /* node -> index into form_cfg.blocks, or -1 if not on the path */
    char node_name[MAX_NODES][12];
    int form_err_node;
    DatasetInfo ds[MAX_DATASETS]; /* folders under data/ that hold a usable dataset */
    int n_datasets;
    int dropdown_open; /* dataset dropdown list is showing */
    TrainConfig form_cfg;
    NetShapes form_shapes;
    int form_valid;
    char form_err[200];
    int form_err_field;

    /* current run, as described by the worker */
    int has_model;
    ModelInfo model;
    TrainConfig run_cfg; /* what the dashboard is showing: set on Start/Restart, confirmed by MSG_MODEL */
    NetShapes run_shapes;
    float *weights;
    int weights_valid;
    int has_record;
    Progress prog;
    unsigned char *sample; /* pixels of prog's sample, run_cfg.img_h * img_w */
    DatasetInfo run_ds;    /* the dataset the dashboard is showing, for class names */

    /* Opened on demand by clicking the "view layer weights" button. Textures
     * and label caches belong to weights_ren specifically (SDL textures
     * aren't shared across renderers), so they get their own arrays instead
     * of reusing anything from the main window. All NULL when closed. */
    SDL_Window *weights_win;
    SDL_Renderer *weights_ren;
    int weights_tab; /* index into model.layers; persists across close/reopen */
    int wslot;
    SDL_Texture *weights_label_tex[NUM_WLABEL_SLOTS];
    float yaw, pitch;   /* 3D camera orbit, radians */
    float zoom3d;
} Gui;

#define ZOOM_MIN 0.35f
#define ZOOM_MAX 2.0f
#define CANVAS_TEX_W 2060 /* GRAPH_VIEW.w / ZOOM_MIN, rounded up */
#define CANVAS_TEX_H 1430

/* Screen position -> canvas view space. */
static SDL_Point to_view(const Gui *g, int sx, int sy) {
    SDL_Point v = {(int)floorf((float)(sx - GRAPH_VIEW.x) / g->zoom), (int)floorf((float)(sy - GRAPH_VIEW.y) / g->zoom)};
    return v;
}
/* The part of view space that is on screen: (0,0) to the canvas size / zoom. */
static SDL_Rect view_rect(const Gui *g) {
    SDL_Rect r = {0, 0, (int)ceilf((float)GRAPH_VIEW.w / g->zoom), (int)ceilf((float)GRAPH_VIEW.h / g->zoom)};
    return r;
}

static int palette_count(const Gui *g) { return g->problem == PROBLEM_GENERATE ? 11 : (g->problem == PROBLEM_SEGMENT ? 7 : 8); }
static int palette_type(const Gui *g, int i) {
    return g->problem == PROBLEM_GENERATE ? PALETTE_GEN[i] : (g->problem == PROBLEM_SEGMENT ? PALETTE_SEG[i] : PALETTE_CLS[i]);
}
static int preset_count(const Gui *g) { return g->problem == PROBLEM_CLASSIFY ? 2 : 1; }
static int preset_at(const Gui *g, int i) { return g->problem == PROBLEM_CLASSIFY ? i : (g->problem == PROBLEM_SEGMENT ? 2 : 3); }

/* Segmentation needs masks, so only datasets that have them are offered under that problem. */
static int ds_fits(const Gui *g, const DatasetInfo *d) { return g->problem != PROBLEM_SEGMENT || d->task == TASK_SEGMENT; }
/* Indices into g->ds of the datasets the dropdown offers; returns how many. */
static int ds_visible(const Gui *g, int *idx) {
    int n = 0;
    for (int i = 0; i < g->n_datasets; ++i)
        if (ds_fits(g, &g->ds[i])) idx[n++] = i;
    return n;
}

static void close_weights_window(Gui *g);

/* The UI is laid out for WIN_W x WIN_H. SDL's logical size scales that picture to fit whatever the window or
 * screen is (keeping the proportions, black bars at the sides) and converts mouse positions back, so the same
 * layout works windowed, resized and fullscreen. fullscreen = start covering the whole screen. */
static int gui_init(Gui *g, int fullscreen) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return -1;
    }
    if (TTF_Init() != 0) {
        fprintf(stderr, "TTF_Init failed: %s\n", TTF_GetError());
        return -1;
    }
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear"); /* smooth text when the picture is scaled up */
    g->win = SDL_CreateWindow("tinyDL - training", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIN_W, WIN_H,
                               SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | (fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0));
    if (!g->win) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return -1;
    }
    g->ren = SDL_CreateRenderer(g->win, -1, SDL_RENDERER_SOFTWARE);
    SDL_RenderSetLogicalSize(g->ren, WIN_W, WIN_H);
    SDL_SetRenderDrawBlendMode(g->ren, SDL_BLENDMODE_BLEND);
    /* Drawing the 1-pixel lines and shadows directly at a non-integer scale leaves them striped; drawing at the
     * native size into a texture and scaling that with a bilinear filter keeps them smooth. */
    g->frame_tex = SDL_CreateTexture(g->ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_TARGET, WIN_W, WIN_H);
    if (g->frame_tex) SDL_SetTextureScaleMode(g->frame_tex, SDL_ScaleModeLinear);

    /* A proportional UI font reads more modern than a monospace one; the first of each list that
     * exists is used (Ubuntu ships with most desktops here, DejaVu nearly everywhere). */
    static const char *regular[] = {"/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", NULL};
    static const char *medium[] = {"/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf",
                                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                   "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", NULL};
    for (int i = 0; regular[i] && !g->font; ++i) {
        g->font = TTF_OpenFont(regular[i], 15);
        g->font_small = TTF_OpenFont(regular[i], 12);
        g->font_big = TTF_OpenFont(medium[i], 22);
        g->font_med = TTF_OpenFont(medium[i], 15);
        if (!g->font || !g->font_small || !g->font_big || !g->font_med) {
            if (g->font) TTF_CloseFont(g->font);
            if (g->font_small) TTF_CloseFont(g->font_small);
            if (g->font_big) TTF_CloseFont(g->font_big);
            if (g->font_med) TTF_CloseFont(g->font_med);
            g->font = g->font_small = g->font_big = g->font_med = NULL;
        }
    }
    if (!g->font) {
        fprintf(stderr, "no usable UI font found: %s\n", TTF_GetError());
        return -1;
    }

    return 0;
}

static void gui_shutdown(Gui *g) {
    close_weights_window(g);
    if (g->digit_tex) SDL_DestroyTexture(g->digit_tex);
    for (int k = 0; k < 2; ++k)
        if (g->mask_tex[k]) SDL_DestroyTexture(g->mask_tex[k]);
    for (int i = 0; i < NUM_LABEL_SLOTS; ++i)
        if (g->label_tex[i]) SDL_DestroyTexture(g->label_tex[i]);
    if (g->font) TTF_CloseFont(g->font);
    if (g->font_big) TTF_CloseFont(g->font_big);
    if (g->font_med) TTF_CloseFont(g->font_med);
    if (g->font_small) TTF_CloseFont(g->font_small);
    if (g->canvas_tex) SDL_DestroyTexture(g->canvas_tex);
    if (g->frame_tex) SDL_DestroyTexture(g->frame_tex);
    if (g->ren) SDL_DestroyRenderer(g->ren);
    if (g->win) SDL_DestroyWindow(g->win);
    free(g->weights);
    free(g->sample);
    free(g->pred_data);
    free(g->gen_data);
    for (int i = 0; i < 96; ++i)
        if (g->pred_tex[i]) SDL_DestroyTexture(g->pred_tex[i]);
    TTF_Quit();
    SDL_Quit();
}

static void gui_push_point(Gui *g, float loss) {
    g->loss_hist[g->hist_head] = loss;
    g->hist_head = (g->hist_head + 1) % HISTORY_LEN;
    if (g->hist_count < HISTORY_LEN) g->hist_count++;
}

static void gui_clear_history(Gui *g) {
    g->hist_count = 0;
    g->hist_head = 0;
}

/* `slot` is the address of one persistent per-label SDL_Texture* (a slot in
 * Gui::label_tex for the main window, or in Gui::weights_label_tex for the
 * weights window) so re-rendering a line of text every frame updates that
 * texture's pixels in place instead of doing a fresh SDL_CreateTexture/
 * DestroyTexture - that churn once exhausted the GPU resources of this
 * renderer/driver mid-run. Takes the renderer explicitly so the same function
 * draws into either window. Each slot may be used at most once per frame (the
 * renderer queues draws, so a second upload would overwrite the first). */
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

/* Main-window text: takes the next free label slot of this frame. */
static void txt(Gui *g, TTF_Font *font, const char *s, int x, int y, SDL_Color color) {
    if (!s[0] || g->slot >= NUM_LABEL_SLOTS) return;
    draw_text(g->ren, &g->label_tex[g->slot++], font, s, x, y, color);
}

static void wtxt(Gui *g, const char *s, int x, int y, SDL_Color color) {
    if (!s[0] || g->wslot >= NUM_WLABEL_SLOTS) return;
    draw_text(g->weights_ren, &g->weights_label_tex[g->wslot++], g->font, s, x, y, color);
}

/* -- shapes: rounded rectangles, soft shadows, smooth circles -------------- */

static void set_color(SDL_Renderer *r, SDL_Color c) { SDL_SetRenderDrawColor(r, c.r, c.g, c.b, c.a); }

static SDL_Color col_mix(SDL_Color a, SDL_Color b, float t) {
    SDL_Color c = {(Uint8)(a.r + (b.r - a.r) * t), (Uint8)(a.g + (b.g - a.g) * t), (Uint8)(a.b + (b.b - a.b) * t),
                   (Uint8)(a.a + (b.a - a.a) * t)};
    return c;
}

/* Filled rounded rectangle; the corner pixels get partial alpha so the curve is smooth. */
static void fill_rrect(SDL_Renderer *r, SDL_Rect rc, int rad, SDL_Color c) {
    if (rc.w <= 0 || rc.h <= 0) return;
    if (rad * 2 > rc.h) rad = rc.h / 2;
    if (rad * 2 > rc.w) rad = rc.w / 2;
    set_color(r, c);
    SDL_Rect mid = {rc.x, rc.y + rad, rc.w, rc.h - 2 * rad};
    if (mid.h > 0) SDL_RenderFillRect(r, &mid);
    for (int i = 0; i < rad; ++i) {
        float dy = (float)rad - (float)i - 0.5f;
        float inset = (float)rad - sqrtf((float)rad * (float)rad - dy * dy);
        int ix = (int)inset;
        float cover = 1.0f - (inset - (float)ix); /* how much of the edge pixel is inside the curve */
        int x0 = rc.x + ix + 1, x1 = rc.x + rc.w - ix - 2;
        int yt = rc.y + i, yb = rc.y + rc.h - 1 - i;
        if (x1 >= x0) {
            SDL_RenderDrawLine(r, x0, yt, x1, yt);
            SDL_RenderDrawLine(r, x0, yb, x1, yb);
        }
        SDL_SetRenderDrawColor(r, c.r, c.g, c.b, (Uint8)(c.a * cover));
        SDL_RenderDrawPoint(r, rc.x + ix, yt);
        SDL_RenderDrawPoint(r, rc.x + rc.w - 1 - ix, yt);
        SDL_RenderDrawPoint(r, rc.x + ix, yb);
        SDL_RenderDrawPoint(r, rc.x + rc.w - 1 - ix, yb);
        set_color(r, c);
    }
}

/* Rounded rectangle with a 1px border: border colour underneath, fill inset by one. */
static void rrect_bordered(SDL_Renderer *r, SDL_Rect rc, int rad, SDL_Color fill, SDL_Color border) {
    fill_rrect(r, rc, rad, border);
    SDL_Rect in = {rc.x + 1, rc.y + 1, rc.w - 2, rc.h - 2};
    fill_rrect(r, in, rad > 1 ? rad - 1 : 0, fill);
}

/* A few translucent, growing copies under a card: cheap soft shadow. */
static void draw_shadow(SDL_Renderer *r, SDL_Rect rc, int rad) {
    for (int k = 5; k >= 1; --k) {
        SDL_Rect s = {rc.x - k, rc.y - k + 3, rc.w + 2 * k, rc.h + 2 * k};
        fill_rrect(r, s, rad + k, (SDL_Color){0, 0, 0, 16});
    }
}

/* Filled circle with a smooth edge. */
static void fill_circle_aa(SDL_Renderer *r, int cx, int cy, int rad, SDL_Color c) {
    set_color(r, c);
    for (int dy = -rad; dy <= rad; ++dy) {
        float half = sqrtf((float)(rad * rad - dy * dy) + 0.25f);
        int ix = (int)half;
        float frac = half - (float)ix;
        SDL_RenderDrawLine(r, cx - ix, cy + dy, cx + ix, cy + dy);
        SDL_SetRenderDrawColor(r, c.r, c.g, c.b, (Uint8)(c.a * frac));
        SDL_RenderDrawPoint(r, cx - ix - 1, cy + dy);
        SDL_RenderDrawPoint(r, cx + ix + 1, cy + dy);
        set_color(r, c);
    }
}

static void draw_panel(SDL_Renderer *ren, SDL_Rect r) {
    int rad = r.h >= 80 ? 12 : 7;
    rrect_bordered(ren, r, rad, (SDL_Color){COL_SURFACE}, (SDL_Color){COL_BORDER});
}

/* Secondary button: raised surface, accent outline on hover. */
static void draw_button(Gui *g, SDL_Rect r, const char *label, TTF_Font *font, int hovered, int enabled) {
    SDL_Color fill = hovered && enabled ? (SDL_Color){36, 42, 60, 255} : (SDL_Color){COL_SURFACE2};
    SDL_Color border = !enabled ? (SDL_Color){COL_BORDER} : (hovered ? (SDL_Color){COL_LOSS} : (SDL_Color){COL_BORDER_HI});
    rrect_bordered(g->ren, r, 8, fill, border);
    int tw = 0;
    TTF_SizeText(font, label, &tw, NULL);
    SDL_Color col = enabled ? (SDL_Color){COL_TEXT} : (SDL_Color){COL_TEXT_DIM};
    txt(g, font, label, r.x + (r.w - tw) / 2, r.y + (r.h - TTF_FontHeight(font)) / 2, col);
}

/* Primary action: filled accent, dark label. */
static void draw_button_primary(Gui *g, SDL_Rect r, const char *label, TTF_Font *font, int hovered, int enabled) {
    SDL_Color fill = enabled ? (hovered ? (SDL_Color){128, 166, 255, 255} : (SDL_Color){COL_LOSS})
                             : (SDL_Color){COL_SURFACE2};
    if (enabled) draw_shadow(g->ren, r, 10);
    fill_rrect(g->ren, r, 10, fill);
    int tw = 0;
    TTF_SizeText(font, label, &tw, NULL);
    SDL_Color col = enabled ? (SDL_Color){10, 14, 28, 255} : (SDL_Color){COL_TEXT_DIM};
    txt(g, font, label, r.x + (r.w - tw) / 2, r.y + (r.h - TTF_FontHeight(font)) / 2, col);
}

/* Greedy word wrap into at most `max_lines` lines of `max_chars` characters. */
static void draw_wrapped(Gui *g, TTF_Font *font, const char *s, int x, int y, int max_chars, int line_h,
                          int max_lines, SDL_Color col) {
    char line[128];
    const char *p = s;
    for (int ln = 0; ln < max_lines && *p; ++ln) {
        int n = (int)strlen(p);
        if (n > max_chars) {
            n = max_chars;
            int cut = n;
            while (cut > 0 && p[cut] != ' ') cut--;
            if (cut > 0) n = cut;
        }
        if (n > (int)sizeof(line) - 1) n = (int)sizeof(line) - 1;
        memcpy(line, p, (size_t)n);
        line[n] = '\0';
        txt(g, font, line, x, y + ln * line_h, col);
        p += n;
        while (*p == ' ') p++;
    }
}

/* -- setup form + node editor -------------------------------------------- */

static uint32_t be_u32(const unsigned char *b) { return ((uint32_t)b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]; }

static void parse_defaults(const char *path, DatasetInfo *d) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[96], key[32];
    double v;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%31s %lf", key, &v) != 2) continue;
        if (!strcmp(key, "epochs")) d->def_epochs = (int)v;
        else if (!strcmp(key, "batch_size")) d->def_batch = (int)v;
        else if (!strcmp(key, "lr")) d->def_lr = (float)v;
        else if (!strcmp(key, "momentum")) d->def_momentum = (float)v;
        else if (!strcmp(key, "normalize")) d->def_normalize = (int)v;
        else if (!strcmp(key, "augment")) d->def_augment = (int)v;
    }
    fclose(f);
}

/* Fills *d for data/<name>/ or returns 0 if it isn't a usable dataset: the name
 * has to be one train_cfg_derive() accepts, the IDX files must exist with a
 * supported image size, and there must be at least two classes. */
static int dataset_probe(const char *name, DatasetInfo *d) {
    size_t n = strlen(name);
    if (n == 0 || n >= sizeof(d->name)) return 0;
    for (size_t i = 0; i < n; ++i) {
        char c = name[i];
        if (!(isalnum((unsigned char)c) || c == '-' || c == '_')) return 0;
    }
    memset(d, 0, sizeof(*d));
    snprintf(d->name, sizeof(d->name), "%s", name);
    d->def_epochs = d->def_batch = d->def_normalize = d->def_augment = -1;
    d->def_lr = d->def_momentum = -1.0f;

    char path[128];
    unsigned char hdr[20];
    snprintf(path, sizeof(path), "data/%s/train-images-idx3-ubyte", name);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t got = fread(hdr, 1, sizeof(hdr), f);
    fclose(f);
    /* magic 0x803: (count, rows, cols) grey images; 0x804: (count, channels, rows, cols) colour */
    unsigned magic = got >= 4 ? be_u32(hdr) : 0;
    if (got < 16 || (magic != 0x00000803u && magic != 0x00000804u)) return 0;
    if (magic == 0x00000804u && got < 20) return 0;
    d->count = (int)be_u32(hdr + 4);
    d->channels = magic == 0x00000804u ? (int)be_u32(hdr + 8) : 1;
    d->rows = (int)be_u32(hdr + (magic == 0x00000804u ? 12 : 8));
    d->cols = (int)be_u32(hdr + (magic == 0x00000804u ? 16 : 12));
    if (d->channels < 1 || d->channels > MAX_IMG_CH) return 0;
    if (d->count < 1 || d->rows < 1 || d->rows > MAX_IMG_SIDE || d->cols < 1 || d->cols > MAX_IMG_SIDE) return 0;

    snprintf(path, sizeof(path), "data/%s/train-labels-idx1-ubyte", name);
    f = fopen(path, "rb");
    if (!f) return 0;
    unsigned char lh[8];
    if (fread(lh, 1, sizeof(lh), f) != sizeof(lh) || be_u32(lh) != 0x00000801u) { fclose(f); return 0; }
    int max_label = -1;
    unsigned char buf[4096];
    size_t r;
    while ((r = fread(buf, 1, sizeof(buf), f)) > 0)
        for (size_t i = 0; i < r; ++i)
            if (buf[i] > max_label) max_label = buf[i];
    fclose(f);
    d->n_classes = max_label + 1;
    snprintf(path, sizeof(path), "data/%s/train-masks-idx3-ubyte", name);
    d->task = access(path, R_OK) == 0 ? TASK_SEGMENT : TASK_CLASSIFY;

    snprintf(path, sizeof(path), "data/%s/classes.txt", name);
    f = fopen(path, "r");
    if (f) {
        char line[64];
        int k = 0;
        while (k < MAX_CLASSES && fgets(line, sizeof(line), f)) {
            line[strcspn(line, "\r\n")] = '\0';
            if (!line[0]) continue;
            snprintf(d->class_names[k++], CLASS_NAME_LEN, "%.23s", line);
        }
        fclose(f);
        if (k > 0) {
            d->has_names = 1;
            if (k > d->n_classes) d->n_classes = k;
        }
    } else if (!strcmp(name, "fashion-mnist") && d->n_classes <= 10) {
        for (int k = 0; k < 10; ++k) snprintf(d->class_names[k], CLASS_NAME_LEN, "%s", FASHION_NAMES[k]);
        d->has_names = 1;
        d->n_classes = 10;
    }
    if (d->n_classes < 2 || d->n_classes > MAX_CLASSES) return 0;

    snprintf(path, sizeof(path), "data/%s/defaults.txt", name);
    parse_defaults(path, d);
    return 1;
}

static int cmp_names(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

static void scan_datasets(Gui *g) {
    g->n_datasets = 0;
    DIR *dir = opendir("data");
    if (!dir) return;
    struct dirent *ent;
    while ((ent = readdir(dir)) && g->n_datasets < MAX_DATASETS) {
        if (ent->d_name[0] == '.') continue;
        if (dataset_probe(ent->d_name, &g->ds[g->n_datasets])) g->n_datasets++;
    }
    closedir(dir);
    qsort(g->ds, (size_t)g->n_datasets, sizeof(g->ds[0]), cmp_names);
}

static const DatasetInfo *ds_find(const Gui *g, const char *name) {
    for (int i = 0; i < g->n_datasets; ++i)
        if (strcmp(g->ds[i].name, name) == 0) return &g->ds[i];
    return NULL;
}

/* The dataset currently picked in the form, or NULL. */
static const DatasetInfo *form_dataset(const Gui *g) { return ds_find(g, g->fields[F_DATASET].text); }

static const char *class_name(const DatasetInfo *d, int label, char *buf, size_t bufsz) {
    if (d->has_names && label >= 0 && label < MAX_CLASSES && d->class_names[label][0]) return d->class_names[label];
    snprintf(buf, bufsz, "%d", label);
    return buf;
}

static void field_set(Gui *g, int idx, const char *text, int kind) {
    snprintf(g->fields[idx].text, sizeof(g->fields[idx].text), "%s", text);
    g->fields[idx].kind = kind;
}

/* -- the node graph ------------------------------------------------------- */

static int node_height(int type) {
    switch (type) {
    case BLK_CONV: return 100;
    case BLK_LINEAR:
    case BLK_UPSAMPLE:
    case BLK_MAXPOOL: return 72;
    case BLK_ADD:
    case BLK_CONCAT: return 62;
    case BLK_RESHAPE: return 84;
    case BLK_REPARAM: return 72;
    default: return 46;
    }
}

static int node_nparams(int type) {
    switch (type) {
    case BLK_CONV: return 4;
    case BLK_LINEAR: return 1;
    case BLK_MAXPOOL: return 2;
    case BLK_UPSAMPLE: return 1;
    case BLK_RESHAPE: return 3;
    case BLK_REPARAM: return 1;
    default: return 0;
    }
}
static int node_nin(int type) { return type == NODE_INPUT ? 0 : (type == BLK_ADD || type == BLK_CONCAT ? 2 : 1); }
static int node_deletable(int type) { return type != NODE_INPUT && type != NODE_OUTPUT; }

static const char *node_type_title(int type) {
    switch (type) {
    case NODE_INPUT: return "input";
    case NODE_OUTPUT: return "output";
    case BLK_CONV: return "conv";
    case BLK_RELU: return "relu";
    case BLK_FLATTEN: return "flatten";
    case BLK_ADD: return "add";
    case BLK_BATCHNORM: return "bn";
    case BLK_MAXPOOL: return "pool";
    case BLK_GAP: return "gap";
    case BLK_UPSAMPLE: return "up";
    case BLK_CONCAT: return "concat";
    case BLK_RESHAPE: return "reshape";
    case BLK_REPARAM: return "reparam";
    default: return "fc";
    }
}

static SDL_Color node_accent(int type) {
    switch (type) {
    case BLK_CONV: return (SDL_Color){57, 135, 229, 255};
    case BLK_RELU: return (SDL_Color){96, 181, 120, 255};
    case BLK_FLATTEN: return (SDL_Color){176, 138, 224, 255};
    case BLK_LINEAR: return (SDL_Color){232, 160, 76, 255};
    case BLK_ADD: return (SDL_Color){232, 108, 160, 255};
    case BLK_BATCHNORM: return (SDL_Color){64, 196, 200, 255};
    case BLK_MAXPOOL: return (SDL_Color){214, 200, 90, 255};
    case BLK_GAP: return (SDL_Color){150, 206, 84, 255};
    case BLK_UPSAMPLE: return (SDL_Color){255, 150, 110, 255};
    case BLK_CONCAT: return (SDL_Color){190, 120, 240, 255};
    case BLK_RESHAPE: return (SDL_Color){140, 170, 255, 255};
    case BLK_REPARAM: return (SDL_Color){240, 236, 160, 255};
    default: return (SDL_Color){150, 156, 168, 255};
    }
}

static SDL_Rect node_rect(const Gui *g, int i) {
    const Node *n = &g->nodes[i];
    SDL_Rect r = {(int)floorf(g->pan_x) + n->x, (int)floorf(g->pan_y) + n->y, NODE_W, node_height(n->type)};
    return r;
}

static SDL_Point node_out_port(const Gui *g, int i) {
    SDL_Rect r = node_rect(g, i);
    SDL_Point p = {r.x + r.w, r.y + 11};
    return p;
}

/* Input port k of node i: an add node has two, stacked. */
static SDL_Point node_in_port(const Gui *g, int i, int k) {
    SDL_Rect r = node_rect(g, i);
    SDL_Point p = {r.x, r.y + 11 + k * 22};
    return p;
}

static SDL_Rect node_del_rect(SDL_Rect r) {
    SDL_Rect d = {r.x + r.w - 20, r.y + 4, 16, 16};
    return d;
}

static int near_port(SDL_Point port, SDL_Point p, int rad) {
    int dx = port.x - p.x, dy = port.y - p.y;
    return dx * dx + dy * dy <= rad * rad;
}

static int graph_find(const Gui *g, int type) {
    for (int i = 0; i < MAX_NODES; ++i)
        if (g->nodes[i].used && g->nodes[i].type == type) return i;
    return -1;
}

/* How many input ports across the graph read node i's output. */
static int graph_consumers(const Gui *g, int i) {
    int n = 0;
    for (int j = 0; j < MAX_NODES; ++j) {
        if (!g->nodes[j].used) continue;
        for (int k = 0; k < node_nin(g->nodes[j].type); ++k)
            if (g->nodes[j].in[k] == i) n++;
    }
    return n;
}

/* Does `node` read `on`, directly or through other nodes? */
static int graph_depends(const Gui *g, int node, int on, int *seen) {
    if (node == on) return 1;
    if (seen[node]) return 0;
    seen[node] = 1;
    const Node *n = &g->nodes[node];
    for (int k = 0; k < node_nin(n->type); ++k)
        if (n->in[k] >= 0 && graph_depends(g, n->in[k], on, seen)) return 1;
    return 0;
}

/* Wires from's output into input port `port` of `to`, replacing whatever was there. A link that
 * would close a loop is ignored. */
static void graph_link(Gui *g, int from, int to, int port) {
    if (from == to || from < 0 || to < 0 || !g->nodes[from].used || !g->nodes[to].used) return;
    if (g->nodes[from].type == NODE_OUTPUT || g->nodes[to].type == NODE_INPUT) return;
    if (port < 0 || port >= node_nin(g->nodes[to].type)) return;
    int seen[MAX_NODES] = {0};
    if (graph_depends(g, from, to, seen)) return;
    g->nodes[to].in[port] = from;
}

static void node_set_defaults(Gui *g, int i) {
    static const char *conv_def[4] = {"16", "3", "1", "1"};
    int t = g->nodes[i].type;
    if (t == BLK_CONV)
        for (int j = 0; j < 4; ++j) field_set(g, F_NODE(i, j), conv_def[j], FK_INT);
    else if (t == BLK_LINEAR) field_set(g, F_NODE(i, 0), "64", FK_INT);
    else if (t == BLK_MAXPOOL) {
        field_set(g, F_NODE(i, 0), "2", FK_INT);
        field_set(g, F_NODE(i, 1), "2", FK_INT);
    } else if (t == BLK_UPSAMPLE) field_set(g, F_NODE(i, 0), "2", FK_INT);
    else if (t == BLK_RESHAPE) {
        field_set(g, F_NODE(i, 0), "32", FK_INT);
        field_set(g, F_NODE(i, 1), "7", FK_INT);
        field_set(g, F_NODE(i, 2), "7", FK_INT);
    } else if (t == BLK_REPARAM) field_set(g, F_NODE(i, 0), "1", FK_FLOAT);
}

static int graph_add(Gui *g, int type, int x, int y) {
    for (int i = 0; i < MAX_NODES; ++i) {
        if (g->nodes[i].used) continue;
        g->nodes[i].used = 1;
        g->nodes[i].type = type;
        g->nodes[i].x = x;
        g->nodes[i].y = y;
        g->nodes[i].in[0] = g->nodes[i].in[1] = -1;
        node_set_defaults(g, i);
        return i;
    }
    return -1;
}

/* Deleting a block re-joins the graph: whatever read it now reads what it read (its first input). */
static void graph_remove(Gui *g, int i) {
    if (!g->nodes[i].used || !node_deletable(g->nodes[i].type)) return;
    int repl = g->nodes[i].in[0];
    for (int j = 0; j < MAX_NODES; ++j) {
        if (!g->nodes[j].used || j == i) continue;
        for (int k = 0; k < 2; ++k)
            if (g->nodes[j].in[k] == i) g->nodes[j].in[k] = repl;
    }
    g->nodes[i].used = 0;
    g->nodes[i].in[0] = g->nodes[i].in[1] = -1;
    if (g->focus >= F_NODE(i, 0) && g->focus < F_NODE(i, 4)) g->focus = -1;
}

static int rects_overlap(int ax, int ay, int aw, int ah, int bx, int by, int bw, int bh) {
    return ax < bx + bw && bx < ax + aw && ay < by + bh && by < ay + ah;
}

/* First position at or after (x,y) - scanning right, wrapping down - where a
 * node of `type` wouldn't sit on top of another one. */
static void graph_free_spot(const Gui *g, int ignore, int type, int *x, int *y) {
    int h = node_height(type);
    for (int tries = 0; tries < 400; ++tries) {
        if (*x + NODE_W > GRAPH_VIEW.w - 8) {
            *x = 14;
            *y += 110;
        }
        int hit = 0;
        for (int i = 0; i < MAX_NODES && !hit; ++i) {
            if (!g->nodes[i].used || i == ignore) continue;
            if (rects_overlap(*x - 6, *y - 6, NODE_W + 12, h + 12, g->nodes[i].x, g->nodes[i].y, NODE_W,
                              node_height(g->nodes[i].type)))
                hit = 1;
        }
        if (!hit) return;
        *x += NODE_W + NODE_GAP;
        if (*x + NODE_W > GRAPH_VIEW.w - 8) {
            *x = 14;
            *y += 110;
        }
    }
}

/* Pans the canvas just far enough that node i is on screen, clear of the hint text at the bottom. */
static void graph_reveal(Gui *g, int i) {
    SDL_Rect r = node_rect(g, i), v = view_rect(g);
    int bottom = v.h - (int)(60.0f / g->zoom), right = v.w - 8;
    if (r.y + r.h > bottom) g->pan_y -= (float)(r.y + r.h - bottom);
    if (r.y < 8) g->pan_y += (float)(8 - r.y);
    if (r.x + r.w > right) g->pan_x -= (float)(r.x + r.w - right);
    if (r.x < 8) g->pan_x += (float)(8 - r.x);
}

/* Clicking a palette block appends it after the block that feeds the output node (an add block gets
 * that on its first port; its second port is for you to wire), and the output node now reads it. */
static void graph_append(Gui *g, int type) {
    int in = graph_find(g, NODE_INPUT), out = graph_find(g, NODE_OUTPUT);
    if (in < 0 || out < 0) return;
    int pred = g->nodes[out].in[0];
    if (pred < 0) pred = in;
    int x = g->nodes[pred].x + NODE_W + NODE_GAP, y = g->nodes[pred].y;
    graph_free_spot(g, -1, type, &x, &y);
    int n = graph_add(g, type, x, y);
    if (n < 0) return;
    g->nodes[n].in[0] = pred;
    g->nodes[out].in[0] = n;
    int ox = x + NODE_W + NODE_GAP, oy = y;
    graph_free_spot(g, out, NODE_OUTPUT, &ox, &oy);
    g->nodes[out].x = ox;
    g->nodes[out].y = oy;
    graph_reveal(g, n);
}

/* Depth-first walk backwards from `node` along the input wires; every node is appended to order[]
 * after everything it reads, which is exactly an evaluation order. Returns 0, or 1 if an input port
 * is unwired (err_node and err_port say which), or 2 if the wires loop. */
static int graph_visit(const Gui *g, int node, int *state, int *order, int *n, int *err_node, int *err_port) {
    state[node] = 1;
    const Node *nd = &g->nodes[node];
    for (int k = 0; k < node_nin(nd->type); ++k) {
        int src = nd->in[k];
        if (src < 0 || !g->nodes[src].used) {
            *err_node = node;
            *err_port = k;
            return 1;
        }
        if (state[src] == 1) {
            *err_node = node;
            return 2;
        }
        if (state[src] == 0) {
            int r = graph_visit(g, src, state, order, n, err_node, err_port);
            if (r) return r;
        }
    }
    state[node] = 2;
    order[(*n)++] = node;
    return 0;
}

/* The zoom at which every node fits the canvas (never past 100%), and the bounding box that was measured. */
static float graph_fit_zoom(const Gui *g, int *minx_o, int *miny_o) {
    int minx = 1 << 30, miny = 1 << 30, maxx = -(1 << 30), maxy = -(1 << 30), any = 0;
    for (int i = 0; i < MAX_NODES; ++i) {
        if (!g->nodes[i].used) continue;
        any = 1;
        if (g->nodes[i].x < minx) minx = g->nodes[i].x;
        if (g->nodes[i].y < miny) miny = g->nodes[i].y;
        if (g->nodes[i].x + NODE_W > maxx) maxx = g->nodes[i].x + NODE_W;
        if (g->nodes[i].y + node_height(g->nodes[i].type) > maxy) maxy = g->nodes[i].y + node_height(g->nodes[i].type);
    }
    if (minx_o) *minx_o = any ? minx : 0;
    if (miny_o) *miny_o = any ? miny : 0;
    if (!any) return 1.0f;
    float zx = (float)GRAPH_VIEW.w / (float)(maxx - minx + 48), zy = (float)(GRAPH_VIEW.h - 70) / (float)(maxy - miny + 48);
    float z = fminf(1.0f, fminf(zx, zy));
    return z < ZOOM_MIN ? ZOOM_MIN : z;
}

/* Zooms out (never in past 100%) and pans so that every node is on screen, top-left aligned. */
static void graph_fit(Gui *g) {
    int minx, miny;
    float z = graph_fit_zoom(g, &minx, &miny);
    g->zoom = z;
    g->pan_x = 14.0f / z - (float)minx + 10.0f;
    g->pan_y = 14.0f / z - (float)miny + 10.0f;
}

/* Places the nodes in order left to right, wrapping into rows `wrap` pixels wide. */
static void graph_layout_rows(Gui *g, const int *order, int n, int wrap) {
    int x = 14, y = 14, row_h = 0;
    for (int k = 0; k < n; ++k) {
        Node *nd = &g->nodes[order[k]];
        int h = node_height(nd->type);
        if (x + NODE_W > wrap - 8) {
            x = 14;
            y += row_h + 34;
            row_h = 0;
        }
        nd->x = x;
        nd->y = y;
        x += NODE_W + NODE_GAP;
        if (h > row_h) row_h = h;
    }
}

/* Lays the nodes out left to right in evaluation order, wrapping into rows; anything not connected
 * to the output goes last. */
static void graph_arrange(Gui *g) {
    int order[MAX_NODES], n = 0, state[MAX_NODES] = {0}, en = -1, ep = 0;
    int out = graph_find(g, NODE_OUTPUT);
    if (out >= 0) {
        /* a partial walk still orders everything it reached, so keep what it produced */
        graph_visit(g, out, state, order, &n, &en, &ep);
    }
    int placed[MAX_NODES] = {0};
    for (int k = 0; k < n; ++k) placed[order[k]] = 1;
    for (int i = 0; i < MAX_NODES; ++i)
        if (g->nodes[i].used && !placed[i]) order[n++] = i;

    /* A tall, narrow layout needs a small zoom to fit; a wider one may fit at a larger zoom. Try a few row widths
     * (1x to 2.5x the canvas width) and keep the one that can be shown biggest; a graph that already fits at
     * 100% keeps the plain canvas width. */
    int best_wrap = GRAPH_VIEW.w;
    float best_z = -1.0f;
    for (int step = 0; step < 6; ++step) {
        int wrap = GRAPH_VIEW.w + step * GRAPH_VIEW.w / 4;
        graph_layout_rows(g, order, n, wrap);
        float z = graph_fit_zoom(g, NULL, NULL);
        if (z > best_z + 0.01f) {
            best_z = z;
            best_wrap = wrap;
        }
        if (z >= 1.0f) break;
    }
    graph_layout_rows(g, order, n, best_wrap);
    graph_fit(g); /* the view shows the whole graph, zoomed out if it is big */
}

static int graph_has_type(const Gui *g, int type) {
    for (int i = 0; i < MAX_NODES; ++i)
        if (g->nodes[i].used && g->nodes[i].type == type) return 1;
    return 0;
}

/* Sets the last fc (the one feeding the output node) to the dataset's class count. A VAE's last layer is
 * the reconstructed image, whose channels come from the dataset, so it is left alone. */
static void graph_fit_classes(Gui *g, int n_classes) {
    if (graph_has_type(g, BLK_REPARAM)) return;
    int out = graph_find(g, NODE_OUTPUT);
    int last = out >= 0 ? g->nodes[out].in[0] : -1;
    if (last >= 0 && (g->nodes[last].type == BLK_LINEAR || g->nodes[last].type == BLK_CONV)) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", n_classes);
        field_set(g, F_NODE(last, 0), buf, FK_INT);
    }
}

static void graph_clear(Gui *g) {
    memset(g->nodes, 0, sizeof(g->nodes));
    g->focus = -1;
    g->drag = DRAG_NONE;
}

/* Adds a node with its parameter values (a..d, in field order) and wires its inputs. */
static int gadd(Gui *g, int type, int a, int b, int c, int d, int in0, int in1) {
    int n = graph_add(g, type, 0, 0);
    if (n < 0) return -1;
    int v[4] = {a, b, c, d};
    char buf[16];
    for (int j = 0; j < node_nparams(type); ++j) {
        snprintf(buf, sizeof(buf), "%d", v[j]);
        field_set(g, F_NODE(n, j), buf, FK_INT);
    }
    g->nodes[n].in[0] = in0;
    g->nodes[n].in[1] = in1;
    return n;
}

static void graph_finish(Gui *g, int last) {
    int out = graph_add(g, NODE_OUTPUT, 0, 0);
    g->nodes[out].in[0] = last;
    const DatasetInfo *d = form_dataset(g);
    graph_fit_classes(g, d ? d->n_classes : 10);
    graph_arrange(g);
}

/* The model the trainer used before the editor existed: a plain conv chain. */
static void graph_reset(Gui *g) {
    graph_clear(g);
    static const int def[][5] = {{BLK_CONV, 8, 3, 1, 1},  {BLK_RELU, 0, 0, 0, 0},   {BLK_CONV, 16, 3, 2, 1},
                                  {BLK_RELU, 0, 0, 0, 0},  {BLK_CONV, 32, 3, 2, 1},   {BLK_RELU, 0, 0, 0, 0},
                                  {BLK_CONV, 32, 3, 1, 1}, {BLK_RELU, 0, 0, 0, 0},    {BLK_FLATTEN, 0, 0, 0, 0},
                                  {BLK_LINEAR, 128, 0, 0, 0}, {BLK_RELU, 0, 0, 0, 0}, {BLK_LINEAR, 10, 0, 0, 0}};
    int prev = graph_add(g, NODE_INPUT, 0, 0);
    for (unsigned k = 0; k < sizeof(def) / sizeof(def[0]); ++k)
        prev = gadd(g, def[k][0], def[k][1], def[k][2], def[k][3], def[k][4], prev, -1);
    graph_finish(g, prev);
}

/* conv-bn-relu-conv-bn, plus the block's input added back on (through a 1x1 conv + bn when the
 * block changes the shape), then relu. Returns the node holding the block's output. */
static int resnet_block(Gui *g, int x, int ch, int stride, int project) {
    int c1 = gadd(g, BLK_CONV, ch, 3, stride, 1, x, -1);
    int b1 = gadd(g, BLK_BATCHNORM, 0, 0, 0, 0, c1, -1);
    int r1 = gadd(g, BLK_RELU, 0, 0, 0, 0, b1, -1);
    int c2 = gadd(g, BLK_CONV, ch, 3, 1, 1, r1, -1);
    int b2 = gadd(g, BLK_BATCHNORM, 0, 0, 0, 0, c2, -1);
    int skip = x;
    if (project) {
        int pc = gadd(g, BLK_CONV, ch, 1, stride, 0, x, -1);
        skip = gadd(g, BLK_BATCHNORM, 0, 0, 0, 0, pc, -1);
    }
    int sum = gadd(g, BLK_ADD, 0, 0, 0, 0, b2, skip);
    return gadd(g, BLK_RELU, 0, 0, 0, 0, sum, -1);
}

/* A small ResNet: stem, one identity block, one down-sampling block with a 1x1 shortcut,
 * global average pool, fc. Six convs plus the shortcut. */
static void graph_preset_resnet(Gui *g) {
    graph_clear(g);
    int x = graph_add(g, NODE_INPUT, 0, 0);
    x = gadd(g, BLK_CONV, 16, 3, 1, 1, x, -1);
    x = gadd(g, BLK_BATCHNORM, 0, 0, 0, 0, x, -1);
    x = gadd(g, BLK_RELU, 0, 0, 0, 0, x, -1);
    x = resnet_block(g, x, 16, 1, 0);
    x = resnet_block(g, x, 32, 2, 1);
    x = gadd(g, BLK_GAP, 0, 0, 0, 0, x, -1);
    x = gadd(g, BLK_LINEAR, 10, 0, 0, 0, x, -1);
    graph_finish(g, x);
}

/* conv-bn-relu */
static int cbr(Gui *g, int x, int ch) {
    x = gadd(g, BLK_CONV, ch, 3, 1, 1, x, -1);
    x = gadd(g, BLK_BATCHNORM, 0, 0, 0, 0, x, -1);
    return gadd(g, BLK_RELU, 0, 0, 0, 0, x, -1);
}

/* A two-level U-Net for segmentation: two conv-bn-relu pairs per level, max pool down, nearest
 * upsample + concat with the matching encoder output on the way up, and a 1x1 conv to one score per
 * class per pixel. */
static void graph_preset_unet(Gui *g) {
    graph_clear(g);
    int x = graph_add(g, NODE_INPUT, 0, 0);
    x = cbr(g, x, 16);
    int e1 = cbr(g, x, 16);
    x = gadd(g, BLK_MAXPOOL, 2, 2, 0, 0, e1, -1);
    x = cbr(g, x, 32);
    int e2 = cbr(g, x, 32);
    x = gadd(g, BLK_MAXPOOL, 2, 2, 0, 0, e2, -1);
    x = cbr(g, x, 64);
    x = cbr(g, x, 64);
    x = gadd(g, BLK_UPSAMPLE, 2, 0, 0, 0, x, -1);
    x = gadd(g, BLK_CONCAT, 0, 0, 0, 0, x, e2);
    x = cbr(g, x, 32);
    x = cbr(g, x, 32);
    x = gadd(g, BLK_UPSAMPLE, 2, 0, 0, 0, x, -1);
    x = gadd(g, BLK_CONCAT, 0, 0, 0, 0, x, e1);
    x = cbr(g, x, 16);
    x = cbr(g, x, 16);
    x = gadd(g, BLK_CONV, 2, 1, 1, 0, x, -1);
    graph_finish(g, x);
}

/* What a VAE was tuned with here: Adam with a small learning rate (the loss sums over pixels), raw pixels. */
static void vae_training_defaults(Gui *g) {
    field_set(g, F_EPOCHS, "15", FK_INT);
    field_set(g, F_BATCH, "64", FK_INT);
    field_set(g, F_LR, "0.001", FK_FLOAT);
    field_set(g, F_OPT, "1", FK_INT);
    field_set(g, F_MOMENTUM, "0.9", FK_FLOAT);
    field_set(g, F_NORMALIZE, "0", FK_INT);
    field_set(g, F_AUGMENT, "0", FK_INT);
}

/* A convolutional VAE sized to the selected dataset: stride-2 convs halve the image until the feature map is at
 * most 11 pixels across (28 -> 7, 64 -> 8, 128 -> 8; h and w should be multiples of 2^levels), an fc to the
 * 2*16 means / log-variances, the sampling block, an fc back up to a 32-channel map, and the same number of
 * upsample + conv stages back to the image. The last conv has the image's channels. */
static void graph_preset_vae(Gui *g) {
    graph_clear(g);
    const DatasetInfo *d = form_dataset(g);
    int h = d ? d->rows : 28, c = d ? d->channels : 1, z = 16;
    int levels = 2;
    while ((h >> levels) > 11) levels++;
    int q = h >> levels > 0 ? h >> levels : 1;
    int x = graph_add(g, NODE_INPUT, 0, 0);
    for (int k = 0; k < levels; ++k) {
        x = gadd(g, BLK_CONV, k == 0 ? 16 : 32, 3, 2, 1, x, -1);
        x = gadd(g, BLK_RELU, 0, 0, 0, 0, x, -1);
    }
    x = gadd(g, BLK_FLATTEN, 0, 0, 0, 0, x, -1);
    x = gadd(g, BLK_LINEAR, 2 * z, 0, 0, 0, x, -1);
    x = gadd(g, BLK_REPARAM, 0, 0, 0, 0, x, -1);
    field_set(g, F_NODE(x, 0), "1", FK_FLOAT); /* KL weight */
    x = gadd(g, BLK_LINEAR, 32 * q * q, 0, 0, 0, x, -1);
    x = gadd(g, BLK_RELU, 0, 0, 0, 0, x, -1);
    x = gadd(g, BLK_RESHAPE, 32, q, q, 0, x, -1);
    for (int k = 0; k < levels; ++k) {
        x = gadd(g, BLK_UPSAMPLE, 2, 0, 0, 0, x, -1);
        x = gadd(g, BLK_CONV, k == levels - 1 ? c : 16, 3, 1, 1, x, -1);
        if (k != levels - 1) x = gadd(g, BLK_RELU, 0, 0, 0, 0, x, -1);
    }
    graph_finish(g, x);
    vae_training_defaults(g);
}

/* -- form state ----------------------------------------------------------- */

/* Picking a dataset loads the settings that suit it: its defaults.txt if it
 * ships one, otherwise batch 64 / lr 0.001 / plain SGD and enough epochs for
 * about 3000 steps (so 3 for the 60000-image MNIST sets, more for tiny ones).
 * The model's last fc is also resized to the dataset's class count. */
static void apply_dataset_defaults(Gui *g, const DatasetInfo *d) {
    char buf[32];
    int batch = d->def_batch > 0 ? d->def_batch : 64;
    int steps_per_epoch = (d->count + batch - 1) / batch;
    int epochs = d->def_epochs > 0 ? d->def_epochs : (int)(3000.0 / (steps_per_epoch > 0 ? steps_per_epoch : 1) + 0.5);
    if (epochs < 3) epochs = 3;
    if (epochs > 200) epochs = 200;
    snprintf(buf, sizeof(buf), "%d", batch);
    field_set(g, F_BATCH, buf, FK_INT);
    snprintf(buf, sizeof(buf), "%d", epochs);
    field_set(g, F_EPOCHS, buf, FK_INT);
    snprintf(buf, sizeof(buf), "%g", d->def_lr > 0 ? d->def_lr : 0.001f);
    field_set(g, F_LR, buf, FK_FLOAT);
    snprintf(buf, sizeof(buf), "%g", d->def_momentum >= 0 ? d->def_momentum : 0.0f);
    field_set(g, F_MOMENTUM, buf, FK_FLOAT);
    snprintf(buf, sizeof(buf), "%d", d->def_normalize >= 0 ? d->def_normalize : 0);
    field_set(g, F_NORMALIZE, buf, FK_INT);
    snprintf(buf, sizeof(buf), "%d", d->def_augment >= 0 ? d->def_augment : 0);
    field_set(g, F_AUGMENT, buf, FK_INT);
    field_set(g, F_OPT, "0", FK_INT);

    graph_fit_classes(g, d->n_classes);
}

/* The first dataset that has masks, or NULL. */
static const DatasetInfo *first_segmentation_dataset(const Gui *g) {
    for (int i = 0; i < g->n_datasets; ++i)
        if (g->ds[i].task == TASK_SEGMENT) return &g->ds[i];
    return NULL;
}

/* A model that suits the problem and dataset, plus the training settings that go with it. */
static void load_problem_defaults(Gui *g, const DatasetInfo *d) {
    if (g->problem == PROBLEM_GENERATE) {
        graph_preset_vae(g); /* sized to the dataset's images; sets the VAE training fields */
        if (d) {
            apply_dataset_defaults(g, d);
            vae_training_defaults(g);
        }
    } else {
        if (g->problem == PROBLEM_SEGMENT) graph_preset_unet(g);
        else graph_reset(g);
        if (d) apply_dataset_defaults(g, d);
    }
}

/* Switching the problem swaps in that problem's default model and settings. Segmentation needs masks: if the
 * selected dataset has none, the first dataset that does is selected instead. */
static void set_problem(Gui *g, int problem) {
    if (g->problem == problem) return;
    g->problem = problem;
    g->preset_open = 0;
    const DatasetInfo *d = form_dataset(g);
    if (problem == PROBLEM_SEGMENT && (!d || d->task != TASK_SEGMENT)) {
        const DatasetInfo *seg = first_segmentation_dataset(g);
        if (seg) {
            field_set(g, F_DATASET, seg->name, FK_CHOICE);
            d = seg;
        }
    }
    load_problem_defaults(g, d);
}

static void dataset_choose(Gui *g, const char *name) {
    int changed = strcmp(g->fields[F_DATASET].text, name) != 0;
    field_set(g, F_DATASET, name, FK_CHOICE);
    const DatasetInfo *d = ds_find(g, name);
    if (!d || !changed) return;
    if (g->problem == PROBLEM_CLASSIFY) {
        /* keep a classification model the user has edited; only replace one that cannot classify */
        int out = graph_find(g, NODE_OUTPUT);
        int last = out >= 0 ? g->nodes[out].in[0] : -1;
        int ends_in_fc = last >= 0 && g->nodes[last].type == BLK_LINEAR;
        if (!ends_in_fc) graph_reset(g);
        apply_dataset_defaults(g, d);
    } else {
        /* a U-Net / VAE is sized to the images it works on: rebuild it for the new dataset */
        load_problem_defaults(g, d);
    }
}

/* Starts out as the config the old hardcoded trainer used. */
static void form_defaults(Gui *g) {
    scan_datasets(g);
    const char *def = "";
    for (int i = 0; i < g->n_datasets; ++i)
        if (strcmp(g->ds[i].name, "fashion-mnist") == 0) def = g->ds[i].name;
    if (!def[0] && g->n_datasets > 0) def = g->ds[0].name;
    field_set(g, F_DATASET, "", FK_CHOICE);
    field_set(g, F_SEED, "42", FK_INT);
    field_set(g, F_OPT, "0", FK_INT);
    field_set(g, F_POOL, "8", FK_INT);
    field_set(g, F_VIZ, "1", FK_INT);
    g->focus = -1;
    graph_reset(g);
    if (def[0]) dataset_choose(g, def);
}

/* Screen rectangles of every field. Node fields exist only for live nodes and
 * follow their node as it is dragged or the canvas is panned. */
static void form_layout(Gui *g) {
    for (int i = 0; i < NUM_FIELDS; ++i) g->fields[i].visible = 0;
    for (int i = 0; i < NUM_GENERAL_FIELDS; ++i) {
        g->fields[i].rect = (SDL_Rect){PANEL_TRAIN.x + 196, 58 + i * FIELD_PITCH, 150, 28};
        g->fields[i].visible = 1;
    }
    for (int i = 0; i < MAX_NODES; ++i) {
        if (!g->nodes[i].used) continue;
        SDL_Rect r = node_rect(g, i);
        if (g->nodes[i].type == BLK_CONV) {
            g->fields[F_NODE(i, 0)].rect = (SDL_Rect){r.x + 30, r.y + 28, 36, 20};
            g->fields[F_NODE(i, 1)].rect = (SDL_Rect){r.x + 78, r.y + 28, 28, 20};
            g->fields[F_NODE(i, 2)].rect = (SDL_Rect){r.x + 18, r.y + 54, 28, 20};
            g->fields[F_NODE(i, 3)].rect = (SDL_Rect){r.x + 66, r.y + 54, 28, 20};
        } else if (g->nodes[i].type == BLK_LINEAR) {
            g->fields[F_NODE(i, 0)].rect = (SDL_Rect){r.x + 30, r.y + 28, 64, 20};
        } else if (g->nodes[i].type == BLK_RESHAPE) {
            g->fields[F_NODE(i, 0)].rect = (SDL_Rect){r.x + 8, r.y + 42, 30, 20};
            g->fields[F_NODE(i, 1)].rect = (SDL_Rect){r.x + 41, r.y + 42, 30, 20};
            g->fields[F_NODE(i, 2)].rect = (SDL_Rect){r.x + 74, r.y + 42, 30, 20};
        } else if (g->nodes[i].type == BLK_REPARAM) {
            g->fields[F_NODE(i, 0)].rect = (SDL_Rect){r.x + 56, r.y + 28, 48, 20};
        } else if (g->nodes[i].type == BLK_UPSAMPLE) {
            g->fields[F_NODE(i, 0)].rect = (SDL_Rect){r.x + 60, r.y + 28, 34, 20};
        } else if (g->nodes[i].type == BLK_MAXPOOL) {
            g->fields[F_NODE(i, 0)].rect = (SDL_Rect){r.x + 22, r.y + 28, 28, 20};
            g->fields[F_NODE(i, 1)].rect = (SDL_Rect){r.x + 72, r.y + 28, 28, 20};
        }
        for (int j = 0; j < node_nparams(g->nodes[i].type); ++j) g->fields[F_NODE(i, j)].visible = 1;
    }
}

static int parse_int(const Field *f, int *out) {
    char *end;
    errno = 0;
    long v = strtol(f->text, &end, 10);
    if (!f->text[0] || *end || errno || v < INT_MIN || v > INT_MAX) return -1;
    *out = (int)v;
    return 0;
}

static int form_fail(Gui *g, int idx, const char *what) {
    snprintf(g->form_err, sizeof(g->form_err), "%s: enter a whole number", what);
    g->form_err_field = idx;
    return -1;
}

/* Turns the node graph into form_cfg.blocks: walks backwards from the output node, which orders
 * every block after the blocks it reads, then turns each block's wires into tensor ids (0 = the
 * input, k+1 = block k) and reads its parameter fields. Fails (with form_err / form_err_node set)
 * if a port is unwired, the wires loop, or a block doesn't lead to the output. Shape and range
 * rules are then train_cfg_derive()'s job. */
static int graph_compile(Gui *g) {
    TrainConfig *c = &g->form_cfg;
    for (int i = 0; i < MAX_NODES; ++i) {
        g->node_block[i] = -1;
        snprintf(g->node_name[i], sizeof(g->node_name[i]), "%s", node_type_title(g->nodes[i].type));
    }
    g->chain_len = 0;
    int out = graph_find(g, NODE_OUTPUT);
    int state[MAX_NODES] = {0}, order[MAX_NODES], n_order = 0, err_node = -1, err_port = 0;
    int r = graph_visit(g, out, state, order, &n_order, &err_node, &err_port);
    if (r == 1) {
        int t = g->nodes[err_node].type;
        if (t == NODE_OUTPUT) snprintf(g->form_err, sizeof(g->form_err), "wire a block to the output node");
        else if (t == BLK_ADD || t == BLK_CONCAT) snprintf(g->form_err, sizeof(g->form_err), "%s: wire both of its inputs", node_type_title(t));
        else snprintf(g->form_err, sizeof(g->form_err), "wire the input of the %s block", node_type_title(t));
        g->form_err_node = err_node;
        return -1;
    }
    if (r == 2) {
        snprintf(g->form_err, sizeof(g->form_err), "the wires form a loop");
        g->form_err_node = err_node;
        return -1;
    }
    for (int i = 0; i < MAX_NODES; ++i) {
        if (g->nodes[i].used && node_deletable(g->nodes[i].type) && state[i] != 2) {
            snprintf(g->form_err, sizeof(g->form_err), "a %s block does not lead to the output - wire it in or delete it",
                     node_type_title(g->nodes[i].type));
            g->form_err_node = i;
            return -1;
        }
    }

    static const char *conv_cols[4] = {"out channels", "kernel", "stride", "padding"};
    static const char *pool_cols[2] = {"window", "stride"};
    int id_of_node[MAX_NODES];
    int n_conv = 0, n_lin = 0, n_bn = 0, n_add = 0, n_pool = 0, n_up = 0, n_cat = 0;
    for (int k = 0; k < n_order; ++k) {
        int node = order[k], type = g->nodes[node].type;
        if (type == NODE_INPUT) {
            id_of_node[node] = 0;
            continue;
        }
        if (type == NODE_OUTPUT) continue;
        if (g->chain_len >= MAX_BLOCKS) {
            snprintf(g->form_err, sizeof(g->form_err), "the model can have at most %d blocks", MAX_BLOCKS);
            g->form_err_node = node;
            return -1;
        }
        int bi = g->chain_len++;
        id_of_node[node] = bi + 1;
        Block b = {type, 0, 0, 0, 0, {id_of_node[g->nodes[node].in[0]], -1}, 0.0f};
        if (type == BLK_ADD || type == BLK_CONCAT) b.in[1] = id_of_node[g->nodes[node].in[1]];
        int *dst[4] = {&b.a, &b.b, &b.c, &b.d};
        char *nm = g->node_name[node];
        size_t nsz = sizeof(g->node_name[node]);
        if (type == BLK_CONV) snprintf(nm, nsz, "conv%d", ++n_conv);
        else if (type == BLK_LINEAR) snprintf(nm, nsz, "fc%d", ++n_lin);
        else if (type == BLK_BATCHNORM) snprintf(nm, nsz, "bn%d", ++n_bn);
        else if (type == BLK_ADD) snprintf(nm, nsz, "add%d", ++n_add);
        else if (type == BLK_MAXPOOL) snprintf(nm, nsz, "pool%d", ++n_pool);
        else if (type == BLK_UPSAMPLE) snprintf(nm, nsz, "up%d", ++n_up);
        else if (type == BLK_CONCAT) snprintf(nm, nsz, "cat%d", ++n_cat);
        if (type == BLK_REPARAM) {
            char *end;
            const char *t = g->fields[F_NODE(node, 0)].text;
            float v = strtof(t, &end);
            if (!t[0] || *end || !isfinite(v)) {
                snprintf(g->form_err, sizeof(g->form_err), "reparam kl weight: enter a number like 1 or 0.5");
                g->form_err_node = node;
                g->form_err_field = F_NODE(node, 0);
                return -1;
            }
            b.f = v;
        }
        for (int j = 0; j < node_nparams(type) && type != BLK_REPARAM; ++j) {
            if (parse_int(&g->fields[F_NODE(node, j)], dst[j])) {
                char what[48];
                const char *col = type == BLK_CONV ? conv_cols[j] : (type == BLK_MAXPOOL ? pool_cols[j] : (type == BLK_UPSAMPLE ? "factor" : (type == BLK_RESHAPE ? (j == 0 ? "channels" : (j == 1 ? "height" : "width")) : "out features")));
                snprintf(what, sizeof(what), "%s %s", nm, col);
                g->form_err_node = node;
                return form_fail(g, F_NODE(node, j), what);
            }
        }
        c->blocks[bi] = b;
        g->chain[bi] = node;
        g->node_block[node] = bi;
    }
    c->n_blocks = g->chain_len;
    return 0;
}

/* Reads every field and the graph into form_cfg, then asks
 * train_cfg_derive() (the same check the worker repeats) whether it can run.
 * Runs every frame - cheap, and it keeps the node shapes, the parameter count
 * and the Start button in sync with what has been edited so far. */
static void form_update(Gui *g) {
    TrainConfig *c = &g->form_cfg;
    memset(c, 0, sizeof(*c));
    memset(&g->form_shapes, 0, sizeof(g->form_shapes));
    g->form_shapes.err_block = -1;
    g->form_valid = 0;
    g->form_err[0] = '\0';
    g->form_err_field = -1;
    g->form_err_node = -1;
    g->chain_len = 0;
    for (int i = 0; i < MAX_NODES; ++i) g->node_block[i] = -1;

    if (!g->fields[F_DATASET].text[0]) {
        snprintf(g->form_err, sizeof(g->form_err), "no dataset found under data/ - see README for the download commands");
        g->form_err_field = F_DATASET;
        return;
    }
    const DatasetInfo *ds = form_dataset(g);
    if (!ds) {
        snprintf(g->form_err, sizeof(g->form_err), "dataset '%.20s' is not available any more - pick another", g->fields[F_DATASET].text);
        g->form_err_field = F_DATASET;
        return;
    }
    snprintf(c->dataset, sizeof(c->dataset), "%.31s", ds->name);
    c->img_h = ds->rows;
    c->img_w = ds->cols;
    c->img_c = ds->channels;
    c->task = g->problem == PROBLEM_GENERATE ? TASK_GENERATE : (g->problem == PROBLEM_SEGMENT ? TASK_SEGMENT : TASK_CLASSIFY);
    if (g->problem == PROBLEM_SEGMENT && ds->task != TASK_SEGMENT) {
        snprintf(g->form_err, sizeof(g->form_err),
                 "segmentation needs a dataset with masks (train-masks-idx3-ubyte) - '%.20s' has none", ds->name);
        g->form_err_field = F_DATASET;
        return;
    }
    c->num_classes = ds->n_classes;
    if (parse_int(&g->fields[F_EPOCHS], &c->epochs)) { form_fail(g, F_EPOCHS, "epochs"); return; }
    if (parse_int(&g->fields[F_BATCH], &c->batch_size)) { form_fail(g, F_BATCH, "batch size"); return; }
    {
        char *end;
        const char *t = g->fields[F_LR].text;
        float v = strtof(t, &end);
        if (!t[0] || *end || !isfinite(v)) {
            snprintf(g->form_err, sizeof(g->form_err), "learning rate: enter a number like 0.001");
            g->form_err_field = F_LR;
            return;
        }
        c->lr = v;
    }
    {
        char *end;
        const char *t = g->fields[F_MOMENTUM].text;
        float v = strtof(t, &end);
        if (!t[0] || *end || !isfinite(v)) {
            snprintf(g->form_err, sizeof(g->form_err), "momentum: enter a number like 0.9 (0 = plain SGD)");
            g->form_err_field = F_MOMENTUM;
            return;
        }
        c->momentum = v;
    }
    if (parse_int(&g->fields[F_OPT], &c->optimizer)) { form_fail(g, F_OPT, "optimizer"); return; }
    if (parse_int(&g->fields[F_NORMALIZE], &c->normalize)) { form_fail(g, F_NORMALIZE, "normalize"); return; }
    if (parse_int(&g->fields[F_AUGMENT], &c->augment)) { form_fail(g, F_AUGMENT, "augment"); return; }
    if (parse_int(&g->fields[F_SEED], &c->seed)) { form_fail(g, F_SEED, "seed"); return; }
    if (parse_int(&g->fields[F_POOL], &c->fc_pool)) { form_fail(g, F_POOL, "fc viz pool"); return; }
    if (parse_int(&g->fields[F_VIZ], &c->viz_every)) { form_fail(g, F_VIZ, "weights every N"); return; }

    if (graph_compile(g) != 0) return;
    if (train_cfg_derive(c, &g->form_shapes, g->form_err, sizeof(g->form_err)) == 0) {
        g->form_valid = 1;
    } else if (g->form_shapes.err_block >= 0 && g->form_shapes.err_block < g->chain_len) {
        g->form_err_node = g->chain[g->form_shapes.err_block];
    }
}

static long count_parameters(const TrainConfig *c, const NetShapes *s) {
    long total = 0;
    for (int i = 0; i < c->n_blocks && i < s->n_ok; ++i) {
        const Block *b = &c->blocks[i];
        LayerShape in = train_block_input(s, b);
        if (b->type == BLK_CONV) total += (long)b->a * in.c * b->b * b->b + b->a;
        else if (b->type == BLK_LINEAR) total += (long)in.c * b->a + b->a;
        else if (b->type == BLK_BATCHNORM) total += 2L * in.c; /* scale + shift */
    }
    return total;
}

/* -- drawing --------------------------------------------------------------- */

static void draw_field(Gui *g, int idx) {
    const Field *f = &g->fields[idx];
    int rad = f->rect.h < 24 ? 4 : 7;
    SDL_Color border = g->focus == idx ? (SDL_Color){COL_LOSS}
                       : g->form_err_field == idx ? (SDL_Color){COL_BAD}
                                                  : (SDL_Color){COL_BORDER_HI};
    if (g->focus == idx) { /* soft focus ring */
        SDL_Rect ring = {f->rect.x - 2, f->rect.y - 2, f->rect.w + 4, f->rect.h + 4};
        fill_rrect(g->ren, ring, rad + 2, (SDL_Color){96, 140, 255, 55});
    }
    rrect_bordered(g->ren, f->rect, rad, (SDL_Color){14, 16, 23, 255}, border);

    /* Text is clipped to the field, intersected with whatever clip the caller
     * (the node canvas) already set, and that clip is put back afterwards. */
    SDL_Rect saved = {0, 0, 0, 0}, clip = f->rect;
    int had_clip = SDL_RenderIsClipEnabled(g->ren);
    if (had_clip) {
        SDL_RenderGetClipRect(g->ren, &saved);
        SDL_IntersectRect(&clip, &saved, &clip);
    }
    int pad = f->rect.w < 50 ? 5 : 8;
    SDL_RenderSetClipRect(g->ren, &clip);
    txt(g, g->font, f->text, f->rect.x + pad, f->rect.y + (f->rect.h - TTF_FontHeight(g->font)) / 2,
        (SDL_Color){COL_TEXT});
    SDL_RenderSetClipRect(g->ren, had_clip ? &saved : NULL);

    if (f->kind == FK_CHOICE) {
        int ax = f->rect.x + f->rect.w - 20, ay = f->rect.y + 11;
        SDL_SetRenderDrawColor(g->ren, COL_TEXT_DIM);
        SDL_RenderDrawLine(g->ren, ax, ay, ax + 4, ay + 4);
        SDL_RenderDrawLine(g->ren, ax, ay + 1, ax + 4, ay + 5);
        SDL_RenderDrawLine(g->ren, ax + 4, ay + 4, ax + 8, ay);
        SDL_RenderDrawLine(g->ren, ax + 4, ay + 5, ax + 8, ay + 1);
    }

    if (g->focus == idx && f->kind != FK_CHOICE && (SDL_GetTicks() / 500) % 2 == 0) {
        int tw = 0;
        TTF_SizeText(g->font, f->text, &tw, NULL);
        int cx = f->rect.x + pad + tw + 1;
        if (cx > f->rect.x + f->rect.w - 4) cx = f->rect.x + f->rect.w - 4;
        SDL_SetRenderDrawColor(g->ren, COL_TEXT);
        SDL_RenderDrawLine(g->ren, cx, f->rect.y + 3, cx, f->rect.y + f->rect.h - 4);
    }
}

static SDL_Rect dropdown_row_rect(const Gui *g, int i) {
    const SDL_Rect *f = &g->fields[F_DATASET].rect;
    SDL_Rect r = {f->x, f->y + f->h + 2 + i * 28, f->w, 28};
    return r;
}

/* Drawn after everything else in the form so it floats over the fields below. */
static void draw_dropdown(Gui *g) {
    int vis[MAX_DATASETS], nv = ds_visible(g, vis);
    int rows = nv > 0 ? nv : 1;
    SDL_Rect first = dropdown_row_rect(g, 0);
    SDL_Rect bg = {first.x, first.y, first.w, rows * 28 + 8};
    bg.y -= 4;
    draw_shadow(g->ren, bg, 9);
    rrect_bordered(g->ren, bg, 9, (SDL_Color){COL_SURFACE2}, (SDL_Color){COL_LOSS});
    SDL_Point mouse = {g->mouse_x, g->mouse_y};
    if (nv == 0) {
        txt(g, g->font_small, g->problem == PROBLEM_SEGMENT ? "no dataset with masks in data/" : "no dataset in data/", first.x + 10,
            first.y + 8, (SDL_Color){COL_BAD});
    }
    for (int k = 0; k < nv; ++k) {
        int i = vis[k];
        SDL_Rect r = dropdown_row_rect(g, k);
        int current = strcmp(g->ds[i].name, g->fields[F_DATASET].text) == 0;
        if (SDL_PointInRect(&mouse, &r)) {
            SDL_Rect hl = {r.x + 4, r.y + 1, r.w - 8, r.h - 2};
            fill_rrect(g->ren, hl, 6, (SDL_Color){52, 62, 92, 255});
        }
        if (current) fill_circle_aa(g->ren, r.x + 12, r.y + r.h / 2, 3, (SDL_Color){COL_LOSS});
        txt(g, g->font, g->ds[i].name, r.x + 24, r.y + 4, current ? (SDL_Color){COL_LOSS} : (SDL_Color){COL_TEXT});
    }
}


/* Cubic bezier through four control points: a 2px core with a faint halo either side, which reads
 * as a smooth line on the software renderer. */
static void draw_curve(Gui *g, float p0x, float p0y, float p1x, float p1y, float p2x, float p2y, float p3x, float p3y,
                       SDL_Color col) {
    enum { SEG = 40 };
    int xs[SEG + 1], ys[SEG + 1];
    for (int k = 0; k <= SEG; ++k) {
        float t = (float)k / SEG, u = 1.0f - t;
        xs[k] = (int)(u * u * u * p0x + 3 * u * u * t * p1x + 3 * u * t * t * p2x + t * t * t * p3x);
        ys[k] = (int)(u * u * u * p0y + 3 * u * u * t * p1y + 3 * u * t * t * p2y + t * t * t * p3y);
    }
    for (int pass = 0; pass < 2; ++pass) {
        SDL_SetRenderDrawColor(g->ren, col.r, col.g, col.b, pass == 0 ? 55 : 255);
        for (int k = 1; k <= SEG; ++k) {
            if (pass == 0) {
                SDL_RenderDrawLine(g->ren, xs[k - 1], ys[k - 1] - 1, xs[k], ys[k] - 1);
                SDL_RenderDrawLine(g->ren, xs[k - 1], ys[k - 1] + 2, xs[k], ys[k] + 2);
            } else {
                SDL_RenderDrawLine(g->ren, xs[k - 1], ys[k - 1], xs[k], ys[k]);
                SDL_RenderDrawLine(g->ren, xs[k - 1], ys[k - 1] + 1, xs[k], ys[k] + 1);
            }
        }
    }
}

/* Cubic bezier leaving a rightwards and entering b from the left, ComfyUI style. */
static void draw_wire(Gui *g, SDL_Point a, SDL_Point b, SDL_Color col) {
    float dx = fmaxf(40.0f, fabsf((float)(b.x - a.x)) * 0.5f);
    draw_curve(g, (float)a.x, (float)a.y, (float)a.x + dx, (float)a.y, (float)b.x - dx, (float)b.y, (float)b.x,
               (float)b.y, col);
}

/* A socket: dark ring on the card's edge, filled with the wire colour once something is connected. */
static void draw_port(Gui *g, SDL_Point p, int linked, SDL_Color col) {
    fill_circle_aa(g->ren, p.x, p.y, 6, (SDL_Color){58, 66, 90, 255});
    fill_circle_aa(g->ren, p.x, p.y, 4, (SDL_Color){COL_BG});
    if (linked) fill_circle_aa(g->ren, p.x, p.y, 3, col);
}

static void draw_node(Gui *g, int i, SDL_Point mouse) {
    const Node *n = &g->nodes[i];
    SDL_Rect r = node_rect(g, i);
    SDL_Rect view = view_rect(g);
    if (!SDL_HasIntersection(&r, &view)) return;

    SDL_Color accent = node_accent(n->type);
    int err = g->form_err_node == i;
    int hover = SDL_PointInRect(&mouse, &r) && g->drag == DRAG_NONE;
    SDL_Color border = err ? (SDL_Color){COL_BAD} : (hover ? col_mix((SDL_Color){62, 70, 94, 255}, accent, 0.6f) : (SDL_Color){50, 57, 78, 255});
    draw_shadow(g->ren, r, 9);
    rrect_bordered(g->ren, r, 9, (SDL_Color){COL_SURFACE2}, border);

    /* header: the card's rounded top, tinted with the block's colour (drawn through a clip so only
     * the top 24px of the rounded shape shows) */
    SDL_Rect saved = {0, 0, 0, 0}, hdr = {r.x, r.y, r.w, 25};
    int had_clip = SDL_RenderIsClipEnabled(g->ren);
    if (had_clip) {
        SDL_RenderGetClipRect(g->ren, &saved);
        SDL_IntersectRect(&hdr, &saved, &hdr);
    }
    SDL_RenderSetClipRect(g->ren, &hdr);
    SDL_Rect inner = {r.x + 1, r.y + 1, r.w - 2, r.h - 2};
    fill_rrect(g->ren, inner, 8, col_mix((SDL_Color){COL_SURFACE2}, accent, 0.22f));
    SDL_RenderSetClipRect(g->ren, had_clip ? &saved : NULL);
    SDL_SetRenderDrawColor(g->ren, 20, 23, 32, 120);
    SDL_RenderDrawLine(g->ren, r.x + 1, r.y + 25, r.x + r.w - 2, r.y + 25);

    fill_circle_aa(g->ren, r.x + 14, r.y + 13, 3, accent);
    txt(g, g->font_med, g->node_name[i], r.x + 24, r.y + 4, (SDL_Color){COL_TEXT});
    if (node_deletable(n->type)) {
        SDL_Rect d = node_del_rect(r);
        int over = SDL_PointInRect(&mouse, &d);
        if (over) fill_circle_aa(g->ren, d.x + 8, d.y + 8, 8, (SDL_Color){245, 101, 112, 60});
        SDL_SetRenderDrawColor(g->ren, over ? 245 : 143, over ? 101 : 151, over ? 112 : 170, 255);
        SDL_RenderDrawLine(g->ren, d.x + 5, d.y + 5, d.x + 11, d.y + 11);
        SDL_RenderDrawLine(g->ren, d.x + 11, d.y + 5, d.x + 5, d.y + 11);
    }

    for (int k = 0; k < node_nin(n->type); ++k) draw_port(g, node_in_port(g, i, k), n->in[k] >= 0, (SDL_Color){COL_LOSS});
    if (n->type != NODE_OUTPUT) draw_port(g, node_out_port(g, i), graph_consumers(g, i) > 0, accent);

    SDL_Color dim = {COL_TEXT_DIM};
    if (n->type == BLK_CONV) {
        txt(g, g->font_small, "out", r.x + 6, r.y + 31, dim);
        txt(g, g->font_small, "k", r.x + 70, r.y + 31, dim);
        txt(g, g->font_small, "s", r.x + 6, r.y + 57, dim);
        txt(g, g->font_small, "p", r.x + 56, r.y + 57, dim);
    } else if (n->type == BLK_LINEAR) {
        txt(g, g->font_small, "out", r.x + 6, r.y + 31, dim);
    } else if (n->type == BLK_MAXPOOL) {
        txt(g, g->font_small, "k", r.x + 8, r.y + 31, dim);
        txt(g, g->font_small, "s", r.x + 58, r.y + 31, dim);
    } else if (n->type == BLK_UPSAMPLE) {
        txt(g, g->font_small, "factor", r.x + 8, r.y + 31, dim);
    } else if (n->type == BLK_RESHAPE) {
        txt(g, g->font_small, "channels  h   w", r.x + 8, r.y + 27, dim);
    } else if (n->type == BLK_REPARAM) {
        txt(g, g->font_small, "kl weight", r.x + 8, r.y + 31, dim);
    } else if (n->type == BLK_ADD) {
        txt(g, g->font_small, "top + bottom", r.x + 14, r.y + 24, dim);
    } else if (n->type == BLK_CONCAT) {
        txt(g, g->font_small, "top | bottom", r.x + 14, r.y + 24, dim);
    }
    for (int j = 0; j < node_nparams(n->type); ++j) draw_field(g, F_NODE(i, j));

    char shape[32] = "";
    const DatasetInfo *fds = form_dataset(g);
    if (n->type == NODE_INPUT) {
        if (fds) snprintf(shape, sizeof(shape), "%dx%dx%d", fds->rows, fds->cols, fds->channels);
    } else if (n->type == NODE_OUTPUT) {
        if (fds) snprintf(shape, sizeof(shape), "%d classes", fds->n_classes);
    } else if (g->node_block[i] >= 0 && g->node_block[i] < g->form_shapes.n_ok) {
        const LayerShape *o = &g->form_shapes.out[g->node_block[i]];
        if (o->flat) snprintf(shape, sizeof(shape), "-> %d", o->c);
        else snprintf(shape, sizeof(shape), "-> %dx%dx%d", o->h, o->w, o->c);
    }
    txt(g, g->font_small, shape, r.x + 8, r.y + r.h - 17, dim);
}

/* -- model dependency graph (training dashboard) ------------------------- */

#define ARCH_ROWS 3
#define ARCH_MAX_GROUPS (MAX_BLOCKS + 1)

/* One box of the graph: a block, with the relu / batch-norm blocks that directly follow it (and
 * nothing else reads its output) folded in so a deep net still fits the bar. */
typedef struct ArchGroup {
    char name[16];
    char l2[28];     /* output shape */
    char l2c[16];    /* the same, short ("16@64"), for boxes too narrow for l2 */
    char params[20]; /* kernel / stride, shown when nothing was folded in */
    char fold[20];   /* e.g. "+bn +relu" */
    int type;        /* BLK_* or NODE_INPUT: picks the accent colour */
    int level, row;  /* layer = longest path from the input; row separates parallel branches */
    int src[2], nsrc;
} ArchGroup;

static void shape_str_compact(LayerShape l, char *buf, size_t n) {
    if (l.flat) snprintf(buf, n, "%d", l.c);
    else snprintf(buf, n, "%d@%d", l.c, l.h);
}

static void shape_str(LayerShape l, char *buf, size_t n) {
    if (l.flat) snprintf(buf, n, "%d", l.c);
    else snprintf(buf, n, "%dx%dx%d", l.h, l.w, l.c);
}

static void arch_append(char *dst, size_t n, const char *add) {
    size_t len = strlen(dst);
    if (len + strlen(add) + 2 < n) snprintf(dst + len, n - len, "%s%s", len ? " " : "", add);
}

static int build_arch(const TrainConfig *c, const NetShapes *s, ArchGroup *gr) {
    int n = 0, uses[MAX_BLOCKS + 1] = {0}, gid_of_id[MAX_BLOCKS + 1] = {0}, out_id[ARCH_MAX_GROUPS] = {0};
    int n_conv = 0, n_lin = 0, n_bn = 0, n_add = 0, n_pool = 0, n_up = 0, n_cat = 0;
    int nb = c->n_blocks < s->n_ok ? c->n_blocks : s->n_ok;
    for (int i = 0; i < nb; ++i) {
        uses[c->blocks[i].in[0]]++;
        if (c->blocks[i].type == BLK_ADD || c->blocks[i].type == BLK_CONCAT) uses[c->blocks[i].in[1]]++;
    }

    memset(&gr[0], 0, sizeof(gr[0]));
    snprintf(gr[0].name, sizeof(gr[0].name), "input");
    shape_str(s->input, gr[0].l2, sizeof(gr[0].l2));
    shape_str_compact(s->input, gr[0].l2c, sizeof(gr[0].l2c));
    gr[0].type = NODE_INPUT;
    n = 1;

    for (int i = 0; i < nb; ++i) {
        const Block *k = &c->blocks[i];
        int in0 = k->in[0];
        if ((k->type == BLK_RELU || k->type == BLK_BATCHNORM) && in0 > 0 && uses[in0] == 1 &&
            out_id[gid_of_id[in0]] == in0) {
            int gi = gid_of_id[in0];
            arch_append(gr[gi].fold, sizeof(gr[gi].fold), k->type == BLK_RELU ? "+relu" : "+bn");
            out_id[gi] = i + 1;
            gid_of_id[i + 1] = gi;
            continue;
        }
        ArchGroup *q = &gr[n];
        memset(q, 0, sizeof(*q));
        q->type = k->type;
        q->src[0] = gid_of_id[in0];
        q->nsrc = 1;
        if (k->type == BLK_ADD || k->type == BLK_CONCAT) {
            q->src[1] = gid_of_id[k->in[1]];
            q->nsrc = 2;
        }
        shape_str(s->out[i], q->l2, sizeof(q->l2));
        shape_str_compact(s->out[i], q->l2c, sizeof(q->l2c));
        switch (k->type) {
        case BLK_CONV:
            snprintf(q->name, sizeof(q->name), "conv%d", ++n_conv);
            snprintf(q->params, sizeof(q->params), "%dx%d s%d", k->b, k->b, k->c);
            break;
        case BLK_LINEAR: snprintf(q->name, sizeof(q->name), "fc%d", ++n_lin); break;
        case BLK_ADD: snprintf(q->name, sizeof(q->name), "add%d", ++n_add); break;
        case BLK_UPSAMPLE:
            snprintf(q->name, sizeof(q->name), "up%d", ++n_up);
            snprintf(q->params, sizeof(q->params), "x%d", k->a);
            break;
        case BLK_CONCAT: snprintf(q->name, sizeof(q->name), "cat%d", ++n_cat); break;
        case BLK_BATCHNORM: snprintf(q->name, sizeof(q->name), "bn%d", ++n_bn); break;
        case BLK_MAXPOOL:
            snprintf(q->name, sizeof(q->name), "pool%d", ++n_pool);
            snprintf(q->params, sizeof(q->params), "%dx%d s%d", k->a, k->a, k->b);
            break;
        default: snprintf(q->name, sizeof(q->name), "%s", train_block_type_name(k->type)); break;
        }
        out_id[n] = i + 1;
        gid_of_id[i + 1] = n;
        n++;
    }

    /* layers: one past the deepest thing a box reads; rows: stay in the first input's row unless a
     * parallel branch already took that spot in the same layer. */
    int taken[ARCH_MAX_GROUPS][ARCH_ROWS];
    memset(taken, 0, sizeof(taken));
    taken[0][0] = 1;
    for (int gi = 1; gi < n; ++gi) {
        ArchGroup *q = &gr[gi];
        int lvl = 0;
        for (int k = 0; k < q->nsrc; ++k)
            if (gr[q->src[k]].level + 1 > lvl) lvl = gr[q->src[k]].level + 1;
        q->level = lvl;
        int row = gr[q->src[0]].row;
        while (row < ARCH_ROWS - 1 && taken[lvl][row]) row++;
        q->row = row;
        taken[lvl][row] = 1;
    }
    return n;
}

static void draw_arrow_head(Gui *g, float x, float y, float dx, float dy, SDL_Color col) {
    float px = -dy, py = dx; /* perpendicular */
    SDL_SetRenderDrawColor(g->ren, col.r, col.g, col.b, 255);
    SDL_RenderDrawLine(g->ren, (int)x, (int)y, (int)(x - dx * 6 + px * 3.5f), (int)(y - dy * 6 + py * 3.5f));
    SDL_RenderDrawLine(g->ren, (int)x, (int)y, (int)(x - dx * 6 - px * 3.5f), (int)(y - dy * 6 - py * 3.5f));
}

/* The model as a dependency graph: boxes are layers of the network, wires are "reads the output
 * of". Residual shortcuts show up as pink arcs over the boxes they skip; hover a box to light up
 * everything it reads and everything that reads it. */
static void draw_architecture(Gui *g, SDL_Rect r, const TrainConfig *cfg, const NetShapes *shapes) {
    draw_panel(g->ren, r);
    txt(g, g->font, "model graph", r.x + 10, r.y + 6, (SDL_Color){COL_TEXT_DIM});
    txt(g, g->font_small, "pink = skip / branch    hover a box to trace its wires", r.x + r.w - 420, r.y + r.h - 17,
        (SDL_Color){COL_TEXT_DIM});
    if (!cfg || cfg->n_blocks == 0) return;

    ArchGroup gr[ARCH_MAX_GROUPS];
    int n = build_arch(cfg, shapes, gr);
    int n_levels = 0, n_rows = 1;
    for (int i = 0; i < n; ++i) {
        if (gr[i].level + 1 > n_levels) n_levels = gr[i].level + 1;
        if (gr[i].row + 1 > n_rows) n_rows = gr[i].row + 1;
    }
    int area_x = r.x + 12, area_w = r.w - 24;
    int col_w = area_w / n_levels;
    int box_w = col_w - 14 > 124 ? 124 : col_w - 14;
    int top = r.y + 44; /* room above row 0 for the skip arcs */
    int avail = r.y + r.h - 22 - top;
    int box_h = 46, pitch = 0;
    if (n_rows > 1) {
        pitch = (avail - box_h) / (n_rows - 1);
        if (pitch < box_h + 4) { /* rows don't fit at full height: shrink the boxes instead */
            box_h = (avail - 4 * (n_rows - 1)) / n_rows;
            if (box_h < 30) box_h = 30;
            pitch = box_h + 4;
        }
    } else {
        top += (avail - box_h) / 2 - 4;
    }

    SDL_Rect box[ARCH_MAX_GROUPS];
    for (int i = 0; i < n; ++i) {
        int col_x = area_x + gr[i].level * col_w + (col_w - box_w) / 2;
        int row_y = top + (n_rows > 1 ? gr[i].row * pitch : 0);
        box[i] = (SDL_Rect){col_x, row_y, box_w, box_h};
    }
    SDL_Point mouse = {g->mouse_x, g->mouse_y};
    int hover = -1;
    for (int i = 0; i < n; ++i)
        if (SDL_PointInRect(&mouse, &box[i])) hover = i;

    for (int pass = 0; pass < 3; ++pass) {
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < gr[i].nsrc; ++k) {
                int a = gr[i].src[k];
                int arc = gr[a].row == gr[i].row && gr[i].level - gr[a].level > 1;
                int skip = arc || gr[a].row != gr[i].row;
                int lit = a == hover || i == hover;
                int my_pass = lit ? 2 : (skip ? 1 : 0);
                if (my_pass != pass) continue;
                SDL_Color col = lit ? (SDL_Color){240, 200, 90, 255}
                                    : (skip ? (SDL_Color){232, 108, 160, 255} : (SDL_Color){110, 150, 210, 255});
                if (arc) {
                    float x0 = box[a].x + box[a].w * 0.5f, x1 = box[i].x + box[i].w * 0.5f;
                    float y = (float)box[a].y;
                    float lift = 12.0f + 2.0f * (gr[i].level - gr[a].level - 1);
                    if (lift > 24.0f) lift = 24.0f;
                    draw_curve(g, x0, y, x0, y - lift * 1.35f, x1, y - lift * 1.35f, x1, y - 1, col);
                    draw_arrow_head(g, x1, y - 1, 0.0f, 1.0f, col);
                } else {
                    SDL_Point p0 = {box[a].x + box[a].w, box[a].y + box[a].h / 2};
                    SDL_Point p1 = {box[i].x, box[i].y + box[i].h / 2};
                    float dx = fmaxf(10.0f, (float)(p1.x - p0.x) * 0.5f);
                    draw_curve(g, (float)p0.x, (float)p0.y, p0.x + dx, (float)p0.y, p1.x - dx, (float)p1.y, (float)p1.x - 1,
                               (float)p1.y, col);
                    draw_arrow_head(g, (float)p1.x - 1, (float)p1.y, 1.0f, 0.0f, col);
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        draw_panel(g->ren, box[i]);
        SDL_Color acc = node_accent(gr[i].type);
        SDL_Rect bar = {box[i].x + 7, box[i].y + 1, box[i].w - 14, 3};
        fill_rrect(g->ren, bar, 1, acc);
        if (i == hover) rrect_bordered(g->ren, box[i], 7, (SDL_Color){COL_SURFACE2}, (SDL_Color){250, 204, 96, 255});
        SDL_RenderSetClipRect(g->ren, &box[i]);
        int compact = box_w < 70; /* narrow boxes (deep graphs): smaller name, short shape, no spaces */
        txt(g, compact ? g->font_small : g->font, gr[i].name, box[i].x + 5, box[i].y + (compact ? 5 : 3),
            (SDL_Color){COL_TEXT});
        txt(g, g->font_small, compact ? gr[i].l2c : gr[i].l2, box[i].x + 5, box[i].y + 19, (SDL_Color){COL_TEXT_DIM});
        char l3[24];
        snprintf(l3, sizeof(l3), "%s", gr[i].fold[0] ? gr[i].fold : gr[i].params);
        if (compact) {
            char *w = l3;
            for (char *rd = l3[0] == '+' ? l3 + 1 : l3; *rd; ++rd)
                if (*rd != ' ') *w++ = *rd;
            *w = '\0';
        }
        if (box_h >= 40) txt(g, g->font_small, l3, box[i].x + 5, box[i].y + 31, (SDL_Color){COL_TEXT_DIM});
        SDL_RenderSetClipRect(g->ren, NULL);
    }
}

static const char *palette_label(int type) {
    switch (type) {
    case BLK_CONV: return "conv";
    case BLK_RELU: return "relu";
    case BLK_FLATTEN: return "flat";
    case BLK_LINEAR: return "fc";
    case BLK_ADD: return "add";
    case BLK_BATCHNORM: return "bn";
    case BLK_MAXPOOL: return "pool";
    case BLK_GAP: return "gap";
    case BLK_UPSAMPLE: return "up";
    case BLK_CONCAT: return "cat";
    case BLK_RESHAPE: return "view";
    default: return "vae";
    }
}

/* A palette entry: the block's colour as a dot, then its name. */
static void draw_chip(Gui *g, SDL_Rect r, const char *label, SDL_Color accent, int hovered) {
    SDL_Color fill = hovered ? col_mix((SDL_Color){COL_SURFACE2}, accent, 0.18f) : (SDL_Color){COL_SURFACE2};
    SDL_Color border = hovered ? accent : (SDL_Color){COL_BORDER_HI};
    rrect_bordered(g->ren, r, 8, fill, border);
    fill_circle_aa(g->ren, r.x + 9, r.y + r.h / 2, 3, accent);
    txt(g, g->font_small, label, r.x + 17, r.y + (r.h - TTF_FontHeight(g->font_small)) / 2, (SDL_Color){COL_TEXT});
}

/* The node canvas: grid, wires and nodes are drawn at native size into canvas_tex (so text and fields need no
 * scaling logic) and the visible part is then scaled onto the screen. */
static void draw_canvas(Gui *g, SDL_Point mouse_screen) {
    SDL_Rect view = view_rect(g);
    SDL_Point mouse = to_view(g, mouse_screen.x, mouse_screen.y); /* the cursor in view space */
    SDL_Texture *prev = SDL_GetRenderTarget(g->ren);
    if (!g->canvas_tex) {
        g->canvas_tex = SDL_CreateTexture(g->ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_TARGET, CANVAS_TEX_W, CANVAS_TEX_H);
        if (g->canvas_tex) {
            SDL_SetTextureBlendMode(g->canvas_tex, SDL_BLENDMODE_BLEND);
            SDL_SetTextureScaleMode(g->canvas_tex, SDL_ScaleModeLinear);
        }
    }
    if (!g->canvas_tex) return;
    SDL_SetRenderTarget(g->ren, g->canvas_tex);
    SDL_SetRenderDrawColor(g->ren, 0, 0, 0, 0);
    SDL_RenderClear(g->ren);
    fill_rrect(g->ren, view, (int)(10.0f / g->zoom), (SDL_Color){13, 15, 22, 255}); /* rounded like the panel, at any zoom */
    SDL_RenderSetClipRect(g->ren, &view);

    /* dot grid that moves with the canvas (skipped when zoomed out far enough that it would be noise) */
    if (24.0f * g->zoom >= 8.0f) {
        SDL_SetRenderDrawColor(g->ren, 44, 50, 68, 255);
        int px = (int)floorf(g->pan_x), py = (int)floorf(g->pan_y);
        int off_x = ((px % 24) + 24) % 24, off_y = ((py % 24) + 24) % 24;
        for (int y = off_y; y < view.h; y += 24)
            for (int x = off_x; x < view.w; x += 24) SDL_RenderDrawPoint(g->ren, x, y);
    }

    /* The node under the cursor: its wires light up, which makes "what does this read, who reads it"
     * readable even where wires cross. */
    int hover = -1;
    if (SDL_PointInRect(&mouse_screen, &GRAPH_VIEW) && g->drag == DRAG_NONE) {
        for (int i = MAX_NODES - 1; i >= 0 && hover < 0; --i) {
            SDL_Rect r = node_rect(g, i);
            if (g->nodes[i].used && SDL_PointInRect(&mouse, &r)) hover = i;
        }
    }
    /* Wires are drawn in three passes so the interesting ones end up on top: ordinary
     * (blue), skip / branch (pink: the source isn't the block right before the reader in
     * evaluation order, e.g. a residual shortcut), then the hovered node's (gold). */
    for (int pass = 0; pass < 3; ++pass) {
        for (int t = 0; t < MAX_NODES; ++t) {
            if (!g->nodes[t].used) continue;
            for (int k = 0; k < node_nin(g->nodes[t].type); ++k) {
                int src = g->nodes[t].in[k];
                if (src < 0) continue;
                int src_id = g->nodes[src].type == NODE_INPUT ? 0 : g->node_block[src] + 1;
                int dst_id = g->nodes[t].type == NODE_OUTPUT ? g->chain_len + 1 : g->node_block[t] + 1;
                int on_path = (g->nodes[src].type == NODE_INPUT || g->node_block[src] >= 0) &&
                              (g->nodes[t].type == NODE_OUTPUT || g->node_block[t] >= 0);
                int skip = on_path && src_id != dst_id - 1;
                int lit = src == hover || t == hover;
                int my_pass = lit ? 2 : (skip ? 1 : 0);
                if (my_pass != pass) continue;
                SDL_Color col = lit ? (SDL_Color){240, 200, 90, 255}
                                    : (!on_path ? (SDL_Color){110, 116, 130, 255}
                                                : (skip ? (SDL_Color){232, 108, 160, 255} : (SDL_Color){57, 135, 229, 255}));
                draw_wire(g, node_out_port(g, src), node_in_port(g, t, k), col);
            }
        }
    }
    for (int i = 0; i < MAX_NODES; ++i)
        if (g->nodes[i].used) draw_node(g, i, mouse);

    if (g->drag == DRAG_LINK) draw_wire(g, node_out_port(g, g->drag_node), mouse, (SDL_Color){250, 204, 96, 255});

    SDL_RenderSetClipRect(g->ren, NULL);
    SDL_SetRenderTarget(g->ren, prev);
    SDL_Rect dst = {GRAPH_VIEW.x, GRAPH_VIEW.y, GRAPH_VIEW.w, GRAPH_VIEW.h};
    SDL_RenderCopy(g->ren, g->canvas_tex, &view, &dst);
}

/* Bottom-right of the canvas: zoom out / in, the zoom level, and "fit" (show every node). */
static const SDL_Rect ZOOM_BAR = {GRAPH_VIEW.x + GRAPH_VIEW.w - 200, GRAPH_VIEW.y + GRAPH_VIEW.h - 36, 192, 30};
static SDL_Rect zoom_btn(int k) { /* 0 = minus, 1 = plus, 2 = fit */
    static const int xs[3] = {4, 108, 140}, ws[3] = {28, 28, 48};
    SDL_Rect r = {ZOOM_BAR.x + xs[k], ZOOM_BAR.y + 3, ws[k], 24};
    return r;
}

static void draw_zoom_controls(Gui *g, SDL_Point mouse) {
    rrect_bordered(g->ren, ZOOM_BAR, 9, (SDL_Color){20, 23, 32, 235}, (SDL_Color){COL_BORDER_HI});
    static const char *lab[3] = {"-", "+", "fit"};
    for (int k = 0; k < 3; ++k) {
        SDL_Rect r = zoom_btn(k);
        int hov = SDL_PointInRect(&mouse, &r);
        fill_rrect(g->ren, r, 7, hov ? (SDL_Color){52, 62, 92, 255} : (SDL_Color){COL_SURFACE2});
        int tw = 0;
        TTF_SizeText(g->font, lab[k], &tw, NULL);
        txt(g, g->font, lab[k], r.x + (r.w - tw) / 2, r.y + (r.h - TTF_FontHeight(g->font)) / 2, (SDL_Color){COL_TEXT});
    }
    char buf[16];
    snprintf(buf, sizeof(buf), "%d%%", (int)(g->zoom * 100.0f + 0.5f));
    int tw = 0;
    TTF_SizeText(g->font_small, buf, &tw, NULL);
    txt(g, g->font_small, buf, ZOOM_BAR.x + 36 + (68 - tw) / 2, ZOOM_BAR.y + 8, (SDL_Color){COL_TEXT_DIM});
}

static void draw_graph(Gui *g) {
    SDL_Point mouse = {g->mouse_x, g->mouse_y};
    draw_panel(g->ren, PANEL_GRAPH);
    for (int i = 0; i < palette_count(g); ++i) {
        SDL_Rect pr = palette_rect(i);
        draw_chip(g, pr, palette_label(palette_type(g, i)), node_accent(palette_type(g, i)), SDL_PointInRect(&mouse, &pr));
    }
    draw_button(g, BTN_ARRANGE, "arrange", g->font, SDL_PointInRect(&mouse, &BTN_ARRANGE), 1);
    draw_button(g, BTN_PRESETS, g->preset_open ? "presets  ^" : "presets  v", g->font,
                SDL_PointInRect(&mouse, &BTN_PRESETS) || g->preset_open, 1);

    draw_canvas(g, mouse);
    draw_zoom_controls(g, mouse);
    if (g->drag == DRAG_PAL && g->drag_moved) {
        SDL_Rect ghost = {mouse.x - (int)(NODE_W * g->zoom / 2), mouse.y - (int)(12 * g->zoom), (int)(NODE_W * g->zoom),
                          (int)(node_height(g->drag_type) * g->zoom)};
        SDL_Color a = node_accent(g->drag_type);
        SDL_SetRenderDrawColor(g->ren, a.r, a.g, a.b, 90);
        SDL_RenderFillRect(g->ren, &ghost);
        SDL_SetRenderDrawColor(g->ren, a.r, a.g, a.b, 255);
        SDL_RenderDrawRect(g->ren, &ghost);
    }

    SDL_RenderSetClipRect(g->ren, &GRAPH_VIEW);
    SDL_Color dim = {128, 136, 156, 255};
    SDL_Rect hint_bg = {GRAPH_VIEW.x + 6, GRAPH_VIEW.y + GRAPH_VIEW.h - 58, 318, 50};
    fill_rrect(g->ren, hint_bg, 8, (SDL_Color){13, 15, 22, 215});
    txt(g, g->font_small, "click a block to append it, or drag it onto the canvas", GRAPH_VIEW.x + 14,
        GRAPH_VIEW.y + GRAPH_VIEW.h - 50, dim);
    txt(g, g->font_small, "wire: drag dot to dot; an output can feed many blocks", GRAPH_VIEW.x + 14,
        GRAPH_VIEW.y + GRAPH_VIEW.h - 34, dim);
    txt(g, g->font_small, "pink = skip / branch; hover to trace; scroll = zoom", GRAPH_VIEW.x + 14,
        GRAPH_VIEW.y + GRAPH_VIEW.h - 18, dim);
    SDL_RenderSetClipRect(g->ren, NULL);

    if (g->preset_open) { /* the menu floats over the canvas */
        SDL_Rect bg = {BTN_PRESETS.x - 4, BTN_PRESETS.y + BTN_PRESETS.h + 1, BTN_PRESETS.w + 8, preset_count(g) * 30 + 8};
        draw_shadow(g->ren, bg, 9);
        rrect_bordered(g->ren, bg, 9, (SDL_Color){COL_SURFACE2}, (SDL_Color){COL_LOSS});
        static const char *hint[NUM_PRESETS] = {"4 conv + fc", "residual", "per-pixel masks", "generative"};
        for (int i = 0; i < preset_count(g); ++i) {
            SDL_Rect r = preset_rect(i);
            int id = preset_at(g, i);
            if (SDL_PointInRect(&mouse, &r)) {
                SDL_Rect hl = {r.x, r.y, r.w, r.h};
                fill_rrect(g->ren, hl, 6, (SDL_Color){52, 62, 92, 255});
            }
            txt(g, g->font, PRESET_NAMES[id], r.x + 10, r.y + 4, (SDL_Color){COL_TEXT});
            int tw = 0;
            TTF_SizeText(g->font_small, hint[id], &tw, NULL);
            txt(g, g->font_small, hint[id], r.x + r.w - tw - 8, r.y + 8, (SDL_Color){COL_TEXT_DIM});
        }
    }
}

static void draw_form(Gui *g) {
    SDL_Point mouse = {g->mouse_x, g->mouse_y};
    form_layout(g);

    /* training parameters */
    draw_panel(g->ren, PANEL_TRAIN);
    static const char *prob_label[3] = {"Classification", "Segmentation", "Generation"};
    for (int i = 0; i < 3; ++i) { /* the problem comes first: a segmented control */
        SDL_Rect pb = PROBLEM_BTN[i];
        int on = g->problem == i, hov = SDL_PointInRect(&mouse, &pb);
        SDL_Color fill = on ? (SDL_Color){COL_LOSS} : (hov ? (SDL_Color){36, 42, 60, 255} : (SDL_Color){COL_SURFACE2});
        rrect_bordered(g->ren, pb, 9, fill, on ? (SDL_Color){COL_LOSS} : (SDL_Color){COL_BORDER_HI});
        int tw = 0;
        TTF_SizeText(g->font_med, prob_label[i], &tw, NULL);
        txt(g, g->font_med, prob_label[i], pb.x + (pb.w - tw) / 2, pb.y + (pb.h - TTF_FontHeight(g->font_med)) / 2,
            on ? (SDL_Color){10, 14, 28, 255} : (SDL_Color){COL_TEXT_DIM});
    }
    static const char *labels[NUM_GENERAL_FIELDS] = {"dataset", "epochs", "batch size", "learning rate", "optimizer (0/1)", "momentum",
                                                       "normalize (0/1)", "augment (0-2)", "seed", "fc viz pool",
                                                       "weights every N"};
    SDL_Color dim = (SDL_Color){COL_TEXT_DIM};
    for (int i = 0; i < NUM_GENERAL_FIELDS; ++i) {
        txt(g, g->font, labels[i], PANEL_TRAIN.x + 18, g->fields[i].rect.y + 5, dim);
        draw_field(g, i);
    }
    int hy = 58 + NUM_GENERAL_FIELDS * FIELD_PITCH + 2;
    const DatasetInfo *fds = form_dataset(g);
    char info[64] = "";
    if (fds) snprintf(info, sizeof(info), "%dx%d%s px  -  %d images  -  %d classes", fds->cols, fds->rows,
                      fds->channels == 3 ? " colour" : "", fds->count, fds->n_classes);
    SDL_Rect pill = {PANEL_TRAIN.x + 16, hy - 2, PANEL_TRAIN.w - 32, 24};
    fill_rrect(g->ren, pill, 8, col_mix((SDL_Color){COL_SURFACE}, (SDL_Color){COL_LOSS}, 0.14f));
    txt(g, g->font_small, info, pill.x + 10, pill.y + 5, (SDL_Color){COL_LOSS});
    txt(g, g->font_small, "picking a dataset loads its suggested settings", PANEL_TRAIN.x + 18, hy + 30, dim);
    txt(g, g->font_small, "normalize: zero-mean, unit-variance pixels", PANEL_TRAIN.x + 18, hy + 44, dim);
    txt(g, g->font_small, "augment: 1 flips both ways, 2 flips + shifts", PANEL_TRAIN.x + 18, hy + 58, dim);
    txt(g, g->font_small, "optimizer: 0 = SGD + momentum, 1 = Adam (lr ~0.001)", PANEL_TRAIN.x + 18, hy + 72, dim);
    txt(g, g->font_small, "fc viz pool, weights every N: viewer only", PANEL_TRAIN.x + 18, hy + 86, dim);

    int ry = hy + 108;
    SDL_SetRenderDrawColor(g->ren, COL_BORDER);
    SDL_RenderDrawLine(g->ren, PANEL_TRAIN.x + 16, ry - 8, PANEL_TRAIN.x + PANEL_TRAIN.w - 16, ry - 8);
    txt(g, g->font_med, "Model rules", PANEL_TRAIN.x + 18, ry, (SDL_Color){COL_TEXT});
    static const char *rules[] = {"conv / pool / gap need an image, fc a flat input (flatten / gap)",
                                  "add sums, cat joins channels (needs the same height and width)",
                                  "every block's output must be read by another",
                                  "end: fc = #classes | conv map = #classes | image (vae)",
                                  "x deletes a block, re-joining its neighbours"};
    for (int i = 0; i < 5; ++i) {
        fill_circle_aa(g->ren, PANEL_TRAIN.x + 22, ry + 30 + i * 15, 2, (SDL_Color){COL_LOSS});
        txt(g, g->font_small, rules[i], PANEL_TRAIN.x + 32, ry + 23 + i * 15, dim);
    }

    draw_graph(g);

    /* status + start */
    char status[300];
    SDL_Color status_col = (SDL_Color){COL_BAD};
    if (g->worker_gone) {
        snprintf(status, sizeof(status), "train_worker is not running - quit and start it again (make train_gui)");
    } else if (!g->form_valid) {
        snprintf(status, sizeof(status), "%s", g->form_err);
    } else if (g->worker_msg[0]) {
        snprintf(status, sizeof(status), "%s", g->worker_msg);
    } else {
        status_col = (SDL_Color){COL_GOOD};
        snprintf(status, sizeof(status), "ready - %ld trainable parameters", count_parameters(&g->form_cfg, &g->form_shapes));
    }
    fill_circle_aa(g->ren, 418, 598, 4, status_col);
    draw_wrapped(g, g->font, status, 430, 589, 50, 20, 3, status_col);
    if (g->run_active) {
        txt(g, g->font_small, "a run is active: applying replaces it", 410, 628, dim);
        draw_button(g, BTN_BACK, "< back to training view", g->font, SDL_PointInRect(&mouse, &BTN_BACK), 1);
        if (g->pred_hdr.n > 0 || g->gen_hdr.n_recon > 0)
            draw_button(g, BTN_VIEW_PRED, g->run_cfg.task == TASK_GENERATE ? "view generations" : "view predictions", g->font,
                        SDL_PointInRect(&mouse, &BTN_VIEW_PRED), 1);
    }
    int can_start = g->form_valid && !g->worker_gone;
    draw_button_primary(g, BTN_START, g->run_active ? "Apply & restart" : "Start training", g->font_big,
                        SDL_PointInRect(&mouse, &BTN_START), can_start);
    if (g->dropdown_open) draw_dropdown(g);
}

/* -- messages from the worker -------------------------------------------- */

static int model_info_ok(const ModelInfo *m) {
    NetShapes s;
    char err[64];
    if (train_cfg_derive(&m->cfg, &s, err, sizeof(err)) != 0) return 0;
    int weighted = 0;
    for (int i = 0; i < m->cfg.n_blocks; ++i)
        if (m->cfg.blocks[i].type == BLK_CONV || m->cfg.blocks[i].type == BLK_LINEAR) weighted++;
    if (m->n_viz != weighted || m->n_viz > MAX_VIZ_LAYERS) return 0;
    long used = 0;
    for (int i = 0; i < m->n_viz; ++i) {
        const LayerViz *lv = &m->layers[i];
        long n;
        if (lv->kind == VIZ_CONV_GRID) {
            if (lv->count < 1 || lv->count > MAX_CONV_CHANNELS || lv->ksize < 1 || lv->ksize > MAX_KERNEL || lv->cols < 1)
                return 0;
            n = (long)lv->count * lv->ksize * lv->ksize;
        } else if (lv->kind == VIZ_MATRIX) {
            if (lv->count < 1 || lv->cols < 1 || (long)lv->count * lv->cols > 4 * 65536) return 0;
            n = (long)lv->count * lv->cols;
        } else {
            return 0;
        }
        if (lv->offset != used) return 0;
        used += n;
    }
    return used == m->total_floats;
}

/* The sample texture and pixel buffer follow the dataset's image size, which
 * can change from one run to the next. */
static void ensure_sample(Gui *g) {
    int w = g->run_cfg.img_w, h = g->run_cfg.img_h, c = g->run_cfg.img_c;
    if (w < 1 || h < 1 || c < 1) return;
    int want_masks = g->run_cfg.task == TASK_SEGMENT || g->run_cfg.task == TASK_GENERATE;
    if (g->digit_tex && g->digit_w == w && g->digit_h == h && g->digit_c == c && (g->mask_tex[0] != NULL) == want_masks) return;
    if (g->digit_tex) SDL_DestroyTexture(g->digit_tex);
    g->digit_tex = SDL_CreateTexture(g->ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, w, h);
    if (g->digit_tex) SDL_SetTextureScaleMode(g->digit_tex, SDL_ScaleModeNearest);
    g->digit_w = w;
    g->digit_h = h;
    g->digit_c = c;
    for (int k = 0; k < 2; ++k) {
        if (g->mask_tex[k]) SDL_DestroyTexture(g->mask_tex[k]);
        g->mask_tex[k] = NULL;
        if (g->run_cfg.task == TASK_SEGMENT || (g->run_cfg.task == TASK_GENERATE && k == 0)) {
            g->mask_tex[k] = SDL_CreateTexture(g->ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, w, h);
            if (g->mask_tex[k]) SDL_SetTextureScaleMode(g->mask_tex[k], SDL_ScaleModeNearest);
        }
    }
    free(g->sample);
    g->sample = (unsigned char *)calloc(train_sample_bytes(&g->run_cfg), 1);
}

static void refresh_run_ds(Gui *g) {
    const DatasetInfo *d = ds_find(g, g->run_cfg.dataset);
    if (d) g->run_ds = *d;
    else memset(&g->run_ds, 0, sizeof(g->run_ds));
}

static void handle_message(Gui *g, uint32_t type, const unsigned char *p, uint32_t bytes) {
    if (type == MSG_MODEL) {
        if (bytes != sizeof(ModelInfo)) return;
        ModelInfo info;
        memcpy(&info, p, sizeof(info));
        if (!model_info_ok(&info)) {
            snprintf(g->worker_msg, sizeof(g->worker_msg), "worker sent a model description this viewer can't use");
            return;
        }
        g->model = info;
        g->run_cfg = info.cfg;
        train_cfg_derive(&g->run_cfg, &g->run_shapes, NULL, 0);
        refresh_run_ds(g);
        ensure_sample(g);
        free(g->weights);
        g->weights = (float *)calloc((size_t)(info.total_floats > 0 ? info.total_floats : 1), sizeof(float));
        g->weights_valid = 0;
        g->has_model = 1;
        g->has_record = 0;
        free(g->pred_data); /* predictions belong to the model that made them */
        g->pred_data = NULL;
        g->pred_hdr.n = 0;
        free(g->gen_data);
        g->gen_data = NULL;
        memset(&g->gen_hdr, 0, sizeof(g->gen_hdr));
        g->pred_page = 0;
        if (g->weights_tab >= g->model.n_viz) g->weights_tab = 0;
        gui_clear_history(g);
    } else if (type == MSG_PROGRESS) {
        if (!g->has_model) return;
        size_t wbytes = sizeof(float) * (size_t)g->model.total_floats;
        size_t img_bytes = train_sample_bytes(&g->run_cfg);
        size_t base = sizeof(Progress) + img_bytes;
        if (!g->sample || bytes < base || (bytes != base && bytes != base + wbytes)) return;
        Progress prog;
        memcpy(&prog, p, sizeof(prog));
        if (prog.has_weights) {
            if (bytes != base + wbytes) return;
            memcpy(g->weights, p + base, wbytes);
            g->weights_valid = 1;
        } else if (bytes != base) {
            return;
        }
        memcpy(g->sample, p + sizeof(Progress), img_bytes);
        g->prog = prog;
        g->has_record = 1;
        gui_push_point(g, prog.loss);
    } else if (type == MSG_GENERATED) {
        if (!g->has_model || g->run_cfg.task != TASK_GENERATE || bytes < sizeof(GenHeader)) return;
        GenHeader h;
        memcpy(&h, p, sizeof(h));
        size_t img = (size_t)g->run_cfg.img_h * g->run_cfg.img_w * g->run_cfg.img_c;
        if (h.n_recon < 1 || h.n_recon > 64 || h.n_samples < 0 || h.n_samples > 64 || h.n_rows < 0 || h.n_rows > 8 ||
            h.n_steps < 0 || h.n_steps > 32 ||
            bytes != sizeof(GenHeader) + img * ((size_t)2 * h.n_recon + h.n_samples + (size_t)h.n_rows * h.n_steps))
            return;
        free(g->gen_data);
        g->gen_data = (unsigned char *)malloc(bytes);
        if (!g->gen_data) { memset(&g->gen_hdr, 0, sizeof(g->gen_hdr)); return; }
        memcpy(g->gen_data, p, bytes);
        g->gen_hdr = h;
    } else if (type == MSG_PREDICTIONS) {
        if (!g->has_model || bytes < sizeof(PredHeader)) return;
        PredHeader h;
        memcpy(&h, p, sizeof(h));
        size_t record = sizeof(PredSample) + train_sample_bytes(&g->run_cfg);
        if (h.n < 1 || h.n > PRED_MAX_SAMPLES || bytes != sizeof(PredHeader) + (size_t)h.n * record) return;
        free(g->pred_data);
        g->pred_data = (unsigned char *)malloc(bytes);
        if (!g->pred_data) { g->pred_hdr.n = 0; return; }
        memcpy(g->pred_data, p, bytes);
        g->pred_hdr = h;
    } else if (type == MSG_ERROR) {
        size_t n = bytes < sizeof(g->worker_msg) ? bytes : sizeof(g->worker_msg) - 1;
        memcpy(g->worker_msg, p, n);
        g->worker_msg[n] = '\0';
        g->run_active = 0;
        g->has_record = 0;
        g->weights_valid = 0;
        g->screen = SCREEN_SETUP;
        SDL_StartTextInput();
    }
}

/* Re-assembles framed messages across partial non-blocking reads. */
typedef struct Rx {
    unsigned char hdr[sizeof(MsgHeader)];
    size_t hdr_have;
    MsgHeader h;
    unsigned char *buf;
    size_t cap;
    size_t have;
    int in_payload;
} Rx;

static void rx_pump(Rx *r, int fd, Gui *g) {
    long budget = 8 * 1024 * 1024; /* keep one frame's worth of reading bounded so the UI stays responsive */
    while (budget > 0 && !g->worker_gone) {
        if (!r->in_payload) {
            ssize_t n = read(fd, r->hdr + r->hdr_have, sizeof(MsgHeader) - r->hdr_have);
            if (n == 0) { g->worker_gone = 1; return; }
            if (n < 0) { if (errno == EINTR) continue; return; }
            r->hdr_have += (size_t)n;
            budget -= n;
            if (r->hdr_have < sizeof(MsgHeader)) continue;
            memcpy(&r->h, r->hdr, sizeof(MsgHeader));
            if (r->h.magic != TRAIN_MAGIC || r->h.bytes > MSG_MAX_BYTES) {
                snprintf(g->worker_msg, sizeof(g->worker_msg), "protocol error on the data stream (mismatched worker build?)");
                g->worker_gone = 1;
                return;
            }
            if (r->h.bytes > r->cap) {
                unsigned char *nb = (unsigned char *)realloc(r->buf, r->h.bytes);
                if (!nb) { g->worker_gone = 1; return; }
                r->buf = nb;
                r->cap = r->h.bytes;
            }
            r->have = 0;
            r->in_payload = 1;
        }
        if (r->have < r->h.bytes) {
            ssize_t n = read(fd, r->buf + r->have, r->h.bytes - r->have);
            if (n == 0) { g->worker_gone = 1; return; }
            if (n < 0) { if (errno == EINTR) continue; return; }
            r->have += (size_t)n;
            budget -= n;
            if (r->have < r->h.bytes) continue;
        }
        handle_message(g, r->h.type, r->buf, r->h.bytes);
        r->in_payload = 0;
        r->hdr_have = 0;
    }
}

/* -- weights window ------------------------------------------------------ */

#define WEIGHTS_WIN_W 700
#define WEIGHTS_WIN_H 600
#define WEIGHTS_TAB_Y 20
#define WEIGHTS_TAB_H 44
#define WEIGHTS_CONTENT_W (WEIGHTS_WIN_W - 40)
#define WEIGHTS_CONTENT_H (WEIGHTS_WIN_H - (WEIGHTS_TAB_Y + WEIGHTS_TAB_H + 16) - 20)

/* Pure function of (i, n) (no Gui) so render_weights_window (draws tabs) and
 * pump_events (hit-tests clicks on them) use the exact same rects. */
static SDL_Rect weights_tab_rect(int i, int n) {
    int bar_w = WEIGHTS_WIN_W - 40;
    int tab_w = bar_w / (n > 0 ? n : 1);
    SDL_Rect r = {20 + i * tab_w, WEIGHTS_TAB_Y, tab_w - 4, WEIGHTS_TAB_H};
    return r;
}

static void close_weights_window(Gui *g) {
    if (!g->weights_win) return;
    for (int i = 0; i < NUM_WLABEL_SLOTS; ++i) {
        if (g->weights_label_tex[i]) SDL_DestroyTexture(g->weights_label_tex[i]);
        g->weights_label_tex[i] = NULL;
    }
    if (g->weights_ren) SDL_DestroyRenderer(g->weights_ren);
    g->weights_ren = NULL;
    SDL_DestroyWindow(g->weights_win);
    g->weights_win = NULL;
}

static void open_weights_window(Gui *g) {
    if (g->weights_win) { SDL_RaiseWindow(g->weights_win); return; }

    g->weights_win = SDL_CreateWindow("tinyDL - layer weights (live)", SDL_WINDOWPOS_UNDEFINED,
                                       SDL_WINDOWPOS_UNDEFINED, WEIGHTS_WIN_W, WEIGHTS_WIN_H,
                                       SDL_WINDOW_SHOWN | SDL_WINDOW_ALWAYS_ON_TOP /* stays visible over a fullscreen main window */);
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


/* -- 3D weights view ----------------------------------------------------- */

/* The viewer has no 3D library: this is a tiny painter's-algorithm renderer.
 * Every surface becomes a flat-shaded quad (4 projected corners), all quads
 * of a frame are sorted far-to-near and handed to SDL_RenderGeometry in one
 * call. conv filters are drawn as bars (one bar per weight, height = value);
 * fc weight matrices as a height-field surface. */
#define CAM_DIST 3.4f
#define VIEW3D_HEIGHT 0.5f /* world height of the largest |weight| on small layers */
#define VIEW3D_GRID_MAX 64 /* fc matrices are averaged down to at most this many cells per side */

typedef struct V3 {
    float x, y, z;
} V3;

typedef struct Cam {
    float cosy, siny, cosp, sinp, focal, ox, oy;
} Cam;

typedef struct Face {
    float depth; /* view-space z of the centre: larger = farther */
    SDL_FPoint p[4];
    SDL_Color c[4];
} Face;

static Face *g_faces;
static int g_faces_n, g_faces_cap;
static SDL_Vertex *g_verts;
static int *g_idx;
static int g_geo_cap;

/* World -> view space (yaw about the vertical axis, then pitch down onto the
 * scene). +z in view space points away from the camera. */
static V3 cam_rot(const Cam *c, V3 p) {
    float x1 = p.x * c->cosy + p.z * c->siny;
    float z1 = -p.x * c->siny + p.z * c->cosy;
    V3 r = {x1, p.y * c->cosp + z1 * c->sinp, -p.y * c->sinp + z1 * c->cosp};
    return r;
}

static SDL_FPoint cam_proj(const Cam *c, V3 v) {
    float d = v.z + CAM_DIST;
    if (d < 0.3f) d = 0.3f;
    SDL_FPoint r = {c->ox + v.x * c->focal / d, c->oy - v.y * c->focal / d};
    return r;
}

static SDL_Color color_lerp(SDL_Color a, SDL_Color b, float t) {
    SDL_Color r = {(Uint8)(a.r + (b.r - a.r) * t), (Uint8)(a.g + (b.g - a.g) * t), (Uint8)(a.b + (b.b - a.b) * t), 255};
    return r;
}

/* t in [-1,1]: blue for negative weights, warm red for positive, gray at 0. */
static SDL_Color color_signed(float t) {
    if (t > 1.0f) t = 1.0f;
    if (t < -1.0f) t = -1.0f;
    SDL_Color zero = {176, 181, 192, 255};
    SDL_Color neg = {57, 135, 229, 255};
    SDL_Color pos = {236, 104, 72, 255};
    return color_lerp(zero, t >= 0 ? pos : neg, fabsf(t));
}

/* t in [0,1], for the non-negative mean|w| maps. */
static SDL_Color color_seq(float t) {
    if (t > 1.0f) t = 1.0f;
    if (t < 0.0f) t = 0.0f;
    SDL_Color lo = {52, 84, 168, 255};
    SDL_Color hi = {248, 208, 84, 255};
    return color_lerp(lo, hi, t);
}

static void faces_reserve(int extra) {
    if (g_faces_n + extra <= g_faces_cap) return;
    int cap = g_faces_cap ? g_faces_cap : 1024;
    while (cap < g_faces_n + extra) cap *= 2;
    g_faces = (Face *)realloc(g_faces, sizeof(Face) * (size_t)cap);
    g_faces_cap = cap;
}

/* `v` are the 4 view-space corners, `n` the unit view-space normal facing the camera. */
static void face_push(const Cam *cam, const V3 v[4], const SDL_Color col[4], V3 n) {
    static const V3 light = {-0.4f, 0.8f, -0.45f};
    float diffuse = n.x * light.x + n.y * light.y + n.z * light.z;
    if (diffuse < 0.0f) diffuse = 0.0f;
    float shade = 0.42f + 0.58f * diffuse;
    faces_reserve(1);
    Face *f = &g_faces[g_faces_n++];
    f->depth = 0.0f;
    for (int i = 0; i < 4; ++i) {
        f->p[i] = cam_proj(cam, v[i]);
        float r = col[i].r * shade, gr = col[i].g * shade, b = col[i].b * shade;
        f->c[i] = (SDL_Color){(Uint8)fminf(r, 255.0f), (Uint8)fminf(gr, 255.0f), (Uint8)fminf(b, 255.0f), 255};
        f->depth += v[i].z * 0.25f;
    }
}

/* One side of an axis-aligned box: drawn only if it faces the camera. */
static void add_box_face(const Cam *cam, V3 a, V3 b, V3 c, V3 d, V3 normal, SDL_Color col) {
    V3 v[4] = {cam_rot(cam, a), cam_rot(cam, b), cam_rot(cam, c), cam_rot(cam, d)};
    V3 nv = cam_rot(cam, normal);
    V3 mid = {(v[0].x + v[2].x) * 0.5f, (v[0].y + v[2].y) * 0.5f, (v[0].z + v[2].z) * 0.5f + CAM_DIST};
    if (nv.x * mid.x + nv.y * mid.y + nv.z * mid.z >= 0.0f) return;
    SDL_Color cols[4] = {col, col, col, col};
    face_push(cam, v, cols, nv);
}

static void add_box(const Cam *cam, float x0, float x1, float y0, float y1, float z0, float z1, SDL_Color col) {
    add_box_face(cam, (V3){x0, y1, z0}, (V3){x1, y1, z0}, (V3){x1, y1, z1}, (V3){x0, y1, z1}, (V3){0, 1, 0}, col);
    add_box_face(cam, (V3){x0, y0, z0}, (V3){x1, y0, z0}, (V3){x1, y1, z0}, (V3){x0, y1, z0}, (V3){0, 0, -1}, col);
    add_box_face(cam, (V3){x0, y0, z1}, (V3){x1, y0, z1}, (V3){x1, y1, z1}, (V3){x0, y1, z1}, (V3){0, 0, 1}, col);
    add_box_face(cam, (V3){x0, y0, z0}, (V3){x0, y0, z1}, (V3){x0, y1, z1}, (V3){x0, y1, z0}, (V3){-1, 0, 0}, col);
    add_box_face(cam, (V3){x1, y0, z0}, (V3){x1, y0, z1}, (V3){x1, y1, z1}, (V3){x1, y1, z0}, (V3){1, 0, 0}, col);
}

/* Two-sided quad (height-field cells, floors): the normal comes from the
 * corners and is flipped to face the camera. `behind_all` forces it to be
 * drawn first, under everything else. */
static void add_surface_quad(const Cam *cam, V3 a, V3 b, V3 c, V3 d, const SDL_Color col[4], int behind_all) {
    V3 v[4] = {cam_rot(cam, a), cam_rot(cam, b), cam_rot(cam, c), cam_rot(cam, d)};
    V3 e1 = {v[1].x - v[0].x, v[1].y - v[0].y, v[1].z - v[0].z};
    V3 e2 = {v[3].x - v[0].x, v[3].y - v[0].y, v[3].z - v[0].z};
    V3 n = {e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x};
    float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len < 1e-9f) return;
    n.x /= len; n.y /= len; n.z /= len;
    if (n.x * v[0].x + n.y * v[0].y + n.z * (v[0].z + CAM_DIST) > 0.0f) { n.x = -n.x; n.y = -n.y; n.z = -n.z; }
    face_push(cam, v, col, n);
    if (behind_all) g_faces[g_faces_n - 1].depth = 1e9f;
}

static void add_floor(const Cam *cam, float x0, float x1, float z0, float z1) {
    SDL_Color floor_col = {30, 34, 46, 255};
    SDL_Color cols[4] = {floor_col, floor_col, floor_col, floor_col};
    add_surface_quad(cam, (V3){x0, 0, z0}, (V3){x1, 0, z0}, (V3){x1, 0, z1}, (V3){x0, 0, z1}, cols, 1);
}

static int face_cmp_far_first(const void *a, const void *b) {
    float da = ((const Face *)a)->depth, db = ((const Face *)b)->depth;
    return (da < db) - (da > db);
}

static void flush_faces(Gui *g, SDL_Rect clip) {
    int n = g_faces_n;
    if (n == 0) return;
    qsort(g_faces, (size_t)n, sizeof(Face), face_cmp_far_first);
    if (4 * n > g_geo_cap) {
        g_geo_cap = 4 * n;
        g_verts = (SDL_Vertex *)realloc(g_verts, sizeof(SDL_Vertex) * (size_t)g_geo_cap);
        g_idx = (int *)realloc(g_idx, sizeof(int) * 6 * (size_t)n);
    }
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 4; ++k) {
            g_verts[4 * i + k].position = g_faces[i].p[k];
            g_verts[4 * i + k].color = g_faces[i].c[k];
            g_verts[4 * i + k].tex_coord = (SDL_FPoint){0, 0};
        }
        static const int quad[6] = {0, 1, 2, 0, 2, 3};
        for (int k = 0; k < 6; ++k) g_idx[6 * i + k] = 4 * i + quad[k];
    }
    SDL_RenderSetClipRect(g->weights_ren, &clip);
    SDL_RenderGeometry(g->weights_ren, NULL, g_verts, 4 * n, g_idx, 6 * n);
    SDL_RenderSetClipRect(g->weights_ren, NULL);
}

/* conv layer: the filters tiled on a floor, one bar per weight. Heights share
 * one scale across the whole layer so filters can be compared by eye. */
static void build_conv_3d(const Cam *cam, const LayerViz *lv, const float *data) {
    int k = lv->ksize, count = lv->count, cols = lv->cols;
    int rows = (count + cols - 1) / cols;
    float gap = 1.0f;
    float span_x = cols * k + (cols - 1) * gap, span_z = rows * k + (rows - 1) * gap;
    float s = 2.0f / fmaxf(span_x, span_z);
    float hmax = 1e-12f;
    for (int i = 0; i < count * k * k; ++i) hmax = fmaxf(hmax, fabsf(data[i]));
    float height = fminf(VIEW3D_HEIGHT, 1.5f * s * k);
    faces_reserve(count * (k * k * 3 + 1) + 8);

    for (int f = 0; f < count; ++f) {
        float x0 = -span_x * 0.5f + (f % cols) * (k + gap);
        float z0 = -span_z * 0.5f + (rows - 1 - f / cols) * (k + gap); /* filter row 0 at the far edge, like the flat view */
        add_floor(cam, x0 * s, (x0 + k) * s, z0 * s, (z0 + k) * s);
        for (int ky = 0; ky < k; ++ky) {
            for (int kx = 0; kx < k; ++kx) {
                float t = data[(f * k + ky) * k + kx] / hmax;
                float h = t * height;
                float y0 = h >= 0 ? 0.0f : h, y1 = h >= 0 ? h : 0.0f;
                if (y1 - y0 < 0.004f) y1 = y0 + 0.004f; /* keep zero weights visible as a sliver */
                float bz = z0 + (k - 1 - ky);
                SDL_Color col = lv->full_detail ? color_signed(t) : color_seq(t);
                add_box(cam, (x0 + kx + 0.07f) * s, (x0 + kx + 0.93f) * s, y0, y1, (bz + 0.07f) * s, (bz + 0.93f) * s,
                        col);
            }
        }
    }
}

static void resample_avg(const float *src, int sr, int sc, int dr, int dc, float *dst) {
    for (int r = 0; r < dr; ++r) {
        int r0 = (int)((long)r * sr / dr), r1 = (int)((long)(r + 1) * sr / dr);
        if (r1 <= r0) r1 = r0 + 1;
        for (int c = 0; c < dc; ++c) {
            int c0 = (int)((long)c * sc / dc), c1 = (int)((long)(c + 1) * sc / dc);
            if (c1 <= c0) c1 = c0 + 1;
            float sum = 0.0f;
            for (int y = r0; y < r1; ++y)
                for (int x = c0; x < c1; ++x) sum += src[y * sc + x];
            dst[r * dc + c] = sum / (float)((r1 - r0) * (c1 - c0));
        }
    }
}

/* fc weight matrix: a height-field surface, averaged down to a grid the
 * software renderer can redraw every frame. */
static void build_matrix_3d(const Cam *cam, const LayerViz *lv, const float *data) {
    static float grid[VIEW3D_GRID_MAX * VIEW3D_GRID_MAX];
    int rows = lv->count, cols = lv->cols;
    int R = rows < VIEW3D_GRID_MAX ? rows : VIEW3D_GRID_MAX;
    int C = cols < VIEW3D_GRID_MAX ? cols : VIEW3D_GRID_MAX;
    if (R < 2) R = 2;
    if (C < 2) C = 2;
    resample_avg(data, rows, cols, R, C, grid);

    float hmax = 1e-12f;
    for (int i = 0; i < R * C; ++i) hmax = fmaxf(hmax, fabsf(grid[i]));
    float ratio = (float)cols / (float)rows;
    float hx = ratio >= 1.0f ? 1.0f : fmaxf(0.25f, ratio);
    float hz = ratio >= 1.0f ? fmaxf(0.25f, 1.0f / ratio) : 1.0f;
    float height = VIEW3D_HEIGHT * 0.6f;
    faces_reserve((R - 1) * (C - 1) + 8);

    add_floor(cam, -hx, hx, -hz, hz);
    for (int r = 0; r < R - 1; ++r) {
        for (int c = 0; c < C - 1; ++c) {
            V3 corner[4];
            SDL_Color col[4];
            static const int dr[4] = {0, 0, 1, 1}, dc[4] = {0, 1, 1, 0};
            for (int k = 0; k < 4; ++k) {
                int rr = r + dr[k], cc = c + dc[k];
                float t = grid[rr * C + cc] / hmax;
                corner[k] = (V3){-hx + 2.0f * hx * cc / (C - 1), t * height, hz - 2.0f * hz * rr / (R - 1)};
                col[k] = color_signed(t * 2.0f); /* most weights are small next to the largest: stretch the colours */
            }
            add_surface_quad(cam, corner[0], corner[1], corner[2], corner[3], col, 0);
        }
    }
}

static void draw_3d_view(Gui *g, SDL_Rect box, const LayerViz *lv, const float *data, const char *caption) {
    draw_panel(g->weights_ren, box);

    Cam cam = {cosf(g->yaw), sinf(g->yaw), cosf(g->pitch), sinf(g->pitch), 600.0f * g->zoom3d, box.x + box.w * 0.5f,
               box.y + box.h * 0.5f + 20.0f};
    g_faces_n = 0;
    int n_data;
    if (lv->kind == VIZ_CONV_GRID) {
        build_conv_3d(&cam, lv, data);
        n_data = lv->count * lv->ksize * lv->ksize;
    } else {
        build_matrix_3d(&cam, lv, data);
        n_data = lv->count * lv->cols;
    }
    flush_faces(g, box);

    SDL_Color dim = {COL_TEXT_DIM};
    wtxt(g, caption, box.x + 10, box.y + 8, (SDL_Color){COL_TEXT});
    wtxt(g, "drag = rotate   scroll = zoom   0 = reset", box.x + 10, box.y + 28, dim);

    float vmin = data[0], vmax = data[0];
    for (int i = 1; i < n_data; ++i) {
        if (data[i] < vmin) vmin = data[i];
        if (data[i] > vmax) vmax = data[i];
    }
    float vmean, vrms;
    compute_mean_rms(data, n_data, &vmean, &vrms);
    char stats[96];
    snprintf(stats, sizeof(stats), "min % .5f  max % .5f  mean % .5f  rms %.5f", vmin, vmax, vmean, vrms);
    wtxt(g, stats, box.x + 10, box.y + box.h - 22, dim);
    const char *legend = (lv->kind == VIZ_CONV_GRID && !lv->full_detail)
                             ? "height = mean |w| (blue low, yellow high)"
                             : "height = weight (blue < 0, red > 0)";
    wtxt(g, legend, box.x + 10, box.y + box.h - 42, dim);
}

static void render_weights_window(Gui *g) {
    if (!g->weights_win) return;
    g->wslot = 0;

    SDL_SetRenderDrawColor(g->weights_ren, COL_BG);
    SDL_RenderClear(g->weights_ren);

    if (!g->has_model || !g->has_record || !g->weights_valid) {
        wtxt(g, "waiting for weights from the trainer...", 30, 40, (SDL_Color){COL_TEXT_DIM});
        SDL_RenderPresent(g->weights_ren);
        return;
    }

    int n = g->model.n_viz;
    for (int i = 0; i < n; ++i) {
        SDL_Rect tab = weights_tab_rect(i, n);
        int active = (g->weights_tab == i);
        rrect_bordered(g->weights_ren, tab, 8, active ? (SDL_Color){COL_LOSS} : (SDL_Color){COL_SURFACE},
                       active ? (SDL_Color){COL_LOSS} : (SDL_Color){COL_BORDER});
        SDL_Color label_col = active ? (SDL_Color){10, 14, 28, 255} : (SDL_Color){COL_TEXT_DIM};
        int tw = 0;
        TTF_SizeText(g->font, g->model.layers[i].name, &tw, NULL);
        wtxt(g, g->model.layers[i].name, tab.x + (tab.w - tw) / 2, tab.y + 13, label_col);
    }

    const LayerViz *lv = &g->model.layers[g->weights_tab];
    const float *data = g->weights + lv->offset;
    SDL_Rect grid_box = {20, WEIGHTS_TAB_Y + WEIGHTS_TAB_H + 16, WEIGHTS_CONTENT_W, WEIGHTS_CONTENT_H};

    char caption[96];
    if (lv->kind == VIZ_CONV_GRID) {
        if (lv->full_detail)
            snprintf(caption, sizeof(caption), "%s - %d filters (%dx%d, full detail)", lv->name, lv->count, lv->ksize,
                     lv->ksize);
        else
            snprintf(caption, sizeof(caption), "%s - %d maps (%dx%d, mean |w| over in-ch)", lv->name, lv->count,
                     lv->ksize, lv->ksize);
    } else if (lv->pool == 1) {
        snprintf(caption, sizeof(caption), "%s weight (%d->%d, full detail)", lv->name, lv->in_dim, lv->out_dim);
    } else {
        snprintf(caption, sizeof(caption), "%s weight (%d->%d, pooled %d rows/px)", lv->name, lv->in_dim, lv->out_dim,
                 lv->pool);
    }

    draw_3d_view(g, grid_box, lv, data, caption);

    SDL_RenderPresent(g->weights_ren);
}

/* -- training dashboard -------------------------------------------------- */

static void draw_chart(Gui *g, SDL_Rect r) {
    draw_panel(g->ren, r);
    /* classification losses live in 0-2.5; a VAE's per-pixel loss is far smaller, so its axis follows the data */
    float ymax = LOSS_CLIP;
    int gen = g->run_cfg.task == TASK_GENERATE;
    if (gen) {
        ymax = 0.0f;
        for (int i = 0; i < g->hist_count; ++i) ymax = fmaxf(ymax, g->loss_hist[i]);
        ymax = ymax > 1e-4f ? ymax * 1.1f : 1.0f;
    }
    char title[48];
    if (gen) snprintf(title, sizeof(title), "loss per pixel (0-%.2f)", ymax);
    else snprintf(title, sizeof(title), "loss (0-2.5)");
    txt(g, g->font, title, r.x + 10, r.y + 8, (SDL_Color){COL_TEXT_DIM});

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
        float v0 = fminf(g->loss_hist[(start + i - 1) % HISTORY_LEN], ymax) / ymax;
        float v1 = fminf(g->loss_hist[(start + i) % HISTORY_LEN], ymax) / ymax;
        int x0 = px0 + plot_w * (i - 1) / (HISTORY_LEN - 1);
        int x1 = px0 + plot_w * i / (HISTORY_LEN - 1);
        int y0 = py1 - (int)(plot_h * v0);
        int y1 = py1 - (int)(plot_h * v1);
        SDL_RenderDrawLine(g->ren, x0, y0, x1, y1);
    }
}

/* Segmentation: the sample, then the true mask and the predicted mask drawn over a dimmed copy of it. */
static void draw_segmentation_sample(Gui *g, SDL_Rect box) {
    draw_panel(g->ren, box);
    if (!g->digit_tex || !g->sample || !g->mask_tex[0] || !g->mask_tex[1]) return;
    int w = g->digit_w, h = g->digit_h, c = g->digit_c;
    size_t plane = (size_t)w * h;
    const unsigned char *gt = g->sample + plane * c, *pred = gt + plane;

    void *pixels;
    int pitch;
    if (SDL_LockTexture(g->digit_tex, NULL, &pixels, &pitch) == 0) {
        for (int y = 0; y < h; ++y) {
            Uint32 *row = (Uint32 *)((Uint8 *)pixels + y * pitch);
            for (int x = 0; x < w; ++x) {
                size_t at = (size_t)y * w + x;
                Uint8 r = g->sample[at], gr = c >= 3 ? g->sample[plane + at] : r, b = c >= 3 ? g->sample[2 * plane + at] : r;
                row[x] = (0xFFu << 24) | (b << 16) | (gr << 8) | r;
            }
        }
        SDL_UnlockTexture(g->digit_tex);
    }
    static const Uint8 tint[2][3] = {{245, 101, 112}, {96, 140, 255}}; /* true: red, predicted: blue */
    const unsigned char *masks[2] = {gt, pred};
    for (int k = 0; k < 2; ++k) {
        if (SDL_LockTexture(g->mask_tex[k], NULL, &pixels, &pitch) != 0) continue;
        for (int y = 0; y < h; ++y) {
            Uint32 *row = (Uint32 *)((Uint8 *)pixels + y * pitch);
            for (int x = 0; x < w; ++x) {
                size_t at = (size_t)y * w + x;
                Uint8 r = g->sample[at], gr = c >= 3 ? g->sample[plane + at] : r, b = c >= 3 ? g->sample[2 * plane + at] : r;
                if (masks[k][at]) { r = tint[k][0]; gr = tint[k][1]; b = tint[k][2]; }
                else { r = (Uint8)(r * 0.55f); gr = (Uint8)(gr * 0.55f); b = (Uint8)(b * 0.55f); }
                row[x] = (0xFFu << 24) | (b << 16) | (gr << 8) | r;
            }
        }
        SDL_UnlockTexture(g->mask_tex[k]);
    }

    int gap = 8, tile = (box.w - 4 * gap) / 3;
    SDL_Texture *tex[3] = {g->digit_tex, g->mask_tex[0], g->mask_tex[1]};
    const char *cap[3] = {"scan", "true tumour", "predicted"};
    for (int i = 0; i < 3; ++i) {
        SDL_Rect dst = {box.x + gap + i * (tile + gap), box.y + 14, tile, tile};
        SDL_RenderCopy(g->ren, tex[i], NULL, &dst);
        txt(g, g->font_small, cap[i], dst.x + 2, dst.y + tile + 3, (SDL_Color){COL_TEXT_DIM});
    }
    long tp = 0, fp = 0, fn = 0, gt_px = 0, pr_px = 0;
    for (size_t i = 0; i < plane; ++i) {
        int a = gt[i] != 0, b = pred[i] != 0;
        tp += a && b; fp += !a && b; fn += a && !b; gt_px += a; pr_px += b;
    }
    char buf[80];
    snprintf(buf, sizeof(buf), "tumour pixels: true %ld, predicted %ld", gt_px, pr_px);
    txt(g, g->font_small, buf, box.x + 10, box.y + box.h - 34, (SDL_Color){COL_TEXT_DIM});
    if (gt_px + pr_px > 0) snprintf(buf, sizeof(buf), "this sample: dice %.2f", 2.0 * tp / (2.0 * tp + fp + fn));
    else snprintf(buf, sizeof(buf), "this sample: no tumour");
    txt(g, g->font_small, buf, box.x + 10, box.y + box.h - 18, (SDL_Color){COL_TEXT});
}

/* VAE: the sample the batch showed, next to the model's reconstruction of it. */
static void draw_reconstruction_sample(Gui *g, SDL_Rect box) {
    draw_panel(g->ren, box);
    if (!g->digit_tex || !g->sample || !g->mask_tex[0]) return;
    int w = g->digit_w, h = g->digit_h, c = g->digit_c;
    size_t px = (size_t)w * h;
    SDL_Texture *tex[2] = {g->digit_tex, g->mask_tex[0]};
    for (int k = 0; k < 2; ++k) {
        const unsigned char *src = g->sample + (size_t)k * px * c;
        void *pixels;
        int pitch;
        if (SDL_LockTexture(tex[k], NULL, &pixels, &pitch) != 0) continue;
        for (int y = 0; y < h; ++y) {
            Uint32 *row = (Uint32 *)((Uint8 *)pixels + y * pitch);
            for (int x = 0; x < w; ++x) {
                size_t at = (size_t)y * w + x;
                Uint8 r = src[at], gr = c >= 3 ? src[px + at] : r, b = c >= 3 ? src[2 * px + at] : r;
                row[x] = (0xFFu << 24) | (b << 16) | (gr << 8) | r;
            }
        }
        SDL_UnlockTexture(tex[k]);
    }
    int gap = 14, tile = (box.w - 3 * gap) / 2;
    if (tile > box.h - 56) tile = box.h - 56;
    static const char *cap[2] = {"input", "reconstruction"};
    for (int k = 0; k < 2; ++k) {
        SDL_Rect dst = {box.x + gap + k * (tile + gap), box.y + 14, tile, tile};
        SDL_RenderCopy(g->ren, tex[k], NULL, &dst);
        txt(g, g->font_small, cap[k], dst.x, dst.y + tile + 4, (SDL_Color){COL_TEXT_DIM});
    }
    txt(g, g->font_small, "the latest batch's first image", box.x + 10, box.y + box.h - 20, (SDL_Color){COL_TEXT_DIM});
}

static void draw_digit(Gui *g, SDL_Rect box) {
    if (g->run_cfg.task == TASK_SEGMENT) {
        draw_segmentation_sample(g, box);
        return;
    }
    if (g->run_cfg.task == TASK_GENERATE) {
        draw_reconstruction_sample(g, box);
        return;
    }
    const Progress *p = &g->prog;
    int correct = p->true_label == p->pred_label;
    draw_panel(g->ren, box);
    if (!g->digit_tex || !g->sample) return;

    void *pixels;
    int pitch;
    if (SDL_LockTexture(g->digit_tex, NULL, &pixels, &pitch) == 0) {
        for (int y = 0; y < g->digit_h; ++y) {
            Uint32 *row = (Uint32 *)((Uint8 *)pixels + y * pitch);
            for (int x = 0; x < g->digit_w; ++x) {
                size_t plane = (size_t)g->digit_w * g->digit_h, at = (size_t)y * g->digit_w + x;
                Uint8 r = g->sample[at];
                Uint8 gr = g->digit_c >= 3 ? g->sample[plane + at] : r;
                Uint8 b = g->digit_c >= 3 ? g->sample[2 * plane + at] : r;
                row[x] = (0xFFu << 24) | (b << 16) | (gr << 8) | r;
            }
        }
        SDL_UnlockTexture(g->digit_tex);
    }

    int size = box.h - 70;
    float scale = (float)size / (float)(g->digit_w > g->digit_h ? g->digit_w : g->digit_h);
    int dw = (int)(g->digit_w * scale), dh = (int)(g->digit_h * scale);
    SDL_Rect dst = {box.x + (box.w - dw) / 2, box.y + 16 + (size - dh) / 2, dw, dh};
    /* COL_GOOD/COL_BAD expand to four comma-separated values, so they can't sit inside ?: directly. */
    SDL_Color verdict = correct ? (SDL_Color){COL_GOOD} : (SDL_Color){COL_BAD};
    SDL_SetRenderDrawColor(g->ren, verdict.r, verdict.g, verdict.b, verdict.a);
    SDL_Rect ring = {dst.x - 4, dst.y - 4, dst.w + 8, dst.h + 8};
    SDL_RenderFillRect(g->ren, &ring);
    SDL_RenderCopy(g->ren, g->digit_tex, NULL, &dst);

    char name_a[16], name_b[16], buf[64];
    snprintf(buf, sizeof(buf), "label: %s", class_name(&g->run_ds, p->true_label, name_a, sizeof(name_a)));
    txt(g, g->font, buf, box.x + 14, box.y + 16 + size + 8, (SDL_Color){COL_TEXT_DIM});
    snprintf(buf, sizeof(buf), "pred:  %s  %s", class_name(&g->run_ds, p->pred_label, name_b, sizeof(name_b)),
             correct ? "ok" : "WRONG");
    txt(g, g->font, buf, box.x + 14, box.y + 16 + size + 28, verdict);
}

/* -- predictions page ---------------------------------------------------- */

static SDL_Texture *pred_texture(Gui *g, int slot) {
    int w = g->run_cfg.img_w, h = g->run_cfg.img_h;
    if (g->pred_tex_w != w || g->pred_tex_h != h) {
        for (int i = 0; i < 96; ++i) {
            if (g->pred_tex[i]) SDL_DestroyTexture(g->pred_tex[i]);
            g->pred_tex[i] = NULL;
        }
        g->pred_tex_w = w;
        g->pred_tex_h = h;
    }
    if (!g->pred_tex[slot]) {
        g->pred_tex[slot] = SDL_CreateTexture(g->ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, w, h);
        if (g->pred_tex[slot]) SDL_SetTextureScaleMode(g->pred_tex[slot], SDL_ScaleModeNearest);
    }
    return g->pred_tex[slot];
}

/* Fills a texture from a sample's planes: mask == NULL draws the scan itself, otherwise the scan dimmed with
 * the mask painted over it in `tint`. */
static void fill_pred_texture(SDL_Texture *tex, const unsigned char *pix, int c, int w, int h, const unsigned char *mask,
                              const Uint8 tint[3]) {
    void *pixels;
    int pitch;
    if (!tex || SDL_LockTexture(tex, NULL, &pixels, &pitch) != 0) return;
    size_t plane = (size_t)w * h;
    for (int y = 0; y < h; ++y) {
        Uint32 *row = (Uint32 *)((Uint8 *)pixels + y * pitch);
        for (int x = 0; x < w; ++x) {
            size_t at = (size_t)y * w + x;
            Uint8 r = pix[at], gr = c >= 3 ? pix[plane + at] : r, b = c >= 3 ? pix[2 * plane + at] : r;
            if (mask) {
                if (mask[at]) { r = tint[0]; gr = tint[1]; b = tint[2]; }
                else { r = (Uint8)(r * 0.55f); gr = (Uint8)(gr * 0.55f); b = (Uint8)(b * 0.55f); }
            }
            row[x] = (0xFFu << 24) | (b << 16) | (gr << 8) | r;
        }
    }
    SDL_UnlockTexture(tex);
}

static void draw_predictions(Gui *g) {
    SDL_Point mouse = {g->mouse_x, g->mouse_y};
    const PredHeader *hd = &g->pred_hdr;
    const TrainConfig *cfg = &g->run_cfg;
    int seg = cfg->task == TASK_SEGMENT;
    int cols = seg ? 3 : 6, rows = seg ? 3 : 4, per_page = cols * rows;
    int pages = hd->n > 0 ? (hd->n + per_page - 1) / per_page : 1;
    if (g->pred_page >= pages) g->pred_page = pages - 1;
    if (g->pred_page < 0) g->pred_page = 0;

    draw_button(g, BTN_PRED_BACK, "< Back", g->font, SDL_PointInRect(&mouse, &BTN_PRED_BACK), 1);
    txt(g, g->font_big, "Predictions", 146, 24, (SDL_Color){COL_TEXT});
    char buf[160];
    if (hd->n > 0) {
        snprintf(buf, sizeof(buf), "%s  -  model after epoch %d  -  %s  -  page %d / %d", seg ? "segmentation" : "classification",
                 hd->epoch + 1, hd->from_test ? "test split" : "training split (no test split)", g->pred_page + 1, pages);
        txt(g, g->font_small, buf, 300, 33, (SDL_Color){COL_TEXT_DIM});
    }
    draw_button(g, BTN_PRED_PREV, "prev", g->font, SDL_PointInRect(&mouse, &BTN_PRED_PREV), g->pred_page > 0);
    draw_button(g, BTN_PRED_NEXT, "next", g->font, SDL_PointInRect(&mouse, &BTN_PRED_NEXT), g->pred_page + 1 < pages);

    SDL_Rect area = {20, 68, 1110, 612};
    if (hd->n < 1 || !g->pred_data) {
        draw_panel(g->ren, area);
        txt(g, g->font_big, g->has_model ? "predictions appear after the first epoch finishes" : "no model yet", area.x + 250,
            area.y + 270, (SDL_Color){COL_TEXT_DIM});
        return;
    }

    int c = cfg->img_c, w = cfg->img_w, h = cfg->img_h;
    size_t sample_bytes = train_sample_bytes(cfg), record = sizeof(PredSample) + sample_bytes, plane = (size_t)w * h;
    int gap = seg ? 12 : 10;
    int cell_w = (area.w - (cols - 1) * gap) / cols, cell_h = (area.h - (rows - 1) * gap) / rows;
    static const Uint8 tint_true[3] = {245, 101, 112}, tint_pred[3] = {96, 140, 255};

    for (int k = 0; k < per_page; ++k) {
        int idx = g->pred_page * per_page + k;
        if (idx >= hd->n) break;
        const unsigned char *rec = g->pred_data + sizeof(PredHeader) + (size_t)idx * record;
        PredSample ps;
        memcpy(&ps, rec, sizeof(ps));
        const unsigned char *pix = rec + sizeof(PredSample);
        SDL_Rect cell = {area.x + (k % cols) * (cell_w + gap), area.y + (k / cols) * (cell_h + gap), cell_w, cell_h};

        if (seg) {
            SDL_Color edge = ps.f < 0 ? (SDL_Color){COL_BORDER_HI} : ps.f >= 0.75f ? (SDL_Color){COL_GOOD}
                             : ps.f >= 0.4f ? (SDL_Color){242, 190, 80, 255} : (SDL_Color){COL_BAD};
            rrect_bordered(g->ren, cell, 10, (SDL_Color){COL_SURFACE}, edge);
            const unsigned char *gt = pix + plane * c, *pred = gt + plane;
            SDL_Texture *t0 = pred_texture(g, k * 3), *t1 = pred_texture(g, k * 3 + 1), *t2 = pred_texture(g, k * 3 + 2);
            fill_pred_texture(t0, pix, c, w, h, NULL, tint_true);
            fill_pred_texture(t1, pix, c, w, h, gt, tint_true);
            fill_pred_texture(t2, pix, c, w, h, pred, tint_pred);
            int tgap = 8, tile = (cell.w - 4 * tgap) / 3;
            if (tile > cell.h - 54) tile = cell.h - 54;
            SDL_Texture *tex[3] = {t0, t1, t2};
            static const char *cap[3] = {"scan", "true", "predicted"};
            for (int i = 0; i < 3; ++i) {
                SDL_Rect dst = {cell.x + tgap + i * (tile + tgap), cell.y + 12, tile, tile};
                if (tex[i]) SDL_RenderCopy(g->ren, tex[i], NULL, &dst);
                txt(g, g->font_small, cap[i], dst.x + 2, dst.y + tile + 2, (SDL_Color){COL_TEXT_DIM});
            }
            if (ps.f >= 0) snprintf(buf, sizeof(buf), "dice %.2f   tumour px: true %d, predicted %d", ps.f, ps.a, ps.b);
            else snprintf(buf, sizeof(buf), "no tumour, none predicted");
            txt(g, g->font_small, buf, cell.x + 10, cell.y + cell.h - 20,
                ps.f < 0 ? (SDL_Color){COL_GOOD} : (SDL_Color){COL_TEXT});
        } else {
            int ok = ps.a == ps.b;
            rrect_bordered(g->ren, cell, 9, (SDL_Color){COL_SURFACE}, ok ? (SDL_Color){COL_GOOD} : (SDL_Color){COL_BAD});
            SDL_Texture *t = pred_texture(g, k);
            fill_pred_texture(t, pix, c, w, h, NULL, tint_true);
            int tile = cell.w - 40 < cell.h - 46 ? cell.w - 40 : cell.h - 46;
            SDL_Rect dst = {cell.x + (cell.w - tile) / 2, cell.y + 8, tile, tile};
            if (t) SDL_RenderCopy(g->ren, t, NULL, &dst);
            char na[CLASS_NAME_LEN + 8], nb[CLASS_NAME_LEN + 8];
            snprintf(buf, sizeof(buf), "pred %s %.0f%%", class_name(&g->run_ds, ps.b, na, sizeof(na)), ps.f * 100.0f);
            txt(g, g->font_small, buf, cell.x + 8, cell.y + cell.h - 34, ok ? (SDL_Color){COL_GOOD} : (SDL_Color){COL_BAD});
            snprintf(buf, sizeof(buf), "true %s", class_name(&g->run_ds, ps.a, nb, sizeof(nb)));
            txt(g, g->font_small, buf, cell.x + 8, cell.y + cell.h - 18, (SDL_Color){COL_TEXT_DIM});
        }
    }
}

/* The VAE's page: reconstructions, samples from the prior, interpolations (see GenHeader). */
static void draw_generations(Gui *g) {
    SDL_Point mouse = {g->mouse_x, g->mouse_y};
    const GenHeader *hd = &g->gen_hdr;
    draw_button(g, BTN_PRED_BACK, "< Back", g->font, SDL_PointInRect(&mouse, &BTN_PRED_BACK), 1);
    txt(g, g->font_big, "Generations", 146, 24, (SDL_Color){COL_TEXT});
    char buf[160];
    if (hd->n_recon > 0) {
        snprintf(buf, sizeof(buf), "VAE  -  latent size %d  -  model after epoch %d", hd->z_dim, hd->epoch + 1);
        txt(g, g->font_small, buf, 320, 33, (SDL_Color){COL_TEXT_DIM});
    }
    SDL_Rect area = {20, 68, 1110, 612};
    if (hd->n_recon < 1 || !g->gen_data) {
        draw_panel(g->ren, area);
        txt(g, g->font_big, "generations appear after the first epoch finishes", area.x + 250, area.y + 270,
            (SDL_Color){COL_TEXT_DIM});
        return;
    }
    int c = g->run_cfg.img_c, w = g->run_cfg.img_w, h = g->run_cfg.img_h;
    size_t img = (size_t)w * h * c;
    const unsigned char *data = g->gen_data + sizeof(GenHeader);
    static const Uint8 none[3] = {0, 0, 0};

    int cols = 14, gap = 6, tile = (area.w - (cols - 1) * gap) / cols;
    int y = area.y;
    int slot = 0;
    SDL_Color dim = {COL_TEXT_DIM};

    /* reconstructions: the original above its reconstruction */
    txt(g, g->font_small, "reconstructions   (top: test image, bottom: what the VAE rebuilds from its latent)", area.x, y, dim);
    y += 18;
    for (int j = 0; j < hd->n_recon && j < cols; ++j)
        for (int k = 0; k < 2; ++k) {
            SDL_Texture *t = pred_texture(g, slot++);
            fill_pred_texture(t, data + ((size_t)2 * j + k) * img, c, w, h, NULL, none);
            SDL_Rect dst = {area.x + j * (tile + gap), y + k * (tile + gap), tile, tile};
            if (t) SDL_RenderCopy(g->ren, t, NULL, &dst);
        }
    y += 2 * (tile + gap) + 8;
    data += (size_t)2 * hd->n_recon * img;

    /* samples from the prior, the same latents every epoch */
    txt(g, g->font_small, "new images   (decoded from random latents z ~ N(0, 1); the same z every epoch)", area.x, y, dim);
    y += 18;
    for (int j = 0; j < hd->n_samples; ++j) {
        SDL_Texture *t = pred_texture(g, slot++);
        fill_pred_texture(t, data + (size_t)j * img, c, w, h, NULL, none);
        SDL_Rect dst = {area.x + (j % cols) * (tile + gap), y + (j / cols) * (tile + gap), tile, tile};
        if (t) SDL_RenderCopy(g->ren, t, NULL, &dst);
    }
    y += ((hd->n_samples + cols - 1) / cols) * (tile + gap) + 8;
    data += (size_t)hd->n_samples * img;

    /* interpolation */
    txt(g, g->font_small, "interpolation   (each row walks the latent from one test image to another)", area.x, y, dim);
    y += 18;
    for (int r = 0; r < hd->n_rows; ++r)
        for (int k = 0; k < hd->n_steps && k < cols; ++k) {
            SDL_Texture *t = pred_texture(g, slot++);
            fill_pred_texture(t, data + ((size_t)r * hd->n_steps + k) * img, c, w, h, NULL, none);
            SDL_Rect dst = {area.x + k * (tile + gap), y + r * (tile + gap), tile, tile};
            if (t && slot <= 96) SDL_RenderCopy(g->ren, t, NULL, &dst);
        }
}

static void draw_dashboard(Gui *g) {
    SDL_Point mouse = {g->mouse_x, g->mouse_y};
    SDL_Rect arch_box = {20, 20, 1110, 164};
    draw_architecture(g, arch_box, &g->run_cfg, &g->run_shapes);

    SDL_Rect chart_box = {20, 194, 790, 486};
    if (g->has_record) {
        draw_chart(g, chart_box);
    } else {
        draw_panel(g->ren, chart_box);
        txt(g, g->font, "loss (0-2.5)", chart_box.x + 10, chart_box.y + 8, (SDL_Color){COL_TEXT_DIM});
        txt(g, g->font_big, g->worker_gone ? "train_worker is not running" : "waiting for training to start...",
            chart_box.x + 220, chart_box.y + 250, (SDL_Color){COL_TEXT_DIM});
    }

    SDL_Rect digit_box = {830, 194, 300, 166};
    if (g->has_record) draw_digit(g, digit_box);
    else draw_panel(g->ren, digit_box);

    SDL_Rect wb = WEIGHTS_BUTTON;
    int wb_hover = SDL_PointInRect(&mouse, &wb);
    rrect_bordered(g->ren, wb, 12, wb_hover ? col_mix((SDL_Color){COL_SURFACE}, (SDL_Color){COL_LOSS}, 0.16f) : (SDL_Color){COL_SURFACE},
                   wb_hover ? (SDL_Color){COL_LOSS} : (SDL_Color){COL_BORDER_HI});
    txt(g, g->font_big, "View layer weights", wb.x + 20, wb.y + 10, (SDL_Color){COL_LOSS});
    txt(g, g->font_small, "a live 3D weights window", wb.x + 20, wb.y + 42, (SDL_Color){COL_TEXT_DIM});

    SDL_Rect pb = PRED_BUTTON;
    int pb_hover = SDL_PointInRect(&mouse, &pb);
    rrect_bordered(g->ren, pb, 12, pb_hover ? col_mix((SDL_Color){COL_SURFACE}, (SDL_Color){COL_GOOD}, 0.14f) : (SDL_Color){COL_SURFACE},
                   pb_hover ? (SDL_Color){COL_GOOD} : (SDL_Color){COL_BORDER_HI});
    int gen_run = g->run_cfg.task == TASK_GENERATE;
    int have = gen_run ? g->gen_hdr.n_recon > 0 : g->pred_hdr.n > 0;
    txt(g, g->font_big, gen_run ? "View generations" : "View predictions", pb.x + 20, pb.y + 10, (SDL_Color){COL_GOOD});
    txt(g, g->font_small,
        !have ? "available after the first epoch"
              : (gen_run ? "reconstructions, new images, interpolation" : "browse the model's answers on fixed samples"),
        pb.x + 20, pb.y + 42, (SDL_Color){COL_TEXT_DIM});

    SDL_Rect info_box = {830, 530, 300, 150};
    draw_panel(g->ren, info_box);
    char line[96];
    if (g->has_record) {
        const Progress *p = &g->prog;
        snprintf(line, sizeof(line), "epoch   %d / %d", p->epoch, p->epochs);
        txt(g, g->font_big, line, info_box.x + 14, info_box.y + 12, (SDL_Color){COL_TEXT});
        snprintf(line, sizeof(line), "batch %d / %d   %.1fs", p->batch, p->batch_count, p->elapsed);
        txt(g, g->font, line, info_box.x + 14, info_box.y + 44, (SDL_Color){COL_TEXT_DIM});
        int seg = g->run_cfg.task == TASK_SEGMENT, gen = g->run_cfg.task == TASK_GENERATE;
        if (gen) snprintf(line, sizeof(line), "loss/px %.4f  KL %.1f", p->loss, p->test_iou);
        else if (seg) snprintf(line, sizeof(line), "loss %.4f   dice %.3f", p->loss, p->acc);
        else snprintf(line, sizeof(line), "loss %.4f   acc %.1f%%", p->loss, p->acc * 100.0f);
        txt(g, g->font, line, info_box.x + 14, info_box.y + 66, (SDL_Color){COL_LOSS});
        if (gen) {
            if (p->test_acc >= 0.0f) snprintf(line, sizeof(line), "test recon/px %.4f", p->test_acc);
            else snprintf(line, sizeof(line), "test recon --");
        } else if (seg) {
            if (p->test_acc >= 0.0f) snprintf(line, sizeof(line), "test dice %.3f   IoU %.3f", p->test_acc, p->test_iou);
            else snprintf(line, sizeof(line), "test dice --");
        } else if (p->test_acc >= 0.0f) snprintf(line, sizeof(line), "test acc %.1f%%", p->test_acc * 100.0f);
        else snprintf(line, sizeof(line), "test acc --");
        txt(g, g->font, line, info_box.x + 14, info_box.y + 88, (SDL_Color){COL_GOOD});
    } else {
        txt(g, g->font_big, "not started", info_box.x + 14, info_box.y + 12, (SDL_Color){COL_TEXT_DIM});
    }
    draw_button(g, RESTART_BUTTON, "restart", g->font, SDL_PointInRect(&mouse, &RESTART_BUTTON), !g->worker_gone);
    draw_button(g, SETTINGS_BUTTON, "settings", g->font, SDL_PointInRect(&mouse, &SETTINGS_BUTTON), 1);
}

/* -- input --------------------------------------------------------------- */

static int send_cmd(Gui *g, const TrainConfig *cfg) {
    TrainCmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.magic = TRAIN_MAGIC;
    cmd.type = TRAIN_CMD_START;
    cmd.cfg = *cfg;
    ssize_t w = write(g->cmd_fd, &cmd, sizeof(cmd));
    if (w != (ssize_t)sizeof(cmd)) {
        g->worker_gone = 1;
        return -1;
    }
    return 0;
}

/* Clears the chart and drops the last record so the UI reads as "waiting for
 * the new run" rather than silently continuing to show the old one's data
 * until the worker's first fresh record arrives. */
static void begin_run(Gui *g, const TrainConfig *cfg) {
    g->run_active = 1;
    g->worker_msg[0] = '\0';
    g->has_record = 0;
    g->weights_valid = 0;
    gui_clear_history(g);
    g->run_cfg = *cfg;
    train_cfg_derive(&g->run_cfg, &g->run_shapes, NULL, 0);
    refresh_run_ds(g);
    ensure_sample(g);
}

static void start_from_form(Gui *g) {
    if (!g->form_valid || g->worker_gone) return;
    if (send_cmd(g, &g->form_cfg) != 0) return;
    begin_run(g, &g->form_cfg);
    g->screen = SCREEN_RUN;
    g->focus = -1;
    g->dropdown_open = 0;
    SDL_StopTextInput();
}

static int field_accepts(int kind, char ch) {
    unsigned char c = (unsigned char)ch;
    switch (kind) {
    case FK_INT: return isdigit(c);
    case FK_FLOAT: return isdigit(c) || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E';
    case FK_CHOICE: return 0;
    default: return isalnum(c) || ch == '-' || ch == '_';
    }
}

static void field_insert(Gui *g, const char *utf8) {
    if (g->focus < 0) return;
    Field *f = &g->fields[g->focus];
    size_t len = strlen(f->text);
    for (const char *c = utf8; *c && len < FIELD_MAX_CHARS; ++c) {
        if (field_accepts(f->kind, *c)) {
            f->text[len++] = *c;
            f->text[len] = '\0';
        }
    }
}

static void focus_step(Gui *g, int dir) {
    int idx = g->focus;
    g->dropdown_open = 0;
    for (int tries = 0; tries < NUM_FIELDS; ++tries) {
        idx = (idx + dir + NUM_FIELDS) % NUM_FIELDS;
        if (idx < 0) idx = 0;
        if (g->fields[idx].visible) { g->focus = idx; return; }
    }
}

/* Zooms by `factor` keeping the view-space point under (cx, cy) - canvas-local screen pixels - where it is. */
static void zoom_at(Gui *g, float cx, float cy, float factor) {
    float z0 = g->zoom, z1 = z0 * factor;
    if (z1 < ZOOM_MIN) z1 = ZOOM_MIN;
    if (z1 > ZOOM_MAX) z1 = ZOOM_MAX;
    if (z1 == z0) return;
    g->pan_x += cx / z1 - cx / z0;
    g->pan_y += cy / z1 - cy / z0;
    g->zoom = z1;
}

static void load_preset(Gui *g, int i) {
    if (i == 0) graph_reset(g);
    else if (i == 1) graph_preset_resnet(g);
    else if (i == 2) graph_preset_unet(g);
    else graph_preset_vae(g);
}

static void setup_click(Gui *g, SDL_Point p) {
    form_layout(g);
    g->drag = DRAG_NONE;
    if (g->preset_open) {
        /* the open menu sits on top: a row loads that preset, any other click just closes it */
        g->preset_open = 0;
        for (int i = 0; i < preset_count(g); ++i) {
            SDL_Rect r = preset_rect(i);
            if (SDL_PointInRect(&p, &r)) {
                load_preset(g, preset_at(g, i));
                return;
            }
        }
        if (SDL_PointInRect(&p, &BTN_PRESETS)) return; /* clicking the button again just closes the menu */
    }
    if (g->dropdown_open) {
        /* The open list sits on top of everything: a click on a row picks it,
         * a click on the dataset field closes it, any other click just closes
         * it and is then handled normally. */
        g->dropdown_open = 0;
        int vis[MAX_DATASETS], nv = ds_visible(g, vis);
        for (int k = 0; k < nv; ++k) {
            SDL_Rect r = dropdown_row_rect(g, k);
            if (SDL_PointInRect(&p, &r)) {
                dataset_choose(g, g->ds[vis[k]].name);
                return;
            }
        }
        if (SDL_PointInRect(&p, &g->fields[F_DATASET].rect)) return;
    }
    for (int i = 0; i < 3; ++i) {
        if (SDL_PointInRect(&p, &PROBLEM_BTN[i])) {
            set_problem(g, i);
            g->focus = -1;
            return;
        }
    }
    for (int i = 0; i < NUM_GENERAL_FIELDS; ++i) {
        if (SDL_PointInRect(&p, &g->fields[i].rect)) {
            g->focus = i;
            if (i == F_DATASET) {
                scan_datasets(g);
                g->dropdown_open = 1;
            }
            return;
        }
    }

    if (SDL_PointInRect(&p, &GRAPH_VIEW)) {
        /* zoom controls sit on top of the canvas, in screen space */
        for (int k = 0; k < 3; ++k) {
            SDL_Rect zr = zoom_btn(k);
            if (!SDL_PointInRect(&p, &zr)) continue;
            if (k == 2) graph_fit(g);
            else zoom_at(g, (float)(ZOOM_BAR.x + ZOOM_BAR.w / 2 - GRAPH_VIEW.x), (float)(GRAPH_VIEW.h / 2), k == 0 ? 1.0f / 1.25f : 1.25f);
            g->drag = DRAG_NONE;
            return;
        }
        /* everything below happens in view space: node rects, ports and fields are all expressed in it */
        SDL_Point pv = to_view(g, p.x, p.y);
        g->drag_fx = g->drag_fy = 0.0f;
        int rad = (int)fmaxf(10.0f, 8.0f / g->zoom);
        for (int i = NUM_GENERAL_FIELDS; i < NUM_FIELDS; ++i) {
            if (g->fields[i].visible && SDL_PointInRect(&pv, &g->fields[i].rect)) {
                g->focus = i;
                return;
            }
        }
        g->focus = -1;
        for (int i = MAX_NODES - 1; i >= 0; --i) { /* topmost node first */
            if (!g->nodes[i].used) continue;
            int type = g->nodes[i].type;
            if (type != NODE_OUTPUT && near_port(node_out_port(g, i), pv, rad)) {
                g->drag = DRAG_LINK;
                g->drag_node = i;
                return;
            }
            for (int k = 0; k < node_nin(type); ++k) {
                if (!near_port(node_in_port(g, i, k), pv, rad)) continue;
                /* Grabbing a wired input dot pulls its wire off, to be dropped elsewhere. */
                int src = g->nodes[i].in[k];
                if (src >= 0) {
                    g->nodes[i].in[k] = -1;
                    g->drag = DRAG_LINK;
                    g->drag_node = src;
                }
                return;
            }
            SDL_Rect r = node_rect(g, i);
            if (SDL_PointInRect(&pv, &r)) {
                SDL_Rect d = node_del_rect(r);
                if (node_deletable(type) && SDL_PointInRect(&pv, &d)) {
                    graph_remove(g, i);
                } else {
                    g->drag = DRAG_NODE;
                    g->drag_node = i;
                }
                return;
            }
        }
        g->drag = DRAG_PAN;
        return;
    }

    g->focus = -1;
    for (int i = 0; i < palette_count(g); ++i) {
        SDL_Rect pr = palette_rect(i);
        if (SDL_PointInRect(&p, &pr)) {
            g->drag = DRAG_PAL;
            g->drag_type = palette_type(g, i);
            g->drag_moved = 0;
            g->press_x = p.x;
            g->press_y = p.y;
            return;
        }
    }
    if (SDL_PointInRect(&p, &BTN_ARRANGE)) graph_arrange(g);
    else if (SDL_PointInRect(&p, &BTN_PRESETS)) g->preset_open = 1;
    else if (SDL_PointInRect(&p, &BTN_START)) start_from_form(g);
    else if (g->run_active && (g->pred_hdr.n > 0 || g->gen_hdr.n_recon > 0) && SDL_PointInRect(&p, &BTN_VIEW_PRED)) {
        g->screen = SCREEN_PRED;
        g->pred_return = SCREEN_SETUP;
        SDL_StopTextInput();
    } else if (g->run_active && SDL_PointInRect(&p, &BTN_BACK)) {
        g->screen = SCREEN_RUN;
        SDL_StopTextInput();
    }
    form_layout(g);
}

/* Finishes a drag: a wire dropped on a node connects to it, a palette block
 * dropped on the canvas is placed there, or just clicked is appended. */
static void setup_release(Gui *g, SDL_Point p_screen) {
    form_layout(g);
    SDL_Point p = to_view(g, p_screen.x, p_screen.y); /* node hit tests below are in view space */
    if (g->drag == DRAG_LINK) {
        for (int i = MAX_NODES - 1; i >= 0; --i) {
            if (!g->nodes[i].used || g->nodes[i].type == NODE_INPUT || i == g->drag_node) continue;
            SDL_Rect r = node_rect(g, i);
            if (SDL_PointInRect(&p, &r)) {
                /* An add node has two ports: take the free one if only one is free, else the
                 * port nearest the drop point. */
                int nin = node_nin(g->nodes[i].type), port = 0;
                if (nin == 2) {
                    int free0 = g->nodes[i].in[0] < 0, free1 = g->nodes[i].in[1] < 0;
                    if (free0 != free1) port = free0 ? 0 : 1;
                    else port = abs(p.y - node_in_port(g, i, 0).y) <= abs(p.y - node_in_port(g, i, 1).y) ? 0 : 1;
                }
                graph_link(g, g->drag_node, i, port);
                break;
            }
        }
    } else if (g->drag == DRAG_PAL) {
        if (!g->drag_moved) {
            graph_append(g, g->drag_type);
        } else if (SDL_PointInRect(&p_screen, &GRAPH_VIEW)) {
            graph_add(g, g->drag_type, p.x - (int)floorf(g->pan_x) - NODE_W / 2, p.y - (int)floorf(g->pan_y) - 12);
        }
    }
    g->drag = DRAG_NONE;
    form_layout(g);
}

static int pump_events(Gui *g) {
    SDL_Event e;
    Uint32 main_id = SDL_GetWindowID(g->win);
    form_layout(g);
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_QUIT:
            return 0;
        case SDL_KEYDOWN: {
            SDL_Keycode k = e.key.keysym.sym;
            /* A clicked field owns the keyboard whichever window SDL thinks has focus - with
             * the weights window open and no window manager to move focus on click, key
             * events would otherwise keep arriving tagged with the weights window. */
            if (k == SDLK_F11) { /* fullscreen <-> window, from any screen */
                Uint32 flags = SDL_GetWindowFlags(g->win);
                SDL_SetWindowFullscreen(g->win, (flags & SDL_WINDOW_FULLSCREEN_DESKTOP) ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP);
                break;
            }
            int form_has_keys = e.key.windowID == main_id || g->focus >= 0;
            if (g->screen == SCREEN_PRED && e.key.windowID == main_id) {
                if (k == SDLK_LEFT && g->pred_page > 0) g->pred_page--;
                else if (k == SDLK_RIGHT) g->pred_page++;
                break;
            }
            if (form_has_keys && g->screen == SCREEN_SETUP) {
                if (k == SDLK_ESCAPE) {
                    if (g->dropdown_open) g->dropdown_open = 0;
                    else if (g->focus >= 0) g->focus = -1;
                    else return 0;
                } else if ((k == SDLK_UP || k == SDLK_DOWN) && g->focus == F_DATASET && ds_visible(g, (int[MAX_DATASETS]){0}) > 0) {
                    int vis[MAX_DATASETS], nv = ds_visible(g, vis), cur = -1;
                    for (int i = 0; i < nv; ++i)
                        if (strcmp(g->ds[vis[i]].name, g->fields[F_DATASET].text) == 0) cur = i;
                    cur = (cur + (k == SDLK_DOWN ? 1 : -1) + nv) % nv;
                    dataset_choose(g, g->ds[vis[cur]].name);
                } else if (k == SDLK_BACKSPACE) {
                    if (g->focus >= 0 && g->fields[g->focus].kind != FK_CHOICE) {
                        size_t len = strlen(g->fields[g->focus].text);
                        if (len > 0) g->fields[g->focus].text[len - 1] = '\0';
                    }
                } else if (k == SDLK_TAB) {
                    focus_step(g, (e.key.keysym.mod & KMOD_SHIFT) ? -1 : 1);
                } else if (k == SDLK_RETURN || k == SDLK_KP_ENTER) {
                    start_from_form(g);
                } else if (k == SDLK_v && (e.key.keysym.mod & KMOD_CTRL)) {
                    char *clip = SDL_GetClipboardText();
                    if (clip) { field_insert(g, clip); SDL_free(clip); }
                }
                break;
            }
            if (k == SDLK_ESCAPE) return 0;
            if (g->weights_win && e.key.windowID == SDL_GetWindowID(g->weights_win) && k == SDLK_0) {
                g->yaw = 0.6f;
                g->pitch = 0.75f;
                g->zoom3d = 1.0f;
            }
            break;
        }
        case SDL_TEXTINPUT:
            if (g->screen == SCREEN_SETUP && g->focus >= 0) field_insert(g, e.text.text);
            break;
        case SDL_MOUSEWHEEL:
            if (g->weights_win && e.wheel.windowID == SDL_GetWindowID(g->weights_win)) {
                float factor = e.wheel.y > 0 ? 1.15f : e.wheel.y < 0 ? (1.0f / 1.15f) : 1.0f;
                g->zoom3d *= factor;
                if (g->zoom3d < 0.4f) g->zoom3d = 0.4f;
                if (g->zoom3d > 5.0f) g->zoom3d = 5.0f;
            } else if (g->screen == SCREEN_SETUP && g->win && e.wheel.windowID == main_id) {
                /* scrolling over the node canvas zooms it around the cursor (shift + scroll pans sideways) */
                SDL_Point m = {g->mouse_x, g->mouse_y};
                if (SDL_PointInRect(&m, &GRAPH_VIEW)) {
                    if (SDL_GetModState() & KMOD_SHIFT) {
                        g->pan_x += (float)e.wheel.y * 48.0f / g->zoom;
                    } else if (e.wheel.y != 0) {
                        zoom_at(g, (float)(m.x - GRAPH_VIEW.x), (float)(m.y - GRAPH_VIEW.y), e.wheel.y > 0 ? 1.12f : 1.0f / 1.12f);
                    }
                    g->pan_x += (float)e.wheel.x * 48.0f / g->zoom;
                }
            }
            break;
        case SDL_MOUSEMOTION:
            if (e.motion.windowID == main_id) {
                g->mouse_x = e.motion.x;
                g->mouse_y = e.motion.y;
                if (g->drag != DRAG_NONE) {
                    if (!(e.motion.state & SDL_BUTTON_LMASK)) {
                        g->drag = DRAG_NONE; /* the release happened outside the window */
                    } else if (g->drag == DRAG_NODE) {
                        /* screen pixels -> view units; the fractional part is carried to the next event */
                        g->drag_fx += (float)e.motion.xrel / g->zoom;
                        g->drag_fy += (float)e.motion.yrel / g->zoom;
                        int dx = (int)g->drag_fx, dy = (int)g->drag_fy;
                        g->drag_fx -= (float)dx;
                        g->drag_fy -= (float)dy;
                        g->nodes[g->drag_node].x += dx;
                        g->nodes[g->drag_node].y += dy;
                    } else if (g->drag == DRAG_PAN) {
                        g->pan_x += (float)e.motion.xrel / g->zoom;
                        g->pan_y += (float)e.motion.yrel / g->zoom;
                    } else if (g->drag == DRAG_PAL && abs(e.motion.x - g->press_x) + abs(e.motion.y - g->press_y) > 6) {
                        g->drag_moved = 1;
                    }
                }
            } else if (g->weights_win && e.motion.windowID == SDL_GetWindowID(g->weights_win) &&
                       (e.motion.state & SDL_BUTTON_LMASK)) {
                g->yaw += e.motion.xrel * 0.01f;
                g->pitch += e.motion.yrel * 0.01f;
                if (g->pitch < 0.12f) g->pitch = 0.12f;
                if (g->pitch > 1.5f) g->pitch = 1.5f;
            }
            break;
        case SDL_MOUSEBUTTONDOWN:
            if (e.button.windowID == main_id) {
                SDL_Point p = {e.button.x, e.button.y};
                if (g->screen == SCREEN_SETUP) {
                    if (e.button.button == SDL_BUTTON_LEFT) setup_click(g, p);
                } else if (g->screen == SCREEN_PRED) {
                    if (SDL_PointInRect(&p, &BTN_PRED_BACK)) {
                        g->screen = g->pred_return == SCREEN_SETUP ? SCREEN_SETUP : SCREEN_RUN;
                        if (g->screen == SCREEN_SETUP) SDL_StartTextInput();
                    } else if (SDL_PointInRect(&p, &BTN_PRED_PREV)) {
                        if (g->pred_page > 0) g->pred_page--;
                    } else if (SDL_PointInRect(&p, &BTN_PRED_NEXT)) {
                        g->pred_page++; /* clamped when drawn */
                    }
                } else if (SDL_PointInRect(&p, &PRED_BUTTON)) {
                    g->screen = SCREEN_PRED;
                    g->pred_return = SCREEN_RUN;
                } else if (SDL_PointInRect(&p, &WEIGHTS_BUTTON)) {
                    open_weights_window(g);
                } else if (SDL_PointInRect(&p, &RESTART_BUTTON)) {
                    if (!g->worker_gone && send_cmd(g, &g->run_cfg) == 0) begin_run(g, &g->run_cfg);
                } else if (SDL_PointInRect(&p, &SETTINGS_BUTTON)) {
                    g->screen = SCREEN_SETUP;
                    SDL_StartTextInput();
                }
            } else if (g->weights_win && e.button.windowID == SDL_GetWindowID(g->weights_win) && g->has_model) {
                SDL_Point p = {e.button.x, e.button.y};
                for (int i = 0; i < g->model.n_viz; ++i) {
                    SDL_Rect tab = weights_tab_rect(i, g->model.n_viz);
                    if (SDL_PointInRect(&p, &tab)) {
                        g->weights_tab = i;
                        break;
                    }
                }
            }
            break;
        case SDL_MOUSEBUTTONUP:
            if (e.button.button == SDL_BUTTON_LEFT && e.button.windowID == main_id && g->screen == SCREEN_SETUP) {
                SDL_Point p = {e.button.x, e.button.y};
                setup_release(g, p);
            }
            break;
        case SDL_WINDOWEVENT:
            if (e.window.event == SDL_WINDOWEVENT_CLOSE) {
                if (g->weights_win && e.window.windowID == SDL_GetWindowID(g->weights_win)) {
                    close_weights_window(g);
                } else if (e.window.windowID == main_id) {
                    return 0;
                }
            }
            break;
        }
    }
    return 1;
}

static void render_frame(Gui *g) {
    g->slot = 0;
    if (g->frame_tex) SDL_SetRenderTarget(g->ren, g->frame_tex);
    SDL_SetRenderDrawColor(g->ren, COL_BG);
    SDL_RenderClear(g->ren);

    if (g->screen == SCREEN_SETUP) {
        form_update(g);
        draw_form(g);
    } else if (g->screen == SCREEN_PRED) {
        if (g->run_cfg.task == TASK_GENERATE) draw_generations(g);
        else draw_predictions(g);
    } else {
        draw_dashboard(g);
    }
    if (g->frame_tex) { /* scale the finished picture to the window (letterboxed by the logical size) */
        SDL_SetRenderTarget(g->ren, NULL);
        SDL_SetRenderDrawColor(g->ren, COL_BG);
        SDL_RenderClear(g->ren);
        SDL_RenderCopy(g->ren, g->frame_tex, NULL, NULL);
    }
    SDL_RenderPresent(g->ren);

    render_weights_window(g);
}

int main(int argc, char **argv) {
    int fullscreen = 1; /* the default; --windowed opens a normal 1150x700 window, F11 toggles either way */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--windowed") == 0) fullscreen = 0;
        else if (strcmp(argv[i], "--fullscreen") == 0) fullscreen = 1;
        else {
            fprintf(stderr, "usage: train_viewer [--windowed | --fullscreen]   (F11 toggles fullscreen while it runs)\n");
            return 2;
        }
    }
    /* The viewer writes to the command FIFO; if train_worker has died, that
     * write would raise SIGPIPE and kill the window instead of failing. */
    signal(SIGPIPE, SIG_IGN);

    Gui gui;
    memset(&gui, 0, sizeof(gui)); /* label_tex[]/digit_tex etc. must start NULL - see the lazy-create checks in draw_text() */
    gui.cmd_fd = -1;
    gui.zoom = 1.0f;
    gui.yaw = 0.6f;
    gui.pitch = 0.75f;
    gui.zoom3d = 1.0f;
    form_defaults(&gui);
    if (gui_init(&gui, fullscreen) != 0) return 1;
    SDL_StartTextInput();

    if (mkfifo(TRAIN_FIFO_PATH, 0600) != 0 && errno != EEXIST) {
        fprintf(stderr, "train_viewer: mkfifo(%s) failed: %s\n", TRAIN_FIFO_PATH, strerror(errno));
        gui_shutdown(&gui);
        return 1;
    }
    fprintf(stderr, "train_viewer: waiting for train_worker to connect to %s ...\n", TRAIN_FIFO_PATH);
    int data_fd = open(TRAIN_FIFO_PATH, O_RDONLY); /* blocks until the worker opens it for writing */
    if (data_fd < 0) {
        fprintf(stderr, "train_viewer: could not open fifo for reading: %s\n", strerror(errno));
        gui_shutdown(&gui);
        return 1;
    }
    /* Nothing is streamed until the user clicks Start, so there's nothing to
     * read yet - non-blocking lets the loop below keep pumping events and
     * rendering the idle UI instead of stalling on the first message. */
    fcntl(data_fd, F_SETFL, O_NONBLOCK);

    /* Second FIFO, viewer->worker. open() blocks until the worker's read end
     * is open, which by this point it already is (the worker opens it right
     * after the data FIFO handshake above, before doing anything else). */
    if (mkfifo(TRAIN_CMD_FIFO_PATH, 0600) != 0 && errno != EEXIST) {
        fprintf(stderr, "train_viewer: mkfifo(%s) failed: %s\n", TRAIN_CMD_FIFO_PATH, strerror(errno));
        close(data_fd);
        gui_shutdown(&gui);
        return 1;
    }
    gui.cmd_fd = open(TRAIN_CMD_FIFO_PATH, O_WRONLY);
    if (gui.cmd_fd < 0) {
        fprintf(stderr, "train_viewer: could not open command fifo for writing: %s\n", strerror(errno));
        close(data_fd);
        gui_shutdown(&gui);
        return 1;
    }
    fprintf(stderr, "train_viewer: worker connected. Fill in the settings and click Start.\n");

    Rx rx;
    memset(&rx, 0, sizeof(rx));
    for (;;) {
        if (!pump_events(&gui)) break;
        rx_pump(&rx, data_fd, &gui);
        render_frame(&gui);
        SDL_Delay(8); /* avoid busy-spinning the CPU now that nothing blocks this loop */
    }

    free(rx.buf);
    close(data_fd);
    if (gui.cmd_fd >= 0) close(gui.cmd_fd);
    gui_shutdown(&gui);
    return 0;
}
