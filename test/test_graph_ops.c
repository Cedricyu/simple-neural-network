/* Finite-difference gradient checks for the ops a residual network needs: add, batch norm, max pool and
 * global average pool, plus a residual block (a tensor used by two consumers) to check that gradients
 * from both paths add up. For every op: L = sum(R * out) with a fixed random R; the analytic gradient
 * (tensor_backward with grad_out = R) must match (L(w+eps) - L(w-eps)) / 2eps for every input element
 * and every parameter. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "batchnorm.cuh"
#include "conv2d.cuh"
#include "pool.cuh"
#include "tensor.cuh"
#include "tensorops.cuh"
#include "test.h"

#define MAX_TMP 8

typedef struct Scratch {
    Tensor *t[MAX_TMP];
    int n;
} Scratch;

static Tensor *keep(Scratch *s, Tensor *t) {
    s->t[s->n++] = t;
    return t;
}

static void scratch_free(Scratch *s) {
    for (int i = s->n - 1; i >= 0; --i) tensor_free(s->t[i]);
    s->n = 0;
}

typedef Tensor *(*BuildFn)(Tensor **leaves, void *user, Scratch *s);

static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

static Tensor *make_leaf(int ndim, const int *shape, float scale, float offset) {
    int shp[4];
    memcpy(shp, shape, sizeof(int) * ndim);
    Tensor *t = tensor_create(ndim, shp, 1);
    int n = tensor_numel(ndim, shp);
    float *h = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) h[i] = offset + scale * frand();
    tensor_set_data(t, h);
    free(h);
    return t;
}

/* All values distinct and at least 0.05 apart, so no finite-difference step (+-0.01) can reorder two
 * elements of a pooling window - max pooling is not differentiable at ties. */
static Tensor *make_leaf_spaced(int ndim, const int *shape) {
    int shp[4];
    memcpy(shp, shape, sizeof(int) * ndim);
    Tensor *t = tensor_create(ndim, shp, 1);
    int n = tensor_numel(ndim, shp);
    float *h = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) h[i] = 0.05f * (float)i;
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        float tmp = h[i];
        h[i] = h[j];
        h[j] = tmp;
    }
    tensor_set_data(t, h);
    free(h);
    return t;
}

static double loss_of(Tensor *out, const float *R, float *host) {
    tensor_to_host(out, host);
    int n = tensor_numel(out->ndim, out->shape);
    double l = 0.0;
    for (int i = 0; i < n; ++i) l += (double)R[i] * host[i];
    return l;
}

/* Returns the number of leaf elements whose analytic and numeric gradients disagree. */
static int check_op(const char *name, Tensor **leaves, int n_leaves, BuildFn build, void *user) {
    const float eps = 1e-2f;
    Scratch s = {{0}, 0};
    Tensor *out = build(leaves, user, &s);
    if (!out) {
        printf("  %-34s FAILED to build\n", name);
        scratch_free(&s);
        return 1;
    }
    int m = tensor_numel(out->ndim, out->shape);
    float *R = (float *)malloc(sizeof(float) * m);
    float *host = (float *)malloc(sizeof(float) * m);
    for (int i = 0; i < m; ++i) R[i] = frand();

    for (int l = 0; l < n_leaves; ++l) tensor_zero_grad(leaves[l]);
    Tensor *seed = tensor_create(out->ndim, out->shape, 0);
    tensor_set_data(seed, R);
    tensor_backward(out, seed);
    tensor_free(seed);
    float *ana[8];
    for (int l = 0; l < n_leaves; ++l) {
        int n = tensor_numel(leaves[l]->ndim, leaves[l]->shape);
        ana[l] = (float *)malloc(sizeof(float) * n);
        tensor_grad_to_host(leaves[l], ana[l]);
    }
    scratch_free(&s);

    int bad = 0;
    double worst = 0.0;
    for (int l = 0; l < n_leaves; ++l) {
        int n = tensor_numel(leaves[l]->ndim, leaves[l]->shape);
        float *orig = (float *)malloc(sizeof(float) * n);
        tensor_to_host(leaves[l], orig);
        float *work = (float *)malloc(sizeof(float) * n);
        for (int j = 0; j < n; ++j) {
            double lp, lm;
            memcpy(work, orig, sizeof(float) * n);
            work[j] = orig[j] + eps;
            tensor_set_data(leaves[l], work);
            Scratch sp = {{0}, 0};
            lp = loss_of(build(leaves, user, &sp), R, host);
            scratch_free(&sp);
            work[j] = orig[j] - eps;
            tensor_set_data(leaves[l], work);
            Scratch sm = {{0}, 0};
            lm = loss_of(build(leaves, user, &sm), R, host);
            scratch_free(&sm);
            double num = (lp - lm) / (2.0 * eps);
            double a = ana[l][j];
            double err = fabs(a - num) / (1.0 + 0.5 * (fabs(a) + fabs(num)));
            if (err > worst) worst = err;
            if (err > 2e-2) {
                if (bad < 3) printf("    leaf %d elem %d: analytic %.5f numeric %.5f\n", l, j, a, num);
                bad++;
            }
        }
        tensor_set_data(leaves[l], orig);
        free(orig);
        free(work);
        free(ana[l]);
    }
    printf("  %-34s %s (worst relative error %.2e)\n", name, bad ? "FAIL" : "ok  ", worst);
    free(R);
    free(host);
    return bad;
}

/* -- builders ------------------------------------------------------------- */

static Tensor *build_add(Tensor **leaves, void *user, Scratch *s) {
    (void)user;
    return keep(s, tensor_add(leaves[0], leaves[1]));
}

static Tensor *build_bn(Tensor **leaves, void *user, Scratch *s) {
    (void)leaves;
    return keep(s, batchnorm_forward((BatchNorm *)user, leaves[0], 1));
}

static Tensor *build_maxpool_2_2(Tensor **leaves, void *user, Scratch *s) {
    (void)user;
    return keep(s, maxpool2d_forward(leaves[0], 2, 2));
}

static Tensor *build_maxpool_3_1(Tensor **leaves, void *user, Scratch *s) {
    (void)user;
    return keep(s, maxpool2d_forward(leaves[0], 3, 1));
}

static Tensor *build_gap(Tensor **leaves, void *user, Scratch *s) {
    (void)user;
    return keep(s, global_avg_pool_forward(leaves[0]));
}

/* x -> bn -> relu -> pool, added back onto x pooled the same way: a residual-shaped diamond where x has
 * two consumers and the add merges the two paths. */
static Tensor *build_residual(Tensor **leaves, void *user, Scratch *s) {
    BatchNorm *bn = (BatchNorm *)user;
    Tensor *x = leaves[0];
    Tensor *f = keep(s, batchnorm_forward(bn, x, 1));
    f = keep(s, tensor_relu(f));
    Tensor *sum = keep(s, tensor_add(f, x));
    Tensor *r = keep(s, tensor_relu(sum));
    return keep(s, global_avg_pool_forward(r));
}

static Tensor *build_conv(Tensor **leaves, void *user, Scratch *s) {
    Conv2D *cv = (Conv2D *)user;
    (void)leaves; /* leaves = {x, weight, bias}: the weight and bias tensors are the layer's own */
    return keep(s, conv2d_forward(cv, leaves[0]));
}

/* Gives a batch norm non-trivial scale and shift so the checks aren't run at gamma=1, beta=0. */
static void randomize_bn(BatchNorm *bn) {
    int c = bn->channels;
    float *h = (float *)malloc(sizeof(float) * c);
    for (int i = 0; i < c; ++i) h[i] = 1.0f + 0.5f * frand();
    tensor_set_data(bn->gamma, h);
    for (int i = 0; i < c; ++i) h[i] = 0.5f * frand();
    tensor_set_data(bn->beta, h);
    free(h);
}

void test_graph_ops(void) {
    printf("=== graph ops: finite-difference gradient checks ===\n");
    srand(7);
    int failures = 0;

    {
        int shp[4] = {2, 3, 4, 4};
        Tensor *a = make_leaf(4, shp, 1.0f, 0.0f), *b = make_leaf(4, shp, 1.0f, 0.0f);
        Tensor *leaves[2] = {a, b};
        failures += check_op("add", leaves, 2, build_add, NULL);
        tensor_free(a);
        tensor_free(b);
    }
    {
        int shp[4] = {3, 2, 3, 3};
        BatchNorm *bn = batchnorm_new(2);
        randomize_bn(bn);
        Tensor *x = make_leaf(4, shp, 1.0f, 0.3f);
        Tensor *leaves[3] = {x, bn->gamma, bn->beta};
        failures += check_op("batchnorm (B,C,H,W)", leaves, 3, build_bn, bn);
        tensor_free(x);
        batchnorm_free(bn);
    }
    {
        int shp[2] = {5, 4};
        BatchNorm *bn = batchnorm_new(4);
        randomize_bn(bn);
        Tensor *x = make_leaf(2, shp, 2.0f, -0.5f);
        Tensor *leaves[3] = {x, bn->gamma, bn->beta};
        failures += check_op("batchnorm (B,F)", leaves, 3, build_bn, bn);
        tensor_free(x);
        batchnorm_free(bn);
    }
    {
        int shp[4] = {2, 2, 6, 6};
        Tensor *x = make_leaf_spaced(4, shp);
        Tensor *leaves[1] = {x};
        failures += check_op("maxpool 2x2 stride 2", leaves, 1, build_maxpool_2_2, NULL);
        failures += check_op("maxpool 3x3 stride 1 (overlap)", leaves, 1, build_maxpool_3_1, NULL);
        failures += check_op("global average pool", leaves, 1, build_gap, NULL);
        tensor_free(x);
    }
    {
        int shp[4] = {3, 2, 4, 4};
        BatchNorm *bn = batchnorm_new(2);
        randomize_bn(bn);
        Tensor *x = make_leaf(4, shp, 1.0f, 0.0f);
        Tensor *leaves[3] = {x, bn->gamma, bn->beta};
        failures += check_op("residual block (x used twice)", leaves, 3, build_residual, bn);
        tensor_free(x);
        batchnorm_free(bn);
    }

    /* conv2d: stride / padding / kernel combinations, including the 1x1 shortcut conv and a batch big
     * enough that the weight-gradient reduction spans several threads per tap. */
    {
        static const int cfgs[3][5] = {{3, 3, 1, 1, 6}, {2, 3, 2, 1, 7}, {3, 1, 2, 0, 6}}; /* cin, k, stride, pad, side */
        static const char *names[3] = {"conv2d 3x3 s1 p1", "conv2d 3x3 s2 p1, 7x7 input", "conv2d 1x1 s2 p0"};
        for (int t = 0; t < 3; ++t) {
            int shp[4] = {17, cfgs[t][0], cfgs[t][4], cfgs[t][4]};
            Conv2D *cv = conv2d_new(cfgs[t][0], 2, cfgs[t][1], cfgs[t][2], cfgs[t][3], 1);
            float *h = (float *)malloc(sizeof(float) * 2);
            h[0] = 0.3f;
            h[1] = -0.2f;
            tensor_set_data(cv->bias, h);
            free(h);
            Tensor *x = make_leaf(4, shp, 1.0f, 0.0f);
            Tensor *leaves[3] = {x, cv->weight, cv->bias};
            failures += check_op(names[t], leaves, 3, build_conv, cv);
            tensor_free(x);
            conv2d_free(cv);
        }
    }

    /* eval mode must normalize with the running statistics, not the batch's own. */
    {
        int shp[4] = {4, 2, 3, 3};
        BatchNorm *bn = batchnorm_new(2);
        Tensor *x = make_leaf(4, shp, 1.0f, 3.0f);
        for (int i = 0; i < 80; ++i) tensor_free(batchnorm_forward(bn, x, 1)); /* running stats converge to x's */
        Tensor *train_out = batchnorm_forward(bn, x, 1);
        Tensor *eval_out = batchnorm_forward(bn, x, 0);
        int n = tensor_numel(4, shp);
        float *a = (float *)malloc(sizeof(float) * n), *b = (float *)malloc(sizeof(float) * n);
        tensor_to_host(train_out, a);
        tensor_to_host(eval_out, b);
        double worst = 0.0;
        for (int i = 0; i < n; ++i) worst = fmax(worst, fabs(a[i] - b[i]));
        int ok = worst < 5e-2; /* unbiased running var vs biased batch var differ by n/(n-1) */
        printf("  %-34s %s (max |train - eval| %.2e)\n", "batchnorm eval uses running stats", ok ? "ok  " : "FAIL", worst);
        failures += ok ? 0 : 1;
        free(a);
        free(b);
        tensor_free(train_out);
        tensor_free(eval_out);
        tensor_free(x);
        batchnorm_free(bn);
    }

    printf(failures ? "graph ops: %d FAILURES\n" : "graph ops: all passed\n", failures);
    if (failures) exit(1);
}
