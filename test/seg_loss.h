#ifndef SEG_LOSS_H
#define SEG_LOSS_H

/* Per-pixel loss for segmentation, on the host (like the classification loss): the worker copies the
 * logits back, computes loss and gradient here, and sends the gradient to the GPU.
 *
 *   loss = cross_entropy + dice_weight * (1 - mean over foreground classes of soft dice)
 *
 * Cross entropy is the mean over all B*H*W pixels. The soft dice of class k is computed over the whole
 * batch, (2*sum(p*g) + 1) / (sum(p) + sum(g) + 1) - the +1 keeps it defined (and pushes p to 0) when a
 * batch has no foreground. It is there because foreground is rare (about 1% of the pixels of the brain
 * MRI set): cross entropy alone is happy to predict "background" everywhere. Class 0 is background and
 * has no dice term. */

#include <math.h>
#include <stdlib.h>

/* logits: (B, C, H, W); mask: (B, H, W) class indices; grad: (B, C, H, W) out, d(loss)/d(logits).
 * Returns the loss, or -1 on allocation failure. */
static inline float seg_loss_grad(const float *logits, const unsigned char *mask, int B, int C, int H, int W,
                                  float dice_weight, float *grad) {
    size_t hw = (size_t)H * W, n_pix = (size_t)B * hw;
    float *p = (float *)malloc(sizeof(float) * n_pix * C);
    float *d = (float *)malloc(sizeof(float) * n_pix * C); /* d(dice loss)/d(p) */
    if (!p || !d) {
        free(p);
        free(d);
        return -1.0f;
    }
    /* softmax over the class axis of every pixel */
    for (int b = 0; b < B; ++b) {
        for (size_t i = 0; i < hw; ++i) {
            float mx = logits[((size_t)b * C) * hw + i];
            for (int c = 1; c < C; ++c) {
                float v = logits[((size_t)b * C + c) * hw + i];
                if (v > mx) mx = v;
            }
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                float e = expf(logits[((size_t)b * C + c) * hw + i] - mx);
                p[((size_t)b * C + c) * hw + i] = e;
                sum += e;
            }
            for (int c = 0; c < C; ++c) p[((size_t)b * C + c) * hw + i] /= sum;
        }
    }

    double ce = 0.0;
    for (int b = 0; b < B; ++b)
        for (size_t i = 0; i < hw; ++i) {
            int g = mask[(size_t)b * hw + i];
            ce -= logf(p[((size_t)b * C + g) * hw + i] + 1e-7f);
        }
    ce /= (double)n_pix;

    /* soft dice per foreground class, over the whole batch */
    double dice_sum = 0.0;
    for (size_t k = 0; k < n_pix * C; ++k) d[k] = 0.0f;
    for (int c = 1; c < C; ++c) {
        double inter = 0.0, psum = 0.0, gsum = 0.0;
        for (int b = 0; b < B; ++b)
            for (size_t i = 0; i < hw; ++i) {
                double pk = p[((size_t)b * C + c) * hw + i];
                double gk = mask[(size_t)b * hw + i] == c ? 1.0 : 0.0;
                inter += pk * gk;
                psum += pk;
                gsum += gk;
            }
        double denom = psum + gsum + 1.0, num = 2.0 * inter + 1.0;
        dice_sum += num / denom;
        for (int b = 0; b < B; ++b)
            for (size_t i = 0; i < hw; ++i) {
                double gk = mask[(size_t)b * hw + i] == c ? 1.0 : 0.0;
                d[((size_t)b * C + c) * hw + i] = (float)(-(2.0 * gk * denom - num) / (denom * denom) / (C - 1));
            }
    }
    double dice_loss = C > 1 ? 1.0 - dice_sum / (C - 1) : 0.0;

    /* chain through the softmax: dz_j = p_j * (d_j - sum_i p_i d_i), plus the cross-entropy part */
    for (int b = 0; b < B; ++b) {
        for (size_t i = 0; i < hw; ++i) {
            int g = mask[(size_t)b * hw + i];
            float dot = 0.0f;
            for (int c = 0; c < C; ++c) dot += p[((size_t)b * C + c) * hw + i] * d[((size_t)b * C + c) * hw + i];
            for (int c = 0; c < C; ++c) {
                size_t at = ((size_t)b * C + c) * hw + i;
                float ce_g = (p[at] - (c == g ? 1.0f : 0.0f)) / (float)n_pix;
                grad[at] = ce_g + dice_weight * p[at] * (d[at] - dot);
            }
        }
    }
    free(p);
    free(d);
    return (float)(ce + dice_weight * dice_loss);
}

#endif
