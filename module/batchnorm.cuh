#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "tensor.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/* Batch normalization over the channel axis. Accepts (B, C, H, W) - one mean/variance
 * per channel over B*H*W values - or (B, F), where every feature is its own channel.
 * y = gamma * (x - mean) * rsqrt(var + eps) + beta. In training mode mean/var are the
 * batch's own and the running averages are updated; in eval mode the running averages
 * are used, so a prediction doesn't depend on what else is in the batch. */
typedef struct BatchNorm {
    int channels;
    float eps;
    float momentum;      /* running = (1 - momentum) * running + momentum * batch */
    Tensor *gamma;       /* (C), init 1 */
    Tensor *beta;        /* (C), init 0 */
    float *running_mean; /* device, (C) */
    float *running_var;  /* device, (C) */
    float *save_mean;    /* device, (C): the statistics the latest forward normalized with */
    float *save_invstd;  /* device, (C) */
} BatchNorm;

BatchNorm *batchnorm_new(int channels);
void batchnorm_free(BatchNorm *self);
Tensor *batchnorm_forward(BatchNorm *self, Tensor *x, int training);

Tensor *batchnorm_backward_input(Tensor *x, Tensor *gamma, Tensor *grad_out);
Tensor *batchnorm_backward_gamma(Tensor *gamma, Tensor *x, Tensor *grad_out);
Tensor *batchnorm_backward_beta(Tensor *beta, Tensor *ctx, Tensor *grad_out);

#ifdef __cplusplus
}
#endif

#endif
