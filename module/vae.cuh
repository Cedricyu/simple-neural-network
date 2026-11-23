#ifndef VAE_H
#define VAE_H

#include "tensor.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/* The VAE's sampling step. x is (B, 2*Z): the first Z columns are the mean mu, the last Z the log-variance
 * logvar (clamped to [-10, 10]); eps is (B, Z) standard-normal noise supplied by the caller (all zeros gives
 * the deterministic z = mu used for evaluation). Returns z = mu + exp(logvar / 2) * eps, (B, Z).
 *
 * Backward returns the gradient of z's consumers pushed through that formula PLUS the gradient of
 * beta * KL(N(mu, exp(logvar)) || N(0, 1)) averaged over the batch - so the loss the trainer minimizes is
 *   mean over the batch of (reconstruction + beta * KL),
 * and the KL term needs no separate op. eps must stay alive until the backward pass has run. */
Tensor *reparam_forward(Tensor *x, Tensor *eps, float beta);
Tensor *reparam_backward(Tensor *x, Tensor *out, Tensor *grad_out);

#ifdef __cplusplus
}
#endif

#endif
