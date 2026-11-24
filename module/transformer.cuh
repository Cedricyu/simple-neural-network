#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include "tensor.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/* The pieces of a vision transformer. A token sequence is a (B, D, T, 1) tensor - D features per token,
 * T tokens - so the per-token linear layers are the existing 1x1 convolutions and a residual connection is
 * the existing add. All of these accept any (B, C, H, W) and treat the H*W positions as the tokens. */

/* Layer normalization over the channel axis: every position (token) is normalized on its own across its C
 * values, y = gamma * (x - mean) * rsqrt(var + eps) + beta. Unlike batch norm it has no running statistics
 * and behaves the same in training and evaluation. Accepts (B, C, H, W) or (B, C). */
typedef struct LayerNorm {
    int channels;
    float eps;
    Tensor *gamma; /* (C), init 1 */
    Tensor *beta;  /* (C), init 0 */
} LayerNorm;

LayerNorm *layernorm_new(int channels);
void layernorm_free(LayerNorm *self);
Tensor *layernorm_forward(LayerNorm *self, Tensor *x);
Tensor *layernorm_backward_input(Tensor *x, Tensor *gamma, Tensor *grad_out);
Tensor *layernorm_backward_gamma(Tensor *gamma, Tensor *x, Tensor *grad_out);
Tensor *layernorm_backward_beta(Tensor *beta, Tensor *x, Tensor *grad_out);

/* GELU, x * Phi(x) (exact, with erf). */
Tensor *tensor_gelu(Tensor *x);
Tensor *tensor_gelu_backward(Tensor *x, Tensor *ctx, Tensor *grad_out);

/* Learned position embedding: a (C, H, W) parameter added to every sample of a (B, C, H, W) tensor. */
Tensor *posembed_new(int c, int h, int w); /* normal(0, 0.02) */
Tensor *posembed_forward(Tensor *pos, Tensor *x);
Tensor *posembed_backward_pos(Tensor *pos, Tensor *x, Tensor *grad_out);

/* Multi-head self-attention on packed queries, keys and values: qkv is (B, 3D, H, W) with channels
 * [0, D) the queries, [D, 2D) the keys and [2D, 3D) the values, T = H*W tokens, `heads` heads of D/heads
 * channels each. The result is (B, D, H, W): per head softmax(q k^T / sqrt(D/heads)) v. It has no parameters
 * of its own - the projections before and after it are 1x1 convolutions. */
Tensor *attention_forward(Tensor *qkv, int heads);
Tensor *attention_backward(Tensor *qkv, Tensor *out, Tensor *grad_out);

#ifdef __cplusplus
}
#endif

#endif
