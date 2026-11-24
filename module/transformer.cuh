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

/* Per-token linear layer: x is (B, Din, H, W) with the H*W positions as tokens, W is (Din, Dout), bias (Dout);
 * y[b, o, t] = sum_i x[b, i, t] * W[i, o] + bias[o]. The same fc layer is applied to every token. */
Tensor *tensor_token_linear(Tensor *x, Tensor *weight, Tensor *bias);
Tensor *tensor_token_linear_backward_input(Tensor *x, Tensor *weight, Tensor *grad_out);
Tensor *tensor_token_linear_backward_weight(Tensor *weight, Tensor *x, Tensor *grad_out);
Tensor *tensor_token_linear_backward_bias(Tensor *bias, Tensor *ctx, Tensor *grad_out);

/* Cuts a (B, C, H, W) image into non-overlapping p x p patches and flattens each patch (channel-major) into one
 * token: (B, C*p*p, (H/p)*(W/p), 1). H and W must be multiples of p. A pure re-indexing; backward puts every value
 * back where it came from. */
Tensor *tensor_patchify(Tensor *x, int p);
Tensor *tensor_patchify_backward(Tensor *x, Tensor *out, Tensor *grad_out);

/* The patch embedding of a vision transformer: patchify, then one linear projection (C*p*p -> D) shared by every
 * patch. keep[0] = the patches, owned by the caller like the intermediates of attention_layer_forward. */
typedef struct PatchEmbed {
    int in_c, patch, dim;
    Tensor *w; /* (C*p*p, D) */
    Tensor *b; /* (D) */
} PatchEmbed;

PatchEmbed *patchembed_new(int in_c, int patch, int dim);
void patchembed_free(PatchEmbed *self);
Tensor *patchembed_forward(PatchEmbed *self, Tensor *x, Tensor *keep[1]);

/* A multi-head self-attention layer with its own weights - the module an `attn` block is. D channels in and out
 * (D a multiple of `heads`); it projects every token to queries, keys and values with one (D, 3D) matrix, runs
 * attention_forward on them, and mixes the heads with a (D, D) output projection. */
typedef struct Attention {
    int dim, heads;
    Tensor *wqkv; /* (D, 3D) */
    Tensor *bqkv; /* (3D) */
    Tensor *wo;   /* (D, D) */
    Tensor *bo;   /* (D) */
} Attention;

Attention *attention_new(int dim, int heads); /* Xavier-uniform weights, zero biases */
void attention_free(Attention *self);
/* keep[0] = the packed q,k,v, keep[1] = the attention output before the projection: intermediates the result's
 * backward pass reads, so the caller owns them and frees them after the result (like the result itself). */
Tensor *attention_layer_forward(Attention *self, Tensor *x, Tensor *keep[2]);

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
