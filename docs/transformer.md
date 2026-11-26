# Transformer

[< tinyDL](../README.md)

A token sequence is a `(D, T, 1)` tensor: D features per token, T tokens. `patch` turns an image into one,
`fc` then works on every token, `add` is the residual connection, and `attn` is one block (module `Attention` in
`module/transformer.cu`) that holds its own q,k,v and output projection weights; no convolution is involved. The
`vit` preset (Classification) is a pre-norm vision transformer:

```
patch 4x4 (cut only) - fc 32 (16 -> 32 on every patch) - pos
  [ ln - attn (4 heads) - add(x)
    ln - fc 64 - gelu - fc 32 - add ]      one such block; stack more by hand
ln - gap (mean over tokens) - fc
```

Loading it also selects Adam 0.001. On Fashion-MNIST (D = 32, 49 tokens, 11.1k parameters) it reaches 86.0% test accuracy after 6 epochs (about 6.5 s each, no augmentation
or dropout). The `fc` layers that work on every token (the patch projection, the mlp's) have no picture in the
[weights viewer](weights-viewer.md) - only an fc on a flat vector, like the final one, has a neuron view - and the weights of
`attn`'s projections are not shown; the attention view below shows what it does instead.

The weights window shows an **attention** view when you click an `attn` box: for the dashboard's sample it shows the
sample with its token grid, and for the token you pick (click a patch; the blue box) what that token attends to,
as the average over heads on the big image and per head beside it, plus every head's full token x token matrix
(rows: the token that looks, columns: the token looked at; a click on a row picks that token). Each head is captioned
with the channels it owns (`ch 8-15`: every head has its own slice of the q, k and v channels - a head is not "the q
head" or "the k-v pair") and two words on how it behaves for the sample on screen: `self` / `local` / `mid` / `far`
(how far from itself it looks, on the token grid) and `focused` / `mixed` / `spread` (how concentrated its weights are).
They describe behaviour at that moment; what a head is for is learned and not fixed. Several `attn` blocks
get a chip each. The maps are sent with every batch, so they change as the model trains; layers whose maps would
exceed 262144 values (heads x tokens x tokens) are left out. Attention is quadratic in the number of tokens (limit 1024). There is no dropout,
class token or causal masking; this is the vision (classification) transformer only.
