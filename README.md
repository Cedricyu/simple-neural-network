# tinyDL

A small deep-learning framework written in plain C + CUDA (no C++/STL in the framework itself):
reverse-mode autograd, Linear / Conv2D / BatchNorm / LayerNorm / attention / pooling / upsampling / residual add / channel concat,
SGD with momentum and Adam, classification (CNNs, ResNets, a small vision transformer), segmentation and a VAE, and a live SDL2 GUI where you build a model by
wiring blocks together (a CNN, a ResNet, a U-Net, a VAE), train it, and watch the loss, the predictions /
generated images and the weights while it runs.

The aim is to see how CUDA kernel design affects neural-network training, on a codebase small
enough to read in an afternoon.

![Settings screen with the node editor](docs/gui_setup.png)

## Quick start

```bash
sudo apt install -y libsdl2-dev libsdl2-ttf-dev      # GUI only

## Layout

```plaintext
module/   tensor + autograd, ops (matmul, relu, flatten, add, concat, reshape), Linear, Conv2D, BatchNorm, pooling, upsample, VAE sampling, CUDA kernels; transformer.cu: layer norm, gelu, position embedding, patchify, grayscale, attention
test/     tests, plus the GUI: train_worker.c (CUDA trainer), train_viewer.c (SDL2), train_record.h (shared config)
tool/     dataset converters (NEU-CLS, CIFAR-10, brain MRI), GPU info
```

The optimizers (SGD with momentum, and Adam with bias correction) are `tensor_update_momentum()` and `tensor_update_adam()` in `module/tensor.cu`.

## Tests

```bash
make test        # builds test_main and runs it
```

- **`test_graph_ops`** — finite-difference gradient checks for add, concat, batchnorm (4D and 2D), layer norm, gelu, position embedding, multi-head attention and a whole transformer block, max pool,
  upsample, global average pool, conv2d (stride / padding / 1x1), a residual block where one tensor feeds
  two consumers, a pool-upsample-concat skip, reshape, the VAE sampling step (including its KL gradient)
  the segmentation loss (cross entropy + Dice), and Adam (the first step moves every weight by about the
  learning rate whatever the gradient's size; it minimizes a quadratic).
- **`test_linear`**, **`test_conv2d`** — an MLP and a tiny ConvNet on synthetic tasks.
- **`test_mnist`** — a 4-layer strided ConvNet on MNIST, ~97% test accuracy in 3 epochs. Skipped with a
  message if `data/mnist/` is missing.

## Documentation

| page | what is in it |
|---|---|
| [Live training GUI](docs/gui.md) | the settings screen, the three problems, every training field, optimizers, the two-process design |
| [Model editor](docs/model-editor.md) | the blocks (basic, preprocessing, advanced), wiring, presets, errors, residual units |
| [Transformer](docs/transformer.md) | tokens, `patch` / `attn`, the `vit` preset, the attention view |
| [Training dashboard](docs/dashboard.md) | the model graph, loss curve, GPU strip, stop button, the predictions page |
| [Weights viewer](docs/weights-viewer.md) | conv filters in 3D, a conv layer's result for the current sample, the fc network view |
| [Generative model (VAE)](docs/vae.md) | the `vae` block, the generations page, results |
| [Datasets](docs/datasets.md) | the file format, how to get each dataset, segmentation datasets |

## Development

```bash
make format                    # clang-format
pre-commit run --all-files     # same check as the commit hook
make clean
```
