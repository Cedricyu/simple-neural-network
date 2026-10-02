## Project Purpose

```plaintext
The goal of this project is to explore and analyze how CUDA APIs can optimize and accelerate neural network computation.
By implementing fundamental neural network components (Tensor autograd, Linear, Conv2D, SGD) directly in CUDA, this project aims to:

Understand how CUDA kernel design and memory access patterns affect performance.

Investigate how GPU parallelism can accelerate both training and inference processes.

Identify bottlenecks across different layers and propose potential optimization strategies.

Build an observable, testable, and extensible CUDA-based neural network framework, serving as a foundation for more complex architectures such as CNNs or Transformers.
```


## 🛠️ Build & Run Guide

This project uses a Makefile to compile and run a small CUDA-based neural network framework, written in plain C + CUDA (no C++/STL in the framework itself).

## 📁 Project Structure

```plaintext
├── Makefile
├── README.md
├── module/
│   ├── tensor.cu(.cuh)             # Tensor struct + reverse-mode autograd engine
│   ├── tensorops.cu(.cuh)          # matmul / bias-add / relu ops and their backward fns
│   ├── cudakernels.cu(.cuh)        # tiled matmul, transpose, bias CUDA kernels
│   ├── activationkernels.cu(.cuh)  # sigmoid/relu/tanh/gelu/... CUDA kernels
│   ├── linear.cu(.cuh)             # Linear (fully-connected) layer
│   ├── conv2d.cu(.cuh)             # Conv2D layer (stride, padding, optional bias)
│   └── convkernels.cu(.cuh)        # conv2d forward/backward CUDA kernels
├── tool/
│   └── CudaDeviceInfo.cu(.cuh)     # prints GPU info at startup
├── test/
│   ├── test_main.c                 # entry point; runs the tests below in sequence
│   ├── test_linear.c               # 4-layer MLP on a synthetic checkerboard task
│   ├── test_conv2d.c               # tiny 2-layer ConvNet sanity check
│   ├── test_mnist.c                # 4-layer strided ConvNet trained on real MNIST
│   ├── train_worker.c              # same ConvNet, trained on Fashion-MNIST, streams progress to train_viewer
│   ├── train_viewer.c              # SDL2 live-training window (chart, sample, per-layer weights)
│   ├── train_record.h              # wire format shared between train_worker and train_viewer
│   ├── mnist.c(.h)                 # IDX-format loader (works for MNIST and Fashion-MNIST alike)
│   ├── dataset.c(.h), train_config.c(.h)
│   └── test.h
└── data/
    ├── mnist/                      # (gitignored) downloaded MNIST files, used by test_mnist
    └── fashion-mnist/              # (gitignored) downloaded Fashion-MNIST files, used by train_worker
```

There is no separate `optimizer/` module yet — the SGD step lives inline as
`tensor_update()` in `module/tensor.cu`. `Makefile`'s `OPT_DIR` is a reserved
slot for when that gets split out.


## ✅ Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (version 11.x or later), with `nvcc` on your `PATH`
- `gcc` (the `test/` sources are plain C, not C++)

Check your toolkit:

```bash
nvcc --version
```

If you don't have it yet, follow the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Check your driver and the CUDA version it supports:

```bash
nvidia-smi
```

**If `nvcc --version` reports a newer CUDA version than `nvidia-smi`'s "CUDA
Version" field** (common on WSL2 when the toolkit gets upgraded ahead of the
Windows-side driver), runtime calls like `cudaMalloc` fail with `CUDA driver
version is insufficient for CUDA runtime version`. Fix it by building with a
toolkit that matches your driver instead of the default `nvcc` — install the
matching `cuda-<version>` package alongside your current one, then point the
Makefile at it:

```bash
make NVCC=/usr/local/cuda-12.9/bin/nvcc   # match whatever nvidia-smi reports
```

For code style, you can install:

```bash
pre-commit install
```


## 🔨 Build Commands

### Build everything

Compiles every `.cu` under `module/`/`tool/` and every `.c` under `test/`, and
links them into the `test_main` executable.

```bash
make
```

### Run tests

Builds the project (if needed) and runs `test_main`, which runs three
training runs back to back:

1. **`test_linear`** — a 4-layer MLP learning a synthetic checkerboard
   classification task (validates the `Linear`/`ReLU`/autograd path).
2. **`test_conv2d`** — a tiny 2-layer ConvNet learning a synthetic striped
   image classification task (validates `Conv2D` forward/backward, including
   stride and padding).
3. **`test_mnist`** — a 4-layer strided ConvNet (no pooling layer exists yet,
   so downsampling is done with `stride=2` convs) trained on the real MNIST
   dataset. Reaches ~97% test accuracy in 3 epochs (~10s/epoch on an RTX
   5060). **Requires the MNIST files to be downloaded first — see below.**

```bash
make test
```

#### Downloading MNIST

`test_mnist` expects the raw IDX-format files under `data/mnist/` (that
directory is gitignored, so it isn't shipped in the repo):

```bash
mkdir -p data/mnist && cd data/mnist
BASE=https://ossci-datasets.s3.amazonaws.com/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sS -o "$f.gz" "$BASE/$f.gz" && gunzip -f "$f.gz"
done
cd -
```

If those files are missing, `test_mnist` prints a message and skips itself
instead of failing the build.

### Live training GUI (SDL2)

```bash
make train_gui
```

Opens a window with a live scrolling loss curve, the current training sample
(green border if the model got it right, red if not), a per-layer live
weights viewer (click "view layer weights"), and the model architecture — a
5-layer ConvNet (one layer deeper than `test_mnist`'s: an extra same-resolution
conv before the final logits layer) trained on **Fashion-MNIST** (clothing
categories) instead of handwritten digits, so the two are easy to tell apart
at a glance — reaches ~88-89% test accuracy in 3 epochs, notably below
`test_mnist`'s ~97% since Fashion-MNIST is the harder dataset. Requires
Fashion-MNIST downloaded (below) plus `libsdl2-dev` and `libsdl2-ttf-dev`:

```bash
sudo apt install -y libsdl2-dev libsdl2-ttf-dev
```

#### Downloading Fashion-MNIST

`train_worker` expects the raw IDX-format files under `data/fashion-mnist/`
(same IDX format as MNIST, same filenames, just clothing photos instead of
digits — that directory is gitignored, so it isn't shipped in the repo):

```bash
mkdir -p data/fashion-mnist && cd data/fashion-mnist
BASE=http://fashion-mnist.s3-website.eu-central-1.amazonaws.com
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sS -o "$f.gz" "$BASE/$f.gz" && gunzip -f "$f.gz"
done
cd -
```

This is actually two executables, `train_worker` (pure CUDA, trains and
streams progress) and `train_viewer` (pure SDL2, draws it), handed off over a
FIFO at `/tmp/tinydl_train.fifo`. They're kept in separate processes on
purpose: while chasing a crash in an earlier single-process version (which
briefly also looked like a CUDA/WSLg driver interop issue, since linking
SDL2 pulls in GL/X11/Wayland libraries alongside a live CUDA context), the
actual bug turned out to be mundane — an uninitialized `Gui` struct causing a
stray pointer dereference inside SDL. That's fixed now, but keeping compute
and rendering as separate processes is still the safer default: it means the
training run can never be taken down by a GUI bug (or vice versa), so it's
staying this way. You can also run the two halves by hand in separate
terminals (`./train_viewer`, then `./train_worker`) if you want to watch
their logs independently.

### Code formatting

Format the code using `clang-format` to ensure consistent style across the project.

```bash
make format
```

The pre-commit hook will automatically check the formatting of your code before committing. If the formatting is not correct, it will automatically format the code for you. Also, You can run pre-commit against all files to ensure format passed before commiting your changes to ensure code quality and style consistency.

```bash
pre-commit run --all-files
```

### Clean

Clean generated files and executables.

```bash
make clean
```