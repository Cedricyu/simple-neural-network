# tinyDL

A small deep-learning framework written in plain C + CUDA (no C++/STL in the framework itself):
reverse-mode autograd, Linear / Conv2D / BatchNorm / pooling / residual add, SGD with momentum,
and a live SDL2 GUI where you build a model by wiring blocks together, train it, and watch the loss and
the weights while it runs.

The aim is to see how CUDA kernel design affects neural-network training, on a codebase small
enough to read in an afternoon.

![Settings screen with the node editor](docs/gui_setup.png)

## Quick start

```bash
sudo apt install -y libsdl2-dev libsdl2-ttf-dev      # GUI only
# download a dataset (see "Datasets" below), e.g. Fashion-MNIST, then:
make train_gui
```

Requirements: an NVIDIA GPU, the CUDA Toolkit (`nvcc` on `PATH`) and `gcc`. If `nvcc --version` is
newer than the CUDA version `nvidia-smi` reports (common on WSL2), build with a matching toolkit:
`make NVCC=/usr/local/cuda-12.9/bin/nvcc`.

## Layout

```plaintext
module/   tensor + autograd, ops (matmul, relu, flatten, add), Linear, Conv2D, BatchNorm, pooling, CUDA kernels
test/     tests, plus the GUI: train_worker.c (CUDA trainer), train_viewer.c (SDL2), train_record.h (shared config)
tool/     dataset converters (NEU-CLS, CIFAR-10), GPU info
```

The SGD step lives inline as `tensor_update_momentum()` in `module/tensor.cu`.

## Tests

```bash
make test        # builds test_main and runs it
```

- **`test_graph_ops`** — finite-difference gradient checks for add, batchnorm (4D and 2D), max pool,
  global average pool, conv2d (stride / padding / 1x1) and a residual block where one tensor feeds
  two consumers.
- **`test_linear`**, **`test_conv2d`** — an MLP and a tiny ConvNet on synthetic tasks.
- **`test_mnist`** — a 4-layer strided ConvNet on MNIST, ~97% test accuracy in 3 epochs. Skipped with a
  message if `data/mnist/` is missing.

## Live training GUI

`make train_gui` opens the **settings screen**: training parameters on the left, a **node editor** for
the model on the right. Everything is validated continuously: every block shows the shape it outputs,
the parameter count updates live, and anything that can't run (shape mismatch, a model that doesn't end
in one output per class, a batch that won't fit in GPU memory, ...) is shown in red with the offending
block outlined, instead of reaching the GPU. Start switches to the live dashboard; **restart** re-runs,
**settings** goes back to the form.

| field | meaning |
|---|---|
| dataset | folders under `data/`; image size, channels and class count are read from the files, and picking one loads its suggested settings |
| epochs, batch size, learning rate, seed | |
| momentum | SGD momentum, 0 = plain SGD |
| normalize | 1 = standardize pixels per channel |
| augment | 0 = none, 1 = random horizontal + vertical flips, 2 = horizontal flip + shift by up to 4 px |
| fc viz pool, weights every N | weights viewer only |

### Model editor

The model is a graph: a node reads the output of whatever its input dot is wired to, and one output
can feed several nodes, so branches and skip connections work.

| block | parameters | needs |
|---|---|---|
| conv | out channels 1-64, kernel 1-7, stride 1-4, padding 0-3 | an image |
| relu | | anything |
| bn | learned scale and shift per channel; batch size >= 2 | anything |
| pool | max pool, window 1-4, stride 1-4 | an image |
| gap | global average pool, gives C values (already flat) | an image |
| add | two inputs, summed | two tensors of the same shape |
| flatten | (C,H,W) to C*H*W values | an image |
| fc | out features 1-1024 | a flat input |

The graph must end in a block with one value per class; limits are 32 blocks and 16 conv + fc layers.

- **Build** — click a palette chip to append a block, or drag it onto the canvas. Drag from an output dot
  to an input dot to wire; grab a wired input dot to move its wire; `x` deletes a block and re-joins its
  neighbours; scroll or drag the background to pan; `arrange` lays the nodes out.
- **Presets** — `cnn` (4 conv + fc, the default) and `resnet` (stem, an identity residual block, a stride-2
  block with a 1x1-conv shortcut, global average pool, fc).
- **Read the graph** — wires are blue between neighbours in evaluation order and pink for skips and
  branches; hover a block to light up everything it reads and everything that reads it.

![The resnet preset in the node editor](docs/gui_resnet_editor.png)

### Training dashboard

A **model graph** across the top (a box per layer with its output shape, relu / bn folded in as
`+bn +relu`, residual shortcuts as pink arcs, hover to trace), the loss curve, the current sample
(green border = predicted correctly), the running test accuracy, and the weights viewer.

![Training window with the ResNet dependency graph](docs/gui_resnet_run.png)

### Weights viewer

"View layer weights" opens a second window with one tab per conv / fc layer, drawn in 3D and updated
live: conv filters as bars (height = weight, blue < 0 < red), fc matrices as a height-field surface.
Drag to rotate, scroll to zoom, `0` resets.

![fc1 weights in 3D](docs/weights_fc1.png)

### Two processes

`train_worker` (pure CUDA: trains, streams progress) and `train_viewer` (pure SDL2: form and dashboard)
are separate executables talking over two FIFOs (`/tmp/tinydl_train.fifo`, `/tmp/tinydl_train_cmd.fifo`),
so a GUI bug can't take down a training run and vice versa. The worker re-validates every config it
receives with the same `train_cfg_derive()` the form uses.

## Datasets

A dataset is a folder `data/<name>/` with IDX files (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`,
and optionally `t10k-*` for a test split evaluated every epoch). Images are grey or colour, up to
128x128, 2-16 classes. Colour images use a 4-dimensional variant (magic `0x00000804`: count, channels,
rows, cols, each image stored channel by channel). Optional `classes.txt` names the classes; optional
`defaults.txt` (`key value` lines: `epochs`, `batch_size`, `lr`, `momentum`, `normalize`, `augment`)
is loaded into the form when the dataset is picked.

| dataset | get it | result |
|---|---|---|
| MNIST | `curl` below | ~97% (`test_mnist`) |
| Fashion-MNIST (GUI default) | `curl` below | ~88% in 3 epochs |
| NEU-CLS (steel defects, 6 classes, 1440 images) | `python3 tool/prepare_neu_cls.py` | ~90% (CNN), 88-93% (ResNet) |
| CIFAR-10 (colour, 10 classes) | `python3 tool/prepare_cifar10.py` | ~73% after 10 epochs (ResNet, width 32) |

```bash
# MNIST (Fashion-MNIST: BASE=http://fashion-mnist.s3-website.eu-central-1.amazonaws.com, into data/fashion-mnist)
mkdir -p data/mnist && cd data/mnist
BASE=https://ossci-datasets.s3.amazonaws.com/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sS -o "$f.gz" "$BASE/$f.gz" && gunzip -f "$f.gz"
done
cd -

pip install pyarrow numpy pillow          # for the two converters
```

Notes from training the harder sets: NEU-CLS needs per-epoch shuffling (the files are sorted by class),
standardization, momentum 0.9, flips and lr ~1e-4 for the CNN (its `defaults.txt` has them); a residual
network does not beat the CNN on 1440 images. The CIFAR-10 run was stopped at epoch 10 of 15
(about 50 s/epoch), so ~73% is a partial result, not a tuned one.

![NEU-CLS samples, one class per row](docs/neu_cls_samples.png)

## Development

```bash
make format                    # clang-format
pre-commit run --all-files     # same check as the commit hook
make clean
```
