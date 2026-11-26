# Live training GUI

[< tinyDL](../README.md)

`make train_gui` opens the **settings screen**. The first choice, at the top left, is the **problem**, always
yours to make:

| problem | what the model predicts | dataset | presets | dashboard |
|---|---|---|---|---|
| **Classification** | one class per image | any (a dataset's labels) | `cnn`, `resnet`, `mlp`, `vit` | loss, accuracy, test accuracy |
| **Segmentation** | one class per pixel | one with masks | `unet` | scan / true mask / predicted mask, Dice and IoU |
| **Generation** | new images (a VAE) | any (no labels used) | `vae` | input / reconstruction, generations page |

Everything below follows from it: the task the trainer runs, the datasets the dropdown offers (Segmentation lists
only datasets with masks), the blocks on offer, the presets, and the default training settings (switching
problem clears the model, since another problem's model would not fit; a preset or your own graph is yours to pick).
Below the problem selector are the training parameters, and on the right a **node editor** for the model ([model editor](model-editor.md)). Everything is validated continuously: every block shows the shape it outputs,
the parameter count updates live, and anything that can't run (shape mismatch, a model that doesn't end
in one output per class, a batch that won't fit in GPU memory, ...) is shown in red with the offending
block outlined, instead of reaching the GPU. Start switches to the live dashboard; **restart** re-runs,
**settings** goes back to the form.

| field | meaning |
|---|---|
| problem | Classification, Segmentation or Generation (see above); switching clears the model and loads that problem's training settings |
| dataset | folders under `data/`; image size, channels and class count are read from the files, and picking one loads its suggested settings |
| epochs, batch size, learning rate, seed | the loss is the mean over the batch, so SGD learning rates are the usual ones (0.01-0.1; the default is 0.01) |
| optimizer | 0 = SGD with momentum, 1 = Adam (betas 0.9 / 0.999) |
| momentum | SGD momentum, 0 = plain SGD (ignored by Adam) |
| normalize | 1 = standardize pixels per channel |
| augment | 0 = none, 1 = random horizontal + vertical flips, 2 = horizontal flip + shift by up to 4 px |
| fc viz pool, weights every N | [weights viewer](weights-viewer.md) only |

The model is built in the [model editor](model-editor.md); a run is watched on the [training dashboard](dashboard.md) and in the
[weights viewer](weights-viewer.md). Datasets are described in [datasets](datasets.md).

## Optimizers

SGD with momentum is the default; Adam is one field away. Measured here, Adam is a little faster but not
dramatically: on the brain-tumour U-Net (lr 0.001 against SGD's 0.01) the test Dice after 5 / 10 / 15 epochs was
0.74 / 0.81 / 0.81 with Adam and 0.72 / 0.76 / 0.78 with SGD, within the run-to-run spread of a few points (single
runs). Adam's real advantage here is that it needs little learning-rate tuning: its steps are about the learning
rate whatever the gradient's scale, which matters for losses that sum over pixels.

## Two processes

`train_worker` (pure CUDA: trains, streams progress) and `train_viewer` (pure SDL2: form and dashboard)
are separate executables talking over two FIFOs (`/tmp/tinydl_train.fifo`, `/tmp/tinydl_train_cmd.fifo`),
so a GUI bug can't take down a training run and vice versa. The worker re-validates every config it
receives with the same `train_cfg_derive()` the form uses.
