#!/usr/bin/env python3
"""Downloads NEU-CLS (hot-rolled steel surface defects, 6 classes, 1440 train /
360 test images, 200x200) and converts it to the IDX layout the trainer reads:

    data/neu-cls/train-images-idx3-ubyte   train-labels-idx1-ubyte
    data/neu-cls/t10k-images-idx3-ubyte    t10k-labels-idx1-ubyte
    data/neu-cls/classes.txt               one class name per line
    data/neu-cls/defaults.txt              settings the GUI loads when the dataset is picked

Images are converted to grayscale and area-averaged down to --size x --size
(default 64). Needs: pip install pyarrow numpy pillow
"""
import argparse
import io
import os
import struct
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = "https://huggingface.co/datasets/newguyme/neu_cls/resolve/main/data/"
FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# Measured with the GUI's default model (4 conv + fc 128 + fc 6) at 64x64: ~90% test accuracy after 80
# epochs. The set is tiny (1440 images), so without standardization + momentum + flips the model
# overfits or doesn't train at all; lr above ~3e-4 with momentum 0.9 goes unstable.
DEFAULTS = "epochs 80\nbatch_size 64\nlr 0.0001\nmomentum 0.9\nnormalize 1\naugment 1\n"


def fetch(url, dest):
    if not os.path.exists(dest):
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
    return dest


def convert(parquet_path, size):
    rows = pq.read_table(parquet_path).to_pylist()
    images = np.zeros((len(rows), size, size), dtype=np.uint8)
    labels = np.zeros(len(rows), dtype=np.uint8)
    for i, r in enumerate(rows):
        im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("L")
        images[i] = np.asarray(im.resize((size, size), Image.BOX))
        labels[i] = r["label"]
    return images, labels


def write_idx(out_dir, prefix, images, labels):
    n, h, w = images.shape
    with open(os.path.join(out_dir, f"{prefix}-images-idx3-ubyte"), "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, h, w))
        f.write(images.tobytes())
    with open(os.path.join(out_dir, f"{prefix}-labels-idx1-ubyte"), "wb") as f:
        f.write(struct.pack(">II", 2049, n))
        f.write(labels.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64, help="output image side in pixels")
    ap.add_argument("--out", default="data/neu-cls")
    ap.add_argument("--cache", default="data/.cache-neu-cls", help="where the downloaded parquet files are kept")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    for split, prefix in (("train", "train"), ("test", "t10k")):
        path = fetch(BASE + FILES[split], os.path.join(args.cache, FILES[split]))
        images, labels = convert(path, args.size)
        write_idx(args.out, prefix, images, labels)
        print(f"{split}: {len(labels)} images {args.size}x{args.size}, per-class counts {np.bincount(labels).tolist()}")
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")
    with open(os.path.join(args.out, "defaults.txt"), "w") as f:
        f.write(DEFAULTS)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
