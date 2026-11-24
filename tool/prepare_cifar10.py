#!/usr/bin/env python3
"""Downloads CIFAR-10 (50000 train / 10000 test colour photos, 32x32, 10 classes) and converts it to
the IDX layout the trainer reads:

    data/cifar10/train-images-idx3-ubyte   train-labels-idx1-ubyte
    data/cifar10/t10k-images-idx3-ubyte    t10k-labels-idx1-ubyte
    data/cifar10/classes.txt               one class name per line
    data/cifar10/defaults.txt              settings the GUI loads when the dataset is picked

The image files use this project's colour variant of IDX: magic 0x00000804, then count, channels,
rows, cols (big-endian uint32), then every image stored channel by channel (R plane, G plane, B plane).
Needs: pip install pyarrow numpy pillow
"""
import argparse
import io
import os
import struct
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Adam trains both the plain CNN (no BatchNorm, which diverges with SGD at lr 0.01) and the ResNet preset.
DEFAULTS = "epochs 20\nbatch_size 64\nlr 0.001\nmomentum 0.9\nnormalize 1\naugment 2\noptimizer 1\n"


def fetch(url, dest):
    if not os.path.exists(dest):
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
    return dest


def convert(parquet_path):
    table = pq.read_table(parquet_path)
    img_col = "img" if "img" in table.column_names else "image"
    rows = table.to_pylist()
    images = np.zeros((len(rows), 3, 32, 32), dtype=np.uint8)
    labels = np.zeros(len(rows), dtype=np.uint8)
    for i, r in enumerate(rows):
        im = Image.open(io.BytesIO(r[img_col]["bytes"])).convert("RGB")
        images[i] = np.asarray(im).transpose(2, 0, 1)  # HWC -> CHW
        labels[i] = r["label"]
    return images, labels


def write_idx(out_dir, prefix, images, labels):
    n, c, h, w = images.shape
    with open(os.path.join(out_dir, f"{prefix}-images-idx3-ubyte"), "wb") as f:
        f.write(struct.pack(">IIIII", 0x804, n, c, h, w))
        f.write(images.tobytes())
    with open(os.path.join(out_dir, f"{prefix}-labels-idx1-ubyte"), "wb") as f:
        f.write(struct.pack(">II", 2049, n))
        f.write(labels.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cifar10")
    ap.add_argument("--cache", default="data/.cache-cifar10", help="where the downloaded parquet files are kept")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    for split, prefix in (("train", "train"), ("test", "t10k")):
        path = fetch(BASE + FILES[split], os.path.join(args.cache, FILES[split]))
        images, labels = convert(path)
        write_idx(args.out, prefix, images, labels)
        print(f"{split}: {len(labels)} images 3x32x32, per-class counts {np.bincount(labels).tolist()}")
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")
    with open(os.path.join(args.out, "defaults.txt"), "w") as f:
        f.write(DEFAULTS)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
