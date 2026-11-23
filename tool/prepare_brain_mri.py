#!/usr/bin/env python3
"""Downloads the LGG brain MRI segmentation set (TCGA, 110 patients, 3929 axial slices with a binary
tumour mask; 3-channel 256x256 slices) and converts it to the layout the trainer reads:

    data/brain-mri/train-images-idx3-ubyte    3-channel images (this project's colour IDX, magic 0x804)
    data/brain-mri/train-labels-idx1-ubyte    per slice: 1 = the slice contains tumour, 0 = it does not
    data/brain-mri/train-masks-idx3-ubyte     per pixel class index: 0 = background, 1 = tumour (magic 0x803)
    data/brain-mri/t10k-*                     same three files for the test split
    data/brain-mri/classes.txt, defaults.txt

The presence of the masks file is what makes the trainer treat the dataset as segmentation.
The split is by PATIENT (a patient's neighbouring slices look alike, so splitting by slice would leak the
test set into training). Needs: pip install numpy pillow
"""
import argparse
import io
import os
import random
import re
import struct
import urllib.request
import zipfile

import numpy as np
from PIL import Image

URL = "https://huggingface.co/datasets/gymprathap/Brain-Tumor-Segmentation/resolve/main/Brain-Tumor-Segmentation-Dataset.zip"

DEFAULTS = "epochs 30\nbatch_size 16\nlr 0.01\nmomentum 0.9\nnormalize 1\naugment 2\n"


def fetch(url, dest):
    if not os.path.exists(dest):
        print("downloading", url, "(~370 MB)")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_slices(zip_path, size):
    """Returns {patient: [(image CHW uint8, mask HW uint8), ...]} read straight from the zip."""
    z = zipfile.ZipFile(zip_path)
    names = set(z.namelist())
    out = {}
    for n in sorted(names):
        m = re.match(r"Dataset/([^/]+)/([^/]+?)_(\d+)\.tif$", n)
        if not m or n.endswith("_mask.tif"):
            continue
        mask_name = n[:-4] + "_mask.tif"
        if mask_name not in names:
            continue
        img = Image.open(io.BytesIO(z.read(n))).convert("RGB").resize((size, size), Image.BOX)
        msk = Image.open(io.BytesIO(z.read(mask_name))).convert("L").resize((size, size), Image.BOX)
        out.setdefault(m.group(1), []).append(
            (np.asarray(img).transpose(2, 0, 1).copy(), (np.asarray(msk) >= 128).astype(np.uint8)))
    return out


def write_split(out_dir, prefix, slices):
    images = np.stack([s[0] for s in slices])          # (n, 3, h, w)
    masks = np.stack([s[1] for s in slices])           # (n, h, w)
    labels = (masks.reshape(len(slices), -1).max(axis=1) > 0).astype(np.uint8)
    n, c, h, w = images.shape
    with open(os.path.join(out_dir, f"{prefix}-images-idx3-ubyte"), "wb") as f:
        f.write(struct.pack(">IIIII", 0x804, n, c, h, w))
        f.write(images.tobytes())
    with open(os.path.join(out_dir, f"{prefix}-labels-idx1-ubyte"), "wb") as f:
        f.write(struct.pack(">II", 2049, n))
        f.write(labels.tobytes())
    with open(os.path.join(out_dir, f"{prefix}-masks-idx3-ubyte"), "wb") as f:
        f.write(struct.pack(">IIII", 0x803, n, h, w))
        f.write(masks.tobytes())
    print(f"{prefix}: {n} slices, {int(labels.sum())} with tumour, tumour pixels {masks.mean() * 100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64, help="output side in pixels (a multiple of 4: the U-Net halves it twice)")
    ap.add_argument("--out", default="data/brain-mri")
    ap.add_argument("--cache", default="data/.cache-brain")
    ap.add_argument("--test-patients", type=float, default=0.2, help="fraction of patients held out for the test split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.size % 4:
        ap.error("--size must be a multiple of 4")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    zip_path = fetch(URL, os.path.join(args.cache, "brain.zip"))
    by_patient = load_slices(zip_path, args.size)
    patients = sorted(by_patient)
    random.Random(args.seed).shuffle(patients)
    n_test = max(1, round(len(patients) * args.test_patients))
    test_p, train_p = patients[:n_test], patients[n_test:]
    print(f"{len(patients)} patients: {len(train_p)} train, {len(test_p)} test")
    write_split(args.out, "train", [s for p in train_p for s in by_patient[p]])
    write_split(args.out, "t10k", [s for p in test_p for s in by_patient[p]])
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("background\ntumor\n")
    with open(os.path.join(args.out, "defaults.txt"), "w") as f:
        f.write(DEFAULTS)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
