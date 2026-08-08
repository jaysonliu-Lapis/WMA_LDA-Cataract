# -*- coding: utf-8 -*-
"""
Step 1 of PROP real-data application.

For every right-eye image in a folder, run a pretrained ResNet-50 (ImageNet
weights, V2) with the final fully connected layer removed (global average
pooling kept), and dump the resulting 2048-dim feature vectors to CSV / NPY.

Default: regenerate the packaged features from `data/images/right/` into
`data/processed/features_resnet50.csv` and `features_resnet50.npy`, then verify
the row order against `data/processed/cataract_labels.csv`.

For your own eye images, point --images-dir at your folder and choose output
paths; the model is then trained later by the R pipeline on the packaged data.

Run from the repository root:
    python scripts/extract_features.py
    python scripts/extract_features.py --images-dir my_images --output-csv my_features.csv --output-npy my_features.npy
"""

import argparse
import os
import re
import time

# Anaconda + Intel MKL on Windows ships two OpenMP runtimes; allow both.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths (repository layout: images and data both live under data/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images", "right")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LABELS_CSV = os.path.join(OUTPUT_DIR, "cataract_labels.csv")
DEFAULT_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "features_resnet50.csv")
DEFAULT_OUTPUT_NPY = os.path.join(OUTPUT_DIR, "features_resnet50.npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model: ResNet-50 with the fc layer replaced by Identity, so forward(x)
# returns the 2048-d GAP feature directly.
# ---------------------------------------------------------------------------
def build_model(device: torch.device):
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    # transforms recommended for these weights: resize 232, center-crop 224,
    # ImageNet mean/std normalisation.
    return model, weights.transforms()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Matches the repository's current naming: Patient_1.jpg ... Patient_212.jpg
RIGHT_RE = re.compile(r"^Patient_(\d+)\.jpg$")


def collect_right_eye(folder: str):
    """Return [(patient_id, filename, full_path)] sorted by patient_id."""
    items = []
    for name in os.listdir(folder):
        m = RIGHT_RE.match(name)
        if m:
            items.append((int(m.group(1)), name, os.path.join(folder, name)))
    items.sort(key=lambda t: t[0])
    return items


class EyeImageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pid, _name, path = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), pid


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(items, model, transform, device, batch_size=64, num_workers=4):
    ds = EyeImageDataset(items, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )

    feats = np.zeros((len(items), 2048), dtype=np.float32)
    cursor = 0
    autocast_kwargs = dict(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else None
    for xb, _pid in tqdm(loader, desc="extracting", unit="batch"):
        xb = xb.to(device, non_blocking=True)
        if autocast_kwargs is not None:
            with torch.autocast(**autocast_kwargs):
                yb = model(xb)
            yb = yb.float()
        else:
            yb = model(xb)
        B = yb.size(0)
        feats[cursor : cursor + B] = yb.detach().cpu().numpy()
        cursor += B
    return feats


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def features_to_dataframe(feats, items):
    cols = [f"f{i + 1:04d}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, columns=cols)
    df.insert(0, "ID", [it[0] for it in items])
    df.insert(1, "filename", [it[1] for it in items])
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", default=IMAGES_DIR,
        help="Folder containing Patient_<number>.jpg images (default: data/images/right)",
    )
    parser.add_argument(
        "--output-csv", default=DEFAULT_OUTPUT_CSV,
        help="Output feature CSV (default: data/processed/features_resnet50.csv)",
    )
    parser.add_argument(
        "--output-npy", default=DEFAULT_OUTPUT_NPY,
        help="Output feature NPY (default: data/processed/features_resnet50.npy)",
    )
    parser.add_argument(
        "--no-check-labels", action="store_true",
        help="Skip the cataract_labels.csv row-order check",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device           = {device}")
    if device.type == "cuda":
        print(f"[info] cuda device name = {torch.cuda.get_device_name(0)}")

    model, transform = build_model(device)
    print("[info] model            = ResNet-50 (IMAGENET1K_V2, fc -> Identity)")

    if not os.path.isdir(args.images_dir):
        raise SystemExit(f"[error] images dir not found: {args.images_dir}")
    items = collect_right_eye(args.images_dir)
    if not items:
        raise SystemExit(
            f"[error] no Patient_<number>.jpg images found in {args.images_dir}"
        )
    print(f"[info] right-eye images: {len(items)}")

    t0 = time.time()
    print("[step] extracting features ...")
    feats = extract_features(items, model, transform, device)
    t1 = time.time()
    print(f"[time] extraction: {t1 - t0:.1f}s")

    # Write features (consumed by R)
    feat_df = features_to_dataframe(feats, items)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    feat_df.to_csv(args.output_csv, index=False, float_format="%.6f")
    np.save(args.output_npy, feats)
    print(f"[out ] {args.output_csv}  ({feat_df.shape[0]} rows, {feat_df.shape[1]} cols)")
    print(f"[out ] {args.output_npy}   shape={feats.shape}")

    # Consistency check against the packaged labels: only meaningful when
    # regenerating the packaged features in place.
    default_output = os.path.abspath(DEFAULT_OUTPUT_CSV)
    is_packaged_run = (
        os.path.abspath(args.images_dir) == os.path.abspath(IMAGES_DIR)
        and os.path.abspath(args.output_csv) == default_output
    )
    if not args.no_check_labels and is_packaged_run and os.path.exists(LABELS_CSV):
        lab = pd.read_csv(LABELS_CSV)
        ids_match = lab["ID"].astype(str).tolist() == feat_df["ID"].astype(str).tolist()
        names_match = lab["filename"].astype(str).tolist() == feat_df["filename"].astype(str).tolist()
        if ids_match and names_match:
            print(f"[ok  ] feature rows match cataract_labels.csv ({len(lab)} samples)")
        else:
            print("[warn] extracted feature order does not match cataract_labels.csv")
    elif not args.no_check_labels and not is_packaged_run:
        print("[info] custom images/output: skipped the packaged-labels check")


if __name__ == "__main__":
    main()
