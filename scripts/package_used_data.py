#!/usr/bin/env python
"""Package the exact ResNet-50 cataract cohort used by this repository.

This script is intentionally narrow: it packages the 212 right-eye images,
the original patient fields needed to derive the label, and the cached feature
matrices that were used in the reported PROP experiment.  It does not change
the study definition or re-extract image features.

Example
-------
python scripts/package_used_data.py --source-root D:/eye-project
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ORIGINAL_COLUMNS = [
    "ID", "Patient Age", "Patient Sex", "Left-Fundus", "Right-Fundus",
    "Left-Diagnostic Keywords", "Right-Diagnostic Keywords",
    "N", "D", "G", "C", "A", "H", "M", "O",
]
MODEL_LABEL_COLUMNS = [
    "ID", "filename", "Age", "Sex", "N", "D", "G", "C", "A", "H",
    "M", "O", "Y_cataract",
]
OTHER_DISEASE_COLUMNS = ["D", "G", "A", "H", "M", "O"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", required=True, type=Path,
        help="Existing project root containing data/ and application/.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to package into (default: this repository).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    processed_dir = output_root / "data" / "processed"
    image_dir = output_root / "data" / "images" / "right"
    processed_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = source_root / "data" / "data.xlsx"
    feature_csv = source_root / "application" / "features_cataract.csv"
    feature_npy = source_root / "application" / "features_cataract.npy"
    image_source = source_root / "data" / "Training Images"
    for path in (xlsx_path, feature_csv, feature_npy, image_source):
        if not path.exists():
            raise FileNotFoundError(path)

    patient = pd.read_excel(xlsx_path)
    missing = set(ORIGINAL_COLUMNS).difference(patient.columns)
    if missing:
        raise ValueError(f"data.xlsx is missing columns: {sorted(missing)}")
    patient = patient.loc[:, ORIGINAL_COLUMNS].copy()
    patient["Y_cataract"] = pd.NA
    c1 = patient["C"].eq(1)
    has_other = patient[OTHER_DISEASE_COLUMNS].sum(axis=1).ge(1)
    patient.loc[c1 & ~has_other, "Y_cataract"] = 0
    patient.loc[c1 & has_other, "Y_cataract"] = 1

    features = pd.read_csv(feature_csv)
    feature_columns = [name for name in features.columns if name.startswith("f") and name[1:].isdigit()]
    if features.columns[:2].tolist() != ["ID", "filename"] or len(feature_columns) != 2048:
        raise ValueError("features_cataract.csv must contain ID, filename, and 2048 feature columns")
    if features["ID"].duplicated().any():
        raise ValueError("features_cataract.csv contains duplicated IDs")

    selected = patient.set_index("ID").loc[features["ID"].to_numpy()].reset_index()
    if selected["Y_cataract"].isna().any() or not selected["C"].eq(1).all():
        raise ValueError("Feature rows do not exactly match the cataract cohort")
    if not selected["Right-Fundus"].astype(str).equals(features["filename"].astype(str)):
        raise ValueError("Right-eye image names do not match the feature matrix")
    selected["Y_cataract"] = selected["Y_cataract"].astype("int8")
    selected["right_eye_image"] = features["filename"].astype(str).to_numpy()

    labels = pd.DataFrame({
        "ID": selected["ID"].astype(int),
        "filename": selected["right_eye_image"],
        "Age": selected["Patient Age"].astype(int),
        "Sex": selected["Patient Sex"].astype(str),
    })
    for column in ["N", "D", "G", "C", "A", "H", "M", "O"]:
        labels[column] = selected[column].astype(int)
    labels["Y_cataract"] = selected["Y_cataract"].astype(int)
    labels = labels.loc[:, MODEL_LABEL_COLUMNS]

    metadata_path = processed_dir / "cataract_metadata.csv"
    labels_path = processed_dir / "cataract_labels.csv"
    features_path = processed_dir / "features_resnet50.csv"
    npy_path = processed_dir / "features_resnet50.npy"
    selected.to_csv(metadata_path, index=False, encoding="utf-8", float_format="%.6f")
    labels.to_csv(labels_path, index=False, encoding="utf-8")
    shutil.copy2(feature_csv, features_path)
    shutil.copy2(feature_npy, npy_path)

    matrix = np.load(npy_path)
    if matrix.shape != (len(labels), 2048) or matrix.dtype != np.float32:
        raise ValueError(f"Unexpected NPY feature matrix: shape={matrix.shape}, dtype={matrix.dtype}")

    copied = []
    for filename in labels["filename"]:
        src = image_source / filename
        dst = image_dir / filename
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        copied.append(dst)

    manifest = {
        "dataset": "ODIR-5K cataract patient-level cohort",
        "n_samples": int(len(labels)),
        "class_counts": {str(key): int(value) for key, value in labels["Y_cataract"].value_counts().sort_index().items()},
        "image_side": "right",
        "feature_extractor": "torchvision ResNet-50, IMAGENET1K_V2, FC replaced by Identity",
        "feature_dimension": 2048,
        "source_files": {
            "patient_metadata": "data/data.xlsx",
            "features_csv": "application/features_cataract.csv",
            "features_npy": "application/features_cataract.npy",
            "images": "data/Training Images/*_right.jpg",
        },
    }
    (processed_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    tracked = [metadata_path, labels_path, features_path, npy_path, processed_dir / "dataset_manifest.json", *copied]
    checksum_path = processed_dir / "SHA256SUMS"
    with checksum_path.open("w", encoding="utf-8", newline="\n") as handle:
        for path in sorted(tracked):
            handle.write(f"{sha256(path)}  {path.relative_to(output_root).as_posix()}\n")

    print(f"Packaged {len(labels)} samples to {output_root}")
    print(f"Class counts: {manifest['class_counts']}")
    print(f"Copied right-eye images: {len(copied)}")


if __name__ == "__main__":
    main()
