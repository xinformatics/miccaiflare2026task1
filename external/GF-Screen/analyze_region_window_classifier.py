#!/usr/bin/env python3
"""Audit whether raw CT appearance can safely select abdomen/chest windows.

The GF-Screen training loader uses [-175, 250] HU for the broad non-chest
cohort and [-900, 650] HU for the explicitly chest cohorts. Challenge inference
currently uses the abdomen window universally. This script extracts cheap,
label-free CT features, fits classifiers only on the training split, and reports
a conservative high-specificity chest decision on held-out internal cases plus
the public/hidden challenge distributions. It never reads lesion masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sample_evenly(values: list[dict], count: int, seed: int) -> list[dict]:
    if count >= len(values):
        return list(values)
    return random.Random(seed).sample(values, count)


def axis_stride(size: int, maximum: int = 72) -> int:
    return max(1, int(np.ceil(size / maximum)))


def ct_features(path: Path) -> dict[str, float]:
    image = nib.as_closest_canonical(nib.load(path))
    shape = image.shape[:3]
    sx, sy, sz = (axis_stride(size) for size in shape)
    # Regular strides are supported by every nibabel ArrayProxy and keep the
    # resident request bounded to approximately 72^3 voxels.
    values = np.asarray(image.dataobj[::sx, ::sy, ::sz], dtype=np.float32)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError(f"no finite CT voxels in {path}")

    # Exclude the outer field of view so scanner air does not masquerade as lung.
    x0, x1 = max(1, shape[0] // 6), max(2, shape[0] - shape[0] // 6)
    y0, y1 = max(1, shape[1] // 6), max(2, shape[1] - shape[1] // 6)
    central = np.asarray(image.dataobj[x0:x1:sx, y0:y1:sy, ::sz], dtype=np.float32)
    central = central[np.isfinite(central)]
    clipped = np.clip(values, -1200, 2000)
    central_clipped = np.clip(central, -1200, 2000)
    zooms = image.header.get_zooms()[:3]

    # Slice-wise low-HU occupancy distinguishes paired lungs from isolated gas.
    slab = np.asarray(image.dataobj[x0:x1:sx, y0:y1:sy, ::sz], dtype=np.float32)
    low_by_slice = np.mean(slab < -500, axis=(0, 1))
    result = {
        "shape_x": float(shape[0]),
        "shape_y": float(shape[1]),
        "shape_z": float(shape[2]),
        "extent_x_mm": float(shape[0] * zooms[0]),
        "extent_y_mm": float(shape[1] * zooms[1]),
        "extent_z_mm": float(shape[2] * zooms[2]),
        "central_frac_lt_m900": float(np.mean(central_clipped < -900)),
        "central_frac_lt_m700": float(np.mean(central_clipped < -700)),
        "central_frac_lt_m500": float(np.mean(central_clipped < -500)),
        "central_frac_tissue": float(np.mean((central_clipped >= -300) & (central_clipped <= 300))),
        "slice_low_mean": float(np.mean(low_by_slice)),
        "slice_low_q75": float(np.quantile(low_by_slice, 0.75)),
        "slice_low_q90": float(np.quantile(low_by_slice, 0.90)),
        "slice_low_max": float(np.max(low_by_slice)),
    }
    for quantile in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        result[f"hu_q{quantile:02d}"] = float(np.percentile(clipped, quantile))
    return result


def json_items(path: Path, split: str, region: str) -> list[dict]:
    items = json.loads(path.read_text())[split]
    return [
        {"name": item["name"], "path": Path(item["image"]), "region": region, "split": split}
        for item in items
    ]


def challenge_items(directory: Path, split: str) -> list[dict]:
    return [
        {"name": path.name.removesuffix("_0000.nii.gz"), "path": path, "region": "unknown", "split": split}
        for path in sorted(directory.glob("*_0000.nii.gz"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abdomen-json", type=Path, default=Path("jsons/local/abdomen.json"))
    parser.add_argument("--chest-json", type=Path, default=Path("jsons/local/chest.json"))
    parser.add_argument("--train-per-region", type=int, default=100)
    parser.add_argument("--public-images", type=Path, default=Path("../../validation/Validation-Public-Images"))
    parser.add_argument("--hidden-images", type=Path, default=Path("../../validation/Validation-Hidden-Images"))
    parser.add_argument("--output", type=Path, default=Path("analysis/region_window_classifier_20260820"))
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    abdomen_train = sample_evenly(
        json_items(args.abdomen_json, "training", "abdomen"), args.train_per_region, args.seed
    )
    chest_train = sample_evenly(
        json_items(args.chest_json, "training", "chest"), args.train_per_region, args.seed + 1
    )
    items = abdomen_train + chest_train
    items += json_items(args.abdomen_json, "validation", "abdomen")
    items += json_items(args.chest_json, "validation", "chest")
    items += challenge_items(args.public_images, "public")
    items += challenge_items(args.hidden_images, "hidden")

    rows = []
    for index, item in enumerate(items, 1):
        row = {key: str(value) if isinstance(value, Path) else value for key, value in item.items()}
        row.update(ct_features(item["path"]))
        rows.append(row)
        print(f"[{index:03d}/{len(items):03d}] {item['split']} {item['region']} {item['name']}", flush=True)

    feature_names = sorted(key for key in rows[0] if key not in {"name", "path", "region", "split"})
    train = [row for row in rows if row["split"] == "training"]
    validation = [row for row in rows if row["split"] == "validation"]
    x_train = np.asarray([[row[key] for key in feature_names] for row in train], dtype=np.float64)
    y_train = np.asarray([row["region"] == "chest" for row in train], dtype=np.int64)
    x_val = np.asarray([[row[key] for key in feature_names] for row in validation], dtype=np.float64)
    y_val = np.asarray([row["region"] == "chest" for row in validation], dtype=np.int64)

    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=2000)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=2.0,
            random_state=args.seed,
        ),
    }
    report = {"features": feature_names, "train_cases": len(train), "validation_cases": len(validation), "models": {}}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        train_probability = model.predict_proba(x_train)[:, 1]
        val_probability = model.predict_proba(x_val)[:, 1]
        # No sampled training abdomen may be routed to the chest window. This
        # deliberately favors specificity because a wrong chest window is a
        # catastrophic failure on abdominal scans.
        threshold = float(np.nextafter(train_probability[y_train == 0].max(), 1.0))
        val_prediction = val_probability >= threshold
        tn, fp, fn, tp = confusion_matrix(y_val, val_prediction, labels=[0, 1]).ravel()
        model_report = {
            "threshold_no_train_abdomen_false_positives": threshold,
            "train_auc": float(roc_auc_score(y_train, train_probability)),
            "validation_auc": float(roc_auc_score(y_val, val_probability)),
            "validation_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "validation_specificity": float(tn / (tn + fp)),
            "validation_sensitivity": float(tp / (tp + fn)),
        }
        for split in ("public", "hidden"):
            subset = [row for row in rows if row["split"] == split]
            x = np.asarray([[row[key] for key in feature_names] for row in subset], dtype=np.float64)
            probability = model.predict_proba(x)[:, 1]
            for row, value in zip(subset, probability):
                row[f"{model_name}_chest_probability"] = float(value)
            model_report[f"{split}_chest_routed"] = int(np.sum(probability >= threshold))
            model_report[f"{split}_probability_quantiles"] = {
                str(q): float(np.quantile(probability, q)) for q in (0, 0.5, 0.9, 0.95, 0.99, 1)
            }
        report["models"][model_name] = model_report

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "features.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
