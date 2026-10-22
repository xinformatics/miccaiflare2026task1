#!/usr/bin/env python3
"""Decompose DSC loss into near-boundary and remote FN/FP volume."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


NIFTI_SUFFIX = ".nii.gz"


def tolerance_structure(spacing: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    radii = np.ceil(radius_mm / np.asarray(spacing, dtype=np.float64)).astype(int)
    grids = np.meshgrid(
        *(np.arange(-radius, radius + 1) for radius in radii), indexing="ij"
    )
    distance2 = sum(
        (grid.astype(np.float64) * spacing[axis]) ** 2
        for axis, grid in enumerate(grids)
    )
    return distance2 <= radius_mm**2 + 1e-12


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def analyze(predictions: Path, labels: Path, radius_mm: float) -> tuple[dict[str, object], list[dict[str, object]]]:
    label_paths = sorted(labels.glob(f"*{NIFTI_SUFFIX}"))
    if len(label_paths) != 50:
        raise ValueError(f"exact 50-case public inventory required, found {len(label_paths)}")
    rows: list[dict[str, object]] = []
    for label_path in label_paths:
        prediction_path = predictions / label_path.name
        if not prediction_path.is_file():
            raise ValueError(f"missing prediction: {prediction_path}")
        label_image = nib.load(label_path)
        prediction_image = nib.load(prediction_path)
        if label_image.shape != prediction_image.shape or not np.allclose(
            label_image.affine, prediction_image.affine, atol=1e-4, rtol=0
        ):
            raise ValueError(f"geometry mismatch: {label_path.name}")
        target = np.asarray(label_image.dataobj) > 0
        prediction = np.asarray(prediction_image.dataobj) > 0
        spacing = tuple(float(value) for value in label_image.header.get_zooms()[:3])
        structure = tolerance_structure(spacing, radius_mm)
        target_near = ndimage.binary_dilation(target, structure=structure)
        prediction_near = ndimage.binary_dilation(prediction, structure=structure)
        true_positive = target & prediction
        false_negative = target & ~prediction
        false_positive = prediction & ~target
        near_fn = false_negative & prediction_near
        near_fp = false_positive & target_near
        tp = int(true_positive.sum())
        fn = int(false_negative.sum())
        fp = int(false_positive.sum())
        target_voxels = int(target.sum())
        prediction_voxels = int(prediction.sum())
        rows.append(
            {
                "case": label_path.name[: -len(NIFTI_SUFFIX)],
                "target_voxels": target_voxels,
                "prediction_voxels": prediction_voxels,
                "volume_ratio": safe_ratio(prediction_voxels, target_voxels),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "near_fn": int(near_fn.sum()),
                "remote_fn": int(fn - near_fn.sum()),
                "near_fp": int(near_fp.sum()),
                "remote_fp": int(fp - near_fp.sum()),
                "dsc": safe_ratio(2 * tp, 2 * tp + fp + fn),
            }
        )
    totals = {
        key: int(sum(int(row[key]) for row in rows))
        for key in ("target_voxels", "prediction_voxels", "tp", "fn", "fp", "near_fn", "remote_fn", "near_fp", "remote_fp")
    }
    volume_ratios = np.asarray([float(row["volume_ratio"]) for row in rows])
    summary: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "diagnostic only; public labels are never training inputs",
        "cases": len(rows),
        "radius_mm": radius_mm,
        "totals": totals,
        "aggregate": {
            "volume_ratio": safe_ratio(totals["prediction_voxels"], totals["target_voxels"]),
            "dsc": safe_ratio(2 * totals["tp"], 2 * totals["tp"] + totals["fp"] + totals["fn"]),
            "fn_near_fraction": safe_ratio(totals["near_fn"], totals["fn"]),
            "fn_remote_fraction": safe_ratio(totals["remote_fn"], totals["fn"]),
            "fp_near_fraction": safe_ratio(totals["near_fp"], totals["fp"]),
            "fp_remote_fraction": safe_ratio(totals["remote_fp"], totals["fp"]),
        },
        "case_volume_ratio": {
            "median": float(np.median(volume_ratios)),
            "mean": float(np.mean(volume_ratios)),
            "under_0p8_cases": int(np.count_nonzero(volume_ratios < 0.8)),
            "over_1p2_cases": int(np.count_nonzero(volume_ratios > 1.2)),
        },
        "labels_read": True,
        "predictions_changed": False,
        "submission_created": False,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-mm", type=float, default=2.0)
    args = parser.parse_args()
    if args.radius_mm <= 0:
        raise ValueError("radius must be positive")
    summary, rows = analyze(args.predictions, args.labels, args.radius_mm)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "summary.json"
    temporary = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, json_path)
    with (args.output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
