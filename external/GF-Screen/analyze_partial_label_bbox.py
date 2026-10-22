#!/usr/bin/env python3
"""Audit a partial-label bounding-box strategy without model training.

The public ground truth is treated as complete.  For each case, its largest
connected lesion is retained as the simulated annotated lesion and all other
components are hidden.  The script then measures how much hidden disease is
outside an automatically expanded target box and how much a frozen prediction
already recovers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-mm", type=float, default=16.0)
    parser.add_argument("--min-component-mm3", type=float, default=100.0)
    parser.add_argument(
        "--component-detection-recall",
        type=float,
        default=0.10,
        help="Fraction of a hidden component that prediction must overlap.",
    )
    return parser.parse_args()


def expanded_box(mask: np.ndarray, spacing: np.ndarray, context_mm: float) -> np.ndarray:
    coords = np.where(mask)
    lo = np.array([int(axis.min()) for axis in coords])
    hi = np.array([int(axis.max()) + 1 for axis in coords])
    margin = np.ceil(context_mm / spacing).astype(int)
    lo = np.maximum(0, lo - margin)
    hi = np.minimum(np.asarray(mask.shape), hi + margin)
    box = np.zeros(mask.shape, dtype=bool)
    box[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))] = True
    return box


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for label_path in sorted(args.labels.glob("*.nii.gz")):
        pred_path = args.predictions / label_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

        label_img = nib.load(str(label_path))
        pred_img = nib.load(str(pred_path))
        gt = np.asanyarray(label_img.dataobj) > 0
        pred = np.asanyarray(pred_img.dataobj) > 0
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {label_path.name}: {gt.shape} vs {pred.shape}")

        spacing = np.asarray(label_img.header.get_zooms()[:3], dtype=float)
        voxel_mm3 = float(np.prod(spacing))
        component_map, component_count_all = ndimage.label(gt, CONNECTIVITY_26)
        sizes_vox = np.bincount(component_map.ravel())
        valid_ids = [
            idx
            for idx in range(1, len(sizes_vox))
            if sizes_vox[idx] * voxel_mm3 >= args.min_component_mm3
        ]
        if not valid_ids:
            rows.append(
                {
                    "case": label_path.name,
                    "spacing_mm": "x".join(f"{v:.3f}" for v in spacing),
                    "components_all": component_count_all,
                    "components_included": 0,
                    "status": "no_component_above_threshold",
                }
            )
            continue

        primary_id = max(valid_ids, key=lambda idx: sizes_vox[idx])
        hidden_ids = [idx for idx in valid_ids if idx != primary_id]
        primary = component_map == primary_id
        hidden = np.isin(component_map, hidden_ids)
        box = expanded_box(primary, spacing, args.context_mm)

        hidden_voxels = int(hidden.sum())
        hidden_inside = int((hidden & box).sum())
        hidden_outside = hidden_voxels - hidden_inside
        pred_hidden = int((pred & hidden).sum())
        pred_hidden_inside = int((pred & hidden & box).sum())
        pred_hidden_outside = pred_hidden - pred_hidden_inside

        # Count prediction overlap for every component in one volume pass.  A
        # component-at-a-time boolean scan is prohibitively expensive when a
        # mask contains many tiny disconnected islands.
        overlap_by_id = np.bincount(
            component_map[pred], minlength=len(sizes_vox)
        )
        hidden_recalls = [
            float(overlap_by_id[component_id] / sizes_vox[component_id])
            for component_id in hidden_ids
        ]

        rows.append(
            {
                "case": label_path.name,
                "spacing_mm": "x".join(f"{v:.3f}" for v in spacing),
                "components_all": component_count_all,
                "components_included": len(valid_ids),
                "hidden_components": len(hidden_ids),
                "primary_mm3": float(sizes_vox[primary_id] * voxel_mm3),
                "hidden_mm3": float(hidden_voxels * voxel_mm3),
                "hidden_voxels": hidden_voxels,
                "hidden_inside_box_voxels": hidden_inside,
                "hidden_outside_box_voxels": hidden_outside,
                "pred_hidden_overlap_voxels": pred_hidden,
                "pred_hidden_inside_box_voxels": pred_hidden_inside,
                "pred_hidden_outside_box_voxels": pred_hidden_outside,
                "hidden_components_any_overlap": sum(v > 0 for v in hidden_recalls),
                "hidden_components_detected": sum(
                    v >= args.component_detection_recall for v in hidden_recalls
                ),
                "status": "ok",
            }
        )

    if not rows:
        raise RuntimeError(f"No NIfTI labels found in {args.labels}")

    fieldnames = sorted({key for row in rows for key in row})
    with (args.output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = [row for row in rows if row["status"] == "ok"]
    totals = {
        key: int(sum(int(row.get(key, 0)) for row in ok))
        for key in (
            "hidden_components",
            "hidden_voxels",
            "hidden_inside_box_voxels",
            "hidden_outside_box_voxels",
            "pred_hidden_overlap_voxels",
            "pred_hidden_inside_box_voxels",
            "pred_hidden_outside_box_voxels",
            "hidden_components_any_overlap",
            "hidden_components_detected",
        )
    }
    multi = [row for row in ok if int(row["hidden_components"]) > 0]
    summary = {
        "configuration": {
            "labels": str(args.labels.resolve()),
            "predictions": str(args.predictions.resolve()),
            "context_mm": args.context_mm,
            "min_component_mm3": args.min_component_mm3,
            "component_detection_recall": args.component_detection_recall,
            "simulation": "largest included component is annotated; all others are hidden",
        },
        "cases_total": len(rows),
        "cases_analyzed": len(ok),
        "cases_with_hidden_components": len(multi),
        "totals": totals,
        "metrics": {
            "hidden_voxels_excluded_by_box": ratio(
                totals["hidden_outside_box_voxels"], totals["hidden_voxels"]
            ),
            "hidden_voxels_remaining_inside_box": ratio(
                totals["hidden_inside_box_voxels"], totals["hidden_voxels"]
            ),
            "frozen_prediction_hidden_voxel_recall": ratio(
                totals["pred_hidden_overlap_voxels"], totals["hidden_voxels"]
            ),
            "predicted_true_hidden_voxels_excluded_by_box": ratio(
                totals["pred_hidden_outside_box_voxels"],
                totals["pred_hidden_overlap_voxels"],
            ),
            "hidden_component_any_overlap_rate": ratio(
                totals["hidden_components_any_overlap"], totals["hidden_components"]
            ),
            "hidden_component_detection_rate": ratio(
                totals["hidden_components_detected"], totals["hidden_components"]
            ),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
