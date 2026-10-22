#!/usr/bin/env python3
"""One-pass physical-margin audit for simulated partial lesion labels."""

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
    parser.add_argument("--contexts-mm", type=float, nargs="+", default=[8, 12, 16, 20, 24])
    parser.add_argument("--min-component-mm3", type=float, default=100.0)
    return parser.parse_args()


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contexts = sorted(set(float(value) for value in args.contexts_mm))
    totals = {
        context: {
            "box_voxels": 0,
            "primary_voxels": 0,
            "clean_background_inside_voxels": 0,
            "hidden_voxels": 0,
            "hidden_inside_voxels": 0,
            "teacher_hidden_inside_voxels": 0,
            "cases_with_hidden_inside": 0,
        }
        for context in contexts
    }
    rows: list[dict[str, object]] = []

    for label_path in sorted(args.labels.glob("*.nii.gz")):
        pred_path = args.predictions / label_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")
        label_img = nib.load(str(label_path))
        gt = np.asarray(label_img.dataobj, dtype=np.uint8) > 0
        pred = np.asarray(nib.load(str(pred_path)).dataobj, dtype=np.uint8) > 0
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {label_path.name}: {gt.shape} vs {pred.shape}")

        spacing = np.asarray(label_img.header.get_zooms()[:3], dtype=float)
        voxel_mm3 = float(np.prod(spacing))
        component_map, _ = ndimage.label(gt, CONNECTIVITY_26)
        component_sizes = np.bincount(component_map.ravel())
        valid_ids = [
            component_id
            for component_id in range(1, len(component_sizes))
            if component_sizes[component_id] * voxel_mm3 >= args.min_component_mm3
        ]
        if not valid_ids:
            continue
        primary_id = max(valid_ids, key=lambda component_id: component_sizes[component_id])
        hidden_ids = [component_id for component_id in valid_ids if component_id != primary_id]
        hidden_lookup = np.zeros(len(component_sizes), dtype=bool)
        hidden_lookup[hidden_ids] = True
        hidden = hidden_lookup[component_map]
        primary = component_map == primary_id
        primary_coordinates = np.where(primary)
        lower = np.asarray([int(axis.min()) for axis in primary_coordinates])
        upper = np.asarray([int(axis.max()) + 1 for axis in primary_coordinates])
        hidden_voxels = int(hidden.sum())

        for context in contexts:
            margin = np.ceil(context / spacing).astype(int)
            box_lower = np.maximum(0, lower - margin)
            box_upper = np.minimum(np.asarray(gt.shape), upper + margin)
            slices = tuple(slice(int(a), int(b)) for a, b in zip(box_lower, box_upper))
            box_gt = gt[slices]
            box_hidden = hidden[slices]
            box_pred = pred[slices]
            box_voxels = int(np.prod(box_upper - box_lower))
            hidden_inside = int(box_hidden.sum())
            teacher_hidden_inside = int((box_pred & box_hidden).sum())
            clean_background_inside = int((~box_gt).sum())

            aggregate = totals[context]
            aggregate["box_voxels"] += box_voxels
            aggregate["primary_voxels"] += int(primary.sum())
            aggregate["clean_background_inside_voxels"] += clean_background_inside
            aggregate["hidden_voxels"] += hidden_voxels
            aggregate["hidden_inside_voxels"] += hidden_inside
            aggregate["teacher_hidden_inside_voxels"] += teacher_hidden_inside
            aggregate["cases_with_hidden_inside"] += int(hidden_inside > 0)
            rows.append(
                {
                    "case": label_path.name,
                    "context_mm": context,
                    "voxel_mm3": voxel_mm3,
                    "box_voxels": box_voxels,
                    "primary_voxels": int(primary.sum()),
                    "hidden_voxels": hidden_voxels,
                    "hidden_inside_voxels": hidden_inside,
                    "clean_background_inside_voxels": clean_background_inside,
                    "teacher_hidden_inside_voxels": teacher_hidden_inside,
                }
            )

    with (args.output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for context in contexts:
        aggregate = totals[context]
        hidden_inside = aggregate["hidden_inside_voxels"]
        hidden_total = aggregate["hidden_voxels"]
        summaries.append(
            {
                "context_mm": context,
                **aggregate,
                "hidden_voxels_excluded": 1.0 - safe_ratio(hidden_inside, hidden_total),
                "hidden_voxel_leakage": safe_ratio(hidden_inside, hidden_total),
                "teacher_recall_on_leaked_hidden_voxels": safe_ratio(
                    aggregate["teacher_hidden_inside_voxels"], hidden_inside
                ),
                "clean_background_per_primary_voxel": safe_ratio(
                    aggregate["clean_background_inside_voxels"],
                    aggregate["primary_voxels"],
                ),
                "box_to_primary_volume_ratio": safe_ratio(
                    aggregate["box_voxels"], aggregate["primary_voxels"]
                ),
                "hidden_contamination_of_box": safe_ratio(
                    hidden_inside, aggregate["box_voxels"]
                ),
            }
        )
    output = {
        "configuration": {
            "labels": str(args.labels.resolve()),
            "predictions": str(args.predictions.resolve()),
            "contexts_mm": contexts,
            "min_component_mm3": args.min_component_mm3,
            "simulation": "largest >=threshold component annotated; remaining components hidden",
        },
        "cases": len({row["case"] for row in rows}),
        "margins": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
