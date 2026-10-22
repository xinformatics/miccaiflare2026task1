#!/usr/bin/env python3
"""Evaluate false positives on a known lesion-free image collection.

For an empty reference mask the challenge DSC and NSD are both one only when
the prediction is empty, and zero otherwise.  Labels therefore are not needed
for this audit; every nonzero predicted voxel is a false positive.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=172)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.pred.glob("*.nii.gz"))
    if len(paths) != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} predictions in {args.pred}, found {len(paths)}"
        )

    rows: list[dict[str, object]] = []
    for path in paths:
        image = nib.load(str(path))
        array = np.asanyarray(image.dataobj)
        values = np.unique(array)
        if not np.all(np.isin(values, (0, 1))):
            raise ValueError(f"{path.name} is not binary: {values[:10]}")
        voxels = int(np.count_nonzero(array))
        voxel_mm3 = float(np.prod(image.header.get_zooms()[:3]))
        rows.append(
            {
                "case": path.name,
                "pred_fg_voxels": voxels,
                "pred_fg_mm3": voxels * voxel_mm3,
                "empty": int(voxels == 0),
                "healthy_DSC": float(voxels == 0),
                "healthy_NSD": float(voxels == 0),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    false_positive_cases = sum(not bool(row["empty"]) for row in rows)
    summary = {
        "cases": len(rows),
        "empty_predictions": len(rows) - false_positive_cases,
        "false_positive_cases": false_positive_cases,
        "healthy_DSC": float(1.0 - false_positive_cases / len(rows)),
        "healthy_NSD": float(1.0 - false_positive_cases / len(rows)),
        "mean_false_positive_mm3": float(
            np.mean([float(row["pred_fg_mm3"]) for row in rows])
        ),
        "median_false_positive_mm3": float(
            np.median([float(row["pred_fg_mm3"]) for row in rows])
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
