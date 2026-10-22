#!/usr/bin/env python3
"""Evaluate physical connected-component thresholds without rewriting masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dsc-only", action="store_true")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0, 10, 25, 50, 100, 200, 500, 750, 1000, 1500, 2000,
                 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000],
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.evaluator_dir.resolve()))
    from evaluate import score_case  # pylint: disable=import-outside-toplevel

    thresholds = sorted(set(args.thresholds))
    aggregates = {threshold: [] for threshold in thresholds}
    case_rows = []
    names = sorted(path.name for path in args.gt.glob("*.nii.gz"))
    if not names:
        raise ValueError(f"no labels in {args.gt}")

    for index, name in enumerate(names, start=1):
        prediction_path = args.pred / name
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        prediction_image = nib.load(prediction_path)
        ground_truth_image = nib.load(args.gt / name)
        prediction = np.asanyarray(prediction_image.dataobj).astype(bool)
        ground_truth = np.asanyarray(ground_truth_image.dataobj).astype(np.uint8)
        spacing = ground_truth_image.header.get_zooms()[:3]
        voxel_volume = float(abs(np.linalg.det(prediction_image.affine[:3, :3])))
        components, count = ndimage.label(prediction)
        component_mm3 = np.bincount(components.ravel()) * voxel_volume

        for threshold in thresholds:
            if threshold <= 0:
                filtered = prediction
            else:
                keep = component_mm3 >= threshold
                keep[0] = False
                filtered = keep[components]
            if args.dsc_only:
                gt_sum = int(ground_truth.sum())
                pred_sum = int(filtered.sum())
                if gt_sum == 0 and pred_sum == 0:
                    dsc = 1.0
                elif gt_sum == 0 or pred_sum == 0:
                    dsc = 0.0
                else:
                    dsc = 2.0 * float((ground_truth.astype(bool) & filtered).sum()) / (gt_sum + pred_sum)
                nsd = float("nan")
            else:
                dsc, nsd = score_case(ground_truth, filtered, spacing)
            aggregates[threshold].append((dsc, nsd))
            case_rows.append({
                "Name": name.removesuffix(".nii.gz"),
                "min_mm3": threshold,
                "Lesion_DSC": dsc,
                "Lesion_NSD": nsd,
                "pred_fg_voxels": int(filtered.sum()),
            })
        print(f"[{index:03d}/{len(names):03d}] {name}: {count} components", flush=True)

    summary = []
    for threshold in thresholds:
        values = np.asarray(aggregates[threshold])
        row = {
            "min_mm3": threshold,
            "DSC": float(values[:, 0].mean()),
            "NSD": None if args.dsc_only else float(values[:, 1].mean()),
            "combined": None if args.dsc_only else float(values.mean()),
            "dsc_zero": int((values[:, 0] == 0).sum()),
            "dsc_above_0_2": int((values[:, 0] > 0.2).sum()),
        }
        summary.append(row)
        if args.dsc_only:
            print(f"{threshold:8g} mm3  DSC {row['DSC']:.4f}", flush=True)
        else:
            print(
                f"{threshold:8g} mm3  DSC {row['DSC']:.4f}  NSD {row['NSD']:.4f}  "
                f"combined {row['combined']:.4f}", flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_rows[0].keys())
        writer.writeheader()
        writer.writerows(case_rows)


if __name__ == "__main__":
    main()
