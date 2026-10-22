#!/usr/bin/env python3
"""Screen component-level GF/nnU-Net confirmation without writing NIfTI files.

This is intentionally not a voxel-average ensemble.  GF-Screen remains the
segmenter.  An independently trained nnU-Net prediction is used only as
evidence that a GF connected component is supported.  Large GF components can
be retained without confirmation so that an nnU-Net miss does not erase an
obvious lesion.  The screen reports per-case and macro Dice; only the strongest
few configurations should subsequently be materialized for the official NSD
evaluator.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    import cc3d
except ImportError:  # pragma: no cover
    cc3d = None


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def parse_values(text, cast=float):
    return [cast(value) for value in text.split(",") if value.strip()]


def load_mask(path):
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(bool), image


def dice(prediction, target):
    intersection = np.count_nonzero(prediction & target)
    denominator = np.count_nonzero(prediction) + np.count_nonzero(target)
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def component_subset_dice(keep_ids, component_sizes, target_overlap, target_size):
    """Compute Dice from component sufficient statistics without a 3D mask."""
    prediction_size = int(component_sizes[keep_ids].sum())
    intersection = int(target_overlap[keep_ids].sum())
    denominator = prediction_size + target_size
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def connected_components(mask):
    if cc3d is not None:
        return cc3d.connected_components(mask, connectivity=26, return_N=True)
    return ndimage.label(mask, CONNECTIVITY_26)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gf", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distances-mm", default="0,2,5,10")
    parser.add_argument("--rescue-min-mm3", default="inf,1000,3000,10000")
    args = parser.parse_args()

    distances = parse_values(args.distances_mm)
    rescue_values = parse_values(args.rescue_min_mm3)
    configurations = [(distance, rescue) for distance in distances for rescue in rescue_values]
    scores = {configuration: [] for configuration in configurations}
    baseline_scores = []
    per_case = []

    gf_paths = sorted(args.gf.glob("*.nii.gz"))
    if not gf_paths:
        raise SystemExit(f"no predictions in {args.gf}")

    for index, gf_path in enumerate(gf_paths, 1):
        support_path = args.support / gf_path.name
        label_path = args.labels / gf_path.name
        if not support_path.is_file() or not label_path.is_file():
            raise SystemExit(f"missing aligned file for {gf_path.name}")

        gf, gf_image = load_mask(gf_path)
        support, support_image = load_mask(support_path)
        target, target_image = load_mask(label_path)
        if gf.shape != support.shape or gf.shape != target.shape:
            raise SystemExit(f"shape mismatch for {gf_path.name}")
        if not np.allclose(gf_image.affine, support_image.affine, atol=1e-3):
            raise SystemExit(f"support affine mismatch for {gf_path.name}")
        if not np.allclose(gf_image.affine, target_image.affine, atol=1e-3):
            raise SystemExit(f"label affine mismatch for {gf_path.name}")

        spacing = tuple(float(value) for value in gf_image.header.get_zooms()[:3])
        voxel_volume = float(np.prod(spacing))
        component_map, component_count = connected_components(gf)
        component_sizes = np.bincount(component_map.ravel(), minlength=component_count + 1)
        target_size = int(np.count_nonzero(target))
        # Lesions occupy a tiny fraction of a CT volume.  Counting component
        # IDs only at target voxels is exactly equivalent to a full-volume
        # float64 weighted bincount and avoids multi-gigabyte temporaries on
        # long thin-slice scans.
        target_overlap = np.bincount(
            component_map[target], minlength=component_count + 1
        )
        all_foreground_ids = np.ones(component_count + 1, dtype=bool)
        all_foreground_ids[0] = False
        baseline = component_subset_dice(
            all_foreground_ids, component_sizes, target_overlap, target_size
        )
        baseline_scores.append(baseline)

        # One exact physical-distance transform is much faster than applying
        # several large 3D morphology kernels.  Distances are only queried at
        # GF foreground voxels below, so the temporary float64 map is released
        # at the end of each case.
        if any(distance > 0 for distance in distances):
            support_distance_mm = ndimage.distance_transform_edt(~support, sampling=spacing)
            support_by_distance = {
                distance: (
                    support
                    if distance <= 0
                    else support_distance_mm <= distance + 1e-6
                )
                for distance in distances
            }
        else:
            support_by_distance = {distance: support for distance in distances}

        case_record = {"case": gf_path.name, "baseline_dice": baseline}
        for distance, rescue in configurations:
            supported_ids = np.unique(component_map[support_by_distance[distance] & gf])
            keep_ids = np.zeros(component_count + 1, dtype=bool)
            keep_ids[supported_ids] = True
            keep_ids[0] = False
            if math.isfinite(rescue):
                keep_ids |= component_sizes * voxel_volume >= rescue
                keep_ids[0] = False
            value = component_subset_dice(
                keep_ids, component_sizes, target_overlap, target_size
            )
            scores[(distance, rescue)].append(value)
            case_record[f"d{distance:g}_r{rescue:g}"] = value
        per_case.append(case_record)
        print(f"{index}/{len(gf_paths)} {gf_path.name} baseline={baseline:.4f}", flush=True)

    summary = []
    baseline_array = np.asarray(baseline_scores)
    for distance, rescue in configurations:
        values = np.asarray(scores[(distance, rescue)])
        differences = values - baseline_array
        summary.append({
            "distance_mm": distance,
            "rescue_min_mm3": rescue,
            "mean_dice": float(values.mean()),
            "mean_gain": float(differences.mean()),
            "median_gain": float(np.median(differences)),
            "win_rate": float(np.mean(differences > 0)),
            "loss_rate": float(np.mean(differences < 0)),
            "zero_dice_cases": int(np.count_nonzero(values == 0)),
        })
    summary.sort(key=lambda row: row["mean_dice"], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "gf": str(args.gf.resolve()),
        "support": str(args.support.resolve()),
        "labels": str(args.labels.resolve()),
        "cases": len(gf_paths),
        "baseline_mean_dice": float(baseline_array.mean()),
        "summary": summary,
    }, indent=2) + "\n")
    csv_path = args.output.with_suffix(".cases.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_case[0].keys())
        writer.writeheader()
        writer.writerows(per_case)
    print(json.dumps(summary[:8], indent=2))


if __name__ == "__main__":
    main()
