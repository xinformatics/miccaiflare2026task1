#!/usr/bin/env python3
"""Screen recall-oriented component additions from a support model.

The immutable GF mask is never removed or altered.  Eligible connected
components from an independent support prediction are unioned either only when
fully disjoint from GF (new-lesion mode) or regardless of overlap (all mode).
The screen computes DSC from sufficient statistics and does not write masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np


def parse_values(text: str) -> list[float]:
    return [float(value) for value in text.split(",")]


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_sum = int(prediction.sum())
    target_sum = int(target.sum())
    if pred_sum + target_sum == 0:
        return 1.0
    return float(2 * np.count_nonzero(prediction & target) / (pred_sum + target_sum))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--min-mm3", default="0,100,500,1000,3000,10000")
    parser.add_argument("--modes", default="disjoint,all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    thresholds = parse_values(args.min_mm3)
    modes = [value.strip() for value in args.modes.split(",")]
    if any(mode not in {"disjoint", "all"} for mode in modes):
        raise ValueError(f"unsupported modes: {modes}")
    configs = [(mode, threshold) for mode in modes for threshold in thresholds]
    scores = {config: [] for config in configs}
    baseline_scores: list[float] = []
    case_names: list[str] = []

    label_paths = sorted(args.labels.glob("*.nii.gz"))
    if not label_paths:
        raise RuntimeError(f"no labels found in {args.labels}")
    for index, label_path in enumerate(label_paths, 1):
        base_path = args.base / label_path.name
        support_path = args.support / label_path.name
        if not base_path.exists() or not support_path.exists():
            raise FileNotFoundError(f"missing prediction for {label_path.name}")
        label_image = nib.load(str(label_path))
        base = np.asanyarray(nib.load(str(base_path)).dataobj) > 0
        support = np.asanyarray(nib.load(str(support_path)).dataobj) > 0
        target = np.asanyarray(label_image.dataobj) > 0
        if base.shape != target.shape or support.shape != target.shape:
            raise ValueError(f"shape mismatch for {label_path.name}")
        voxel_mm3 = float(np.prod(label_image.header.get_zooms()[:3]))
        component_map = cc3d.connected_components(support, connectivity=26)
        sizes = np.bincount(component_map.ravel())
        overlaps_base = np.bincount(component_map[base], minlength=len(sizes))
        baseline = dice(base, target)
        baseline_scores.append(baseline)
        case_names.append(label_path.name)

        for mode, threshold in configs:
            eligible = sizes * voxel_mm3 >= threshold
            eligible[0] = False
            if mode == "disjoint":
                eligible &= overlaps_base == 0
            additions = eligible[component_map]
            scores[(mode, threshold)].append(dice(base | additions, target))
        print(f"[{index:03d}/{len(label_paths):03d}] {label_path.name} baseline={baseline:.4f}", flush=True)

    summaries = []
    baseline_array = np.asarray(baseline_scores)
    for config in configs:
        values = np.asarray(scores[config])
        gains = values - baseline_array
        summaries.append(
            {
                "mode": config[0],
                "min_mm3": config[1],
                "mean_dice": float(values.mean()),
                "mean_gain": float(gains.mean()),
                "median_gain": float(np.median(gains)),
                "win_rate": float(np.mean(gains > 0)),
                "loss_rate": float(np.mean(gains < 0)),
                "zero_dice_cases": int(np.sum(values == 0)),
                "catastrophic_regression_rate": float(np.mean(gains < -0.10)),
            }
        )
    summaries.sort(key=lambda row: (row["mean_dice"], row["win_rate"]), reverse=True)
    output = {
        "base": str(args.base.resolve()),
        "support": str(args.support.resolve()),
        "labels": str(args.labels.resolve()),
        "cases": len(case_names),
        "baseline_mean_dice": float(baseline_array.mean()),
        "summary": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
