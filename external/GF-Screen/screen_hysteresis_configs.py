#!/usr/bin/env python3
"""Fast DSC-only screen for Focus-threshold hysteresis configurations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    import cc3d
except ImportError:  # pragma: no cover - compatibility fallback
    cc3d = None


def connected_components(mask: np.ndarray):
    """Exact 26-connected binary components with a fast installed backend."""
    if cc3d is not None:
        return cc3d.connected_components(mask, connectivity=26, return_N=True)
    return ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))


def dice(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    gt_sum = int(ground_truth.sum())
    pred_sum = int(prediction.sum())
    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0
    return 2.0 * float((ground_truth & prediction).sum()) / (gt_sum + pred_sum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.45, 0.5, 0.55, 0.6])
    parser.add_argument("--min-mm3", type=float, nargs="+", default=[0, 500, 1000, 2000, 5000])
    args = parser.parse_args()

    names = sorted(path.name for path in args.gt.glob("*.nii.gz"))
    scores = defaultdict(list)
    for index, name in enumerate(names, start=1):
        ground_truth = np.asanyarray(nib.load(args.gt / name).dataobj).astype(bool)
        images = {}
        masks = {}
        for threshold in args.thresholds:
            image = nib.load(args.root / f"lesion_{threshold:.2f}" / name)
            images[threshold] = image
            masks[threshold] = np.asanyarray(image.dataobj).astype(bool)
        voxel_volume = float(abs(np.linalg.det(images[args.thresholds[0]].affine[:3, :3])))
        for low in args.thresholds:
            components, count = connected_components(masks[low])
            sizes = np.bincount(components.ravel(), minlength=count + 1) * voxel_volume
            for high in args.thresholds:
                if high < low:
                    continue
                seed_ids = np.unique(components[masks[high]])
                seeded = np.zeros(count + 1, dtype=bool)
                seeded[seed_ids] = True
                seeded[0] = False
                for minimum in args.min_mm3:
                    keep = seeded & (sizes >= minimum)
                    prediction = keep[components]
                    scores[(low, high, minimum)].append(dice(ground_truth, prediction))
        print(f"[{index:03d}/{len(names):03d}] {name}", flush=True)

    summary = [
        {
            "low": low,
            "high": high,
            "min_mm3": minimum,
            "DSC": float(np.mean(values)),
            "DSC_std": float(np.std(values)),
            "DSC_zero": int((np.asarray(values) == 0).sum()),
        }
        for (low, high, minimum), values in scores.items()
    ]
    summary.sort(key=lambda row: row["DSC"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary[:15], indent=2))


if __name__ == "__main__":
    main()
