#!/usr/bin/env python3
"""Render axial TP/FP/FN overlays for the lowest-Dice public cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cases-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=12)
    return parser.parse_args()


def component_slices(mask: np.ndarray, other: np.ndarray, limit: int) -> list[int]:
    coordinates = np.nonzero(mask)
    if not coordinates[0].size:
        return []
    extent = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates
    )
    mask_crop = mask[extent]
    other_crop = other[extent]
    component_map, count = ndimage.label(mask_crop, CONNECTIVITY_26)
    if not count:
        return []
    sizes = np.bincount(component_map.ravel())
    ids = sorted(range(1, len(sizes)), key=lambda idx: int(sizes[idx]), reverse=True)
    result: list[int] = []
    for component_id in ids:
        component = component_map == component_id
        # Prioritize components not explained by the other mask.
        if np.any(component & other_crop) and len(result) >= max(1, limit // 2):
            continue
        z_counts = component.sum(axis=(0, 1))
        result.append(int(np.argmax(z_counts)) + int(extent[2].start))
        if len(result) == limit:
            break
    return result


def image_path(images: Path, case_name: str) -> Path:
    case_id = case_name.removesuffix(".nii.gz")
    candidates = (images / f"{case_id}_0000.nii.gz", images / case_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.cases_csv.open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: float(row["dice"]))

    for row in rows[: args.num_cases]:
        case = row["case"]
        ct = np.asanyarray(nib.load(str(image_path(args.images, case))).dataobj)
        gt = np.asanyarray(nib.load(str(args.labels / case)).dataobj) > 0
        pred = np.asanyarray(nib.load(str(args.predictions / case)).dataobj) > 0
        if not (ct.shape == gt.shape == pred.shape):
            raise ValueError(f"Shape mismatch for {case}: {ct.shape}, {gt.shape}, {pred.shape}")

        slices = component_slices(gt, pred, 3) + component_slices(pred, gt, 3)
        slices = list(dict.fromkeys(slices))[:6]
        if not slices:
            slices = [ct.shape[2] // 2]

        figure, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="black")
        axes_flat = list(axes.ravel())
        for axis, z_index in zip(axes_flat, slices):
            ct_slice = np.clip((ct[:, :, z_index] + 175.0) / 425.0, 0.0, 1.0).T
            gt_slice = gt[:, :, z_index].T
            pred_slice = pred[:, :, z_index].T
            overlay = np.zeros((*gt_slice.shape, 4), dtype=np.float32)
            tp = gt_slice & pred_slice
            fp = ~gt_slice & pred_slice
            fn = gt_slice & ~pred_slice
            overlay[tp] = (1.0, 1.0, 0.0, 0.75)
            overlay[fp] = (1.0, 0.0, 0.0, 0.70)
            overlay[fn] = (0.0, 1.0, 0.0, 0.80)
            axis.imshow(ct_slice, cmap="gray", origin="lower", interpolation="nearest")
            axis.imshow(overlay, origin="lower", interpolation="nearest")
            axis.set_title(f"z={z_index}", color="white")
            axis.axis("off")
        for axis in axes_flat[len(slices) :]:
            axis.axis("off")

        figure.suptitle(
            f"{case}  DSC={float(row['dice']):.3f}  "
            f"precision={float(row['precision'] or 0):.3f}  recall={float(row['recall'] or 0):.3f}\n"
            "yellow=TP, red=FP, green=FN",
            color="white",
            fontsize=14,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure.savefig(args.output / f"{case.removesuffix('.nii.gz')}.png", dpi=130)
        plt.close(figure)
        print(case, flush=True)


if __name__ == "__main__":
    main()
