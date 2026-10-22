#!/usr/bin/env python3
"""Evaluate conservative binary-mask fusion between GF-Screen and nnU-Net."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gf", type=Path, required=True)
    p.add_argument("--nnu", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--evaluator-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dsc-only", action="store_true", help="skip expensive surface-distance calculation")
    return p.parse_args()


def load_mask(path: Path):
    nii = nib.load(str(path))
    data = np.asanyarray(nii.dataobj)
    if not np.isin(np.unique(data), (0, 1)).all():
        raise ValueError(f"non-binary prediction: {path}")
    return nii, data.astype(bool, copy=False)


def filter_mm3(mask, voxel_mm3, threshold):
    if threshold <= 0 or not mask.any():
        return mask
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1) * voxel_mm3
    keep = sizes >= threshold
    keep[0] = False
    return keep[labels]


def guided_add(base, donor, dilation_iterations, min_overlap_fraction):
    """Add donor components supported by a spatially nearby base prediction."""
    support = ndimage.binary_dilation(base, iterations=dilation_iterations) if dilation_iterations else base
    labels, count = ndimage.label(donor)
    if count == 0:
        return base
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    overlaps = np.bincount(labels[support].ravel(), minlength=count + 1)
    fraction = overlaps / np.maximum(sizes, 1)
    keep = fraction >= min_overlap_fraction
    keep[0] = False
    return base | keep[labels]


def score_case_task(task):
    name, gf_dir, nnu_dir, gt_dir, evaluator_dir, dsc_only = task
    sys.path.insert(0, evaluator_dir)
    from evaluate import score_case

    gf_nii, gf_raw = load_mask(Path(gf_dir) / name)
    nnu_nii, nnu_raw = load_mask(Path(nnu_dir) / name)
    gt_nii = nib.load(str(Path(gt_dir) / name))
    gt = np.asanyarray(gt_nii.dataobj).astype(np.uint8, copy=False)
    if gf_nii.shape != nnu_nii.shape or gf_nii.shape != gt_nii.shape:
        raise ValueError(f"shape mismatch: {name}")
    if not np.allclose(gf_nii.affine, nnu_nii.affine) or not np.allclose(gf_nii.affine, gt_nii.affine):
        raise ValueError(f"affine mismatch: {name}")
    voxel_mm3 = float(abs(np.linalg.det(gf_nii.affine[:3, :3])))
    spacing = gt_nii.header.get_zooms()[:3]
    gf = filter_mm3(gf_raw, voxel_mm3, 500)
    nnu100 = filter_mm3(nnu_raw, voxel_mm3, 100)
    nnu500 = filter_mm3(nnu_raw, voxel_mm3, 500)

    masks = {
        "gf_mm500": gf,
        "nnu_raw": nnu_raw,
        "nnu_mm100": nnu100,
        "nnu_mm500": nnu500,
        "union_nnu_raw": gf | nnu_raw,
        "union_nnu_mm100": gf | nnu100,
        "union_nnu_mm500": gf | nnu500,
        "intersection": gf & nnu_raw,
        "rescue_gf_empty": nnu500 if not gf.any() else gf,
    }
    for dilation in (1, 2, 4):
        for overlap in (0.01, 0.10, 0.25):
            masks[f"gf_plus_nnu_d{dilation}_o{overlap:g}"] = guided_add(gf, nnu100, dilation, overlap)
            masks[f"nnu_plus_gf_d{dilation}_o{overlap:g}"] = guided_add(nnu500, gf, dilation, overlap)

    if dsc_only:
        gt_bool = gt.astype(bool, copy=False)
        scores = {}
        for key, mask in masks.items():
            denominator = int(gt_bool.sum()) + int(mask.sum())
            dsc = 1.0 if denominator == 0 else 2.0 * int((gt_bool & mask).sum()) / denominator
            scores[key] = (dsc, np.nan)
    else:
        scores = {
            key: score_case(gt, mask.astype(np.uint8, copy=False), spacing)
            for key, mask in masks.items()
        }
    volumes = {key: int(mask.sum()) for key, mask in masks.items()}
    return name, scores, volumes


def main():
    a = parse_args()
    names = sorted(p.name for p in a.gt.glob("*.nii.gz"))
    tasks = [(n, str(a.gf), str(a.nnu), str(a.gt), str(a.evaluator_dir), a.dsc_only) for n in names]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, result in enumerate(pool.map(score_case_task, tasks), 1):
            rows.append(result)
            print(f"[{i:03d}/{len(names):03d}] {result[0]}", flush=True)

    keys = list(rows[0][1])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as f:
        f.write("configuration\tDSC\tNSD\tzero_dsc\n")
        for key in keys:
            values = np.asarray([r[1][key] for r in rows])
            f.write(f"{key}\t{values[:, 0].mean():.4f}\t{values[:, 1].mean():.4f}\t{int((values[:, 0] == 0).sum())}\n")
    print(a.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
