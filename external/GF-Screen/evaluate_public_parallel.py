#!/usr/bin/env python3
"""Parallel public scorer using the unchanged official per-case metric."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CODE = PROJECT_ROOT / "handoff_bundle" / "code"
sys.path.insert(0, str(OFFICIAL_CODE))
from fast_official_lesion_metrics import score_case  # noqa: E402


def score_paths(item: tuple[str, str, str]) -> tuple[str, float, float, int]:
    name, gt_path, pred_path = item
    if not Path(pred_path).exists():
        return name.removesuffix(".nii.gz"), 0.0, 0.0, -1
    gt_image = nib.load(gt_path)
    # ArrayProxy applies any NIfTI slope/intercept just as get_fdata does, but
    # avoids expanding ordinary uint8 masks to a full float64 volume only to
    # cast them straight back to uint8.
    gt = np.asanyarray(gt_image.dataobj).astype(np.uint8, copy=False)
    pred = np.asanyarray(nib.load(pred_path).dataobj).astype(np.uint8, copy=False)
    dsc, nsd = score_case(gt, pred, gt_image.header.get_zooms()[:3])
    return name.removesuffix(".nii.gz"), dsc, nsd, int((pred > 0).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--case-names",
        type=str,
        default=None,
        help="optional comma-separated basenames for a targeted public-label audit",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    names = sorted(path.name for path in args.gt.glob("*.nii.gz"))
    if not names:
        raise RuntimeError(f"No ground truth in {args.gt}")
    if args.case_names:
        requested = {
            value.strip().removesuffix("_0000.nii.gz").removesuffix(".nii.gz")
            for value in args.case_names.split(",")
            if value.strip()
        }
        available = {name.removesuffix(".nii.gz"): name for name in names}
        missing = sorted(requested - available.keys())
        if missing:
            raise FileNotFoundError(
                f"--case-names requested {len(missing)} absent labels; first={missing[0]}"
            )
        names = sorted(available[name] for name in requested)
        print(f"targeted scoring: retained {len(names)} requested cases", flush=True)
    items = [
        (name, str(args.gt / name), str(args.pred / name))
        for name in names
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(score_paths, items))

    for case, dsc, nsd, _ in rows:
        suffix = " MISSING PREDICTION" if dsc == 0 and not (args.pred / f"{case}.nii.gz").exists() else ""
        print(f"{case}.nii.gz  DSC {dsc:.4f}  NSD {nsd:.4f}{suffix}", flush=True)

    dsc_values = np.asarray([row[1] for row in rows])
    nsd_values = np.asarray([row[2] for row in rows])
    print(f"\n=============== LESION-BEARING: n={len(rows)} ===============")
    print(
        f"DSC  mean {dsc_values.mean():.4f}  std {dsc_values.std():.4f}  "
        f"median {np.median(dsc_values):.4f}"
    )
    print(
        f"NSD  mean {nsd_values.mean():.4f}  std {nsd_values.std():.4f}  "
        f"median {np.median(nsd_values):.4f}"
    )
    print(f"cases DSC=0 : {(dsc_values == 0).sum()}   DSC>0.2: {(dsc_values > 0.2).sum()}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD", "pred_fg_voxels"))
        for case, dsc, nsd, foreground in rows:
            # Preserve the official metric values for paired promotion tests.
            # Four-decimal serialization can move a 50-case mean by several
            # 1e-6 and makes an immutable champion anchor impossible to verify
            # against its unrounded aggregate score.
            writer.writerow((case, float(dsc), float(nsd), foreground))
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
