#!/usr/bin/env python3
"""Audit whether pairwise ULS+ agreement can identify useful rescue masks."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--inspect-pair", nargs=2)
    args = parser.parse_args()

    records = json.loads((args.work / "manifest.json").read_text())
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(args.label))) > 0
    gt_flat = np.flatnonzero(gt.ravel())
    masks = []

    for record in records:
        crop = sitk.GetArrayFromImage(
            sitk.ReadImage(str(args.work / "predicted_crops" / f"{record['crop_id']}.nii.gz"))
        ) > 0
        x, y, z = map(int, record["start_xyz"])
        full = np.zeros(gt.shape, dtype=bool)
        z1, y1, x1 = max(z, 0), max(y, 0), max(x, 0)
        z2 = min(z + crop.shape[0], gt.shape[0])
        y2 = min(y + crop.shape[1], gt.shape[1])
        x2 = min(x + crop.shape[2], gt.shape[2])
        pz, py, px = z1 - z, y1 - y, x1 - x
        if z2 > z1 and y2 > y1 and x2 > x1:
            full[z1:z2, y1:y2, x1:x2] = crop[
                pz : pz + z2 - z1,
                py : py + y2 - y1,
                px : px + x2 - x1,
            ]
        indices = np.flatnonzero(full.ravel())
        overlap = np.intersect1d(indices, gt_flat, assume_unique=True).size
        dice = 2 * overlap / (indices.size + gt_flat.size)
        masks.append((record["crop_id"], indices, dice))

    pairs = []
    for (name_a, idx_a, dice_a), (name_b, idx_b, dice_b) in itertools.combinations(masks, 2):
        intersection = np.intersect1d(idx_a, idx_b, assume_unique=True)
        mutual = 2 * intersection.size / (idx_a.size + idx_b.size)
        gt_overlap = np.intersect1d(intersection, gt_flat, assume_unique=True).size
        consensus_gt = 2 * gt_overlap / (intersection.size + gt_flat.size)
        pairs.append((mutual, consensus_gt, name_a, name_b, intersection.size, dice_a, dice_b))

    def show(row):
        mutual, consensus_gt, name_a, name_b, count, dice_a, dice_b = row
        print(
            f"{name_a} {name_b} mutual={mutual:.6f} "
            f"consensus_gt={consensus_gt:.6f} intervox={count} "
            f"individual_gt={dice_a:.6f},{dice_b:.6f}"
        )

    print("TOP_MUTUAL")
    for row in sorted(pairs, reverse=True)[:5]:
        show(row)
    print("BEST_CONSENSUS_GT")
    show(max(pairs, key=lambda row: row[1]))

    if args.inspect_pair:
        target = set(args.inspect_pair)
        row = next(row for row in pairs if {row[2], row[3]} == target)
        print("INSPECT_PAIR")
        show(row)
        rank = 1 + sum(other[0] > row[0] for other in pairs)
        print(f"mutual_rank={rank}/{len(pairs)}")


if __name__ == "__main__":
    main()
