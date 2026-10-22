#!/usr/bin/env python3
"""Materialize label-free ULS+ pair-consensus rescue masks.

Multiple independent proposal clicks that produce nearly identical ULS+ masks
provide an inference-visible confirmation signal. This program intersects the
top agreeing pairs and optionally adds them to an immutable base prediction.
It never reads ground truth.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def crop_indices_and_bbox(
    crop: np.ndarray, full_shape: tuple[int, int, int], start_xyz: list[int]
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    coordinates = np.argwhere(crop > 0)
    if coordinates.size == 0:
        return np.empty(0, dtype=np.int64), None
    coordinates += np.asarray(tuple(reversed(start_xyz)), dtype=coordinates.dtype)
    shape = np.asarray(full_shape)
    keep = np.all((coordinates >= 0) & (coordinates < shape), axis=1)
    coordinates = coordinates[keep]
    if coordinates.size == 0:
        return np.empty(0, dtype=np.int64), None
    indices = np.ravel_multi_index(coordinates.T, full_shape).astype(np.int64, copy=False)
    return indices, (coordinates.min(axis=0), coordinates.max(axis=0) + 1)


def boxes_overlap(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> bool:
    return bool(np.all(first[0] < second[1]) and np.all(second[0] < first[1]))


def same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), atol=1e-4, rtol=0)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), atol=1e-4, rtol=0)
        and np.allclose(first.GetDirection(), second.GetDirection(), atol=1e-4, rtol=0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--min-mutual-dice", type=float, required=True)
    parser.add_argument("--min-pair-mm3", type=float, default=0.0)
    parser.add_argument("--top-pairs-per-case", type=int, default=1)
    args = parser.parse_args()
    if not 0.0 <= args.min_mutual_dice <= 1.0:
        raise ValueError("--min-mutual-dice must be in [0,1]")
    if args.top_pairs_per_case < 1:
        raise ValueError("--top-pairs-per-case must be positive")

    records = json.loads((args.work / "manifest.json").read_text())
    by_case: dict[str, list[dict]] = {}
    for record in records:
        by_case.setdefault(record["case"], []).append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for case, case_records in sorted(by_case.items()):
        proposal_img = sitk.ReadImage(case_records[0]["proposal"])
        shape = tuple(reversed(proposal_img.GetSize()))
        voxel_mm3 = float(np.prod(proposal_img.GetSpacing()))
        masks = []
        for record in case_records:
            path = args.work / "predicted_crops" / f"{record['crop_id']}.nii.gz"
            crop = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
            indices, bbox = crop_indices_and_bbox(crop, shape, record["start_xyz"])
            if indices.size and bbox is not None:
                masks.append((record["crop_id"], indices, bbox))

        pairs = []
        for (name_a, idx_a, bbox_a), (name_b, idx_b, bbox_b) in itertools.combinations(masks, 2):
            if not boxes_overlap(bbox_a, bbox_b):
                continue
            intersection = np.intersect1d(idx_a, idx_b, assume_unique=True)
            if intersection.size == 0:
                continue
            mutual = 2.0 * intersection.size / (idx_a.size + idx_b.size)
            pair_mm3 = float(intersection.size * voxel_mm3)
            if mutual >= args.min_mutual_dice and pair_mm3 >= args.min_pair_mm3:
                pairs.append((mutual, pair_mm3, name_a, name_b, intersection))
        pairs.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
        selected = pairs[: args.top_pairs_per_case]

        rescue = np.zeros(shape, dtype=bool)
        rescue_flat = rescue.ravel()
        for mutual, pair_mm3, name_a, name_b, intersection in selected:
            rescue_flat[intersection] = True
            rows.append(
                {
                    "case": case,
                    "first": name_a,
                    "second": name_b,
                    "mutual_dice": mutual,
                    "intersection_voxels": int(intersection.size),
                    "intersection_mm3": pair_mm3,
                }
            )

        if args.base:
            base_img = sitk.ReadImage(str(args.base / f"{case}.nii.gz"))
            if not same_geometry(proposal_img, base_img):
                raise RuntimeError(f"base geometry mismatch for {case}")
            output = (sitk.GetArrayFromImage(base_img) > 0) | rescue
        else:
            output = rescue
        output_img = sitk.GetImageFromArray(output.astype(np.uint8))
        output_img.CopyInformation(proposal_img)
        sitk.WriteImage(output_img, str(args.output / f"{case}.nii.gz"), True)
        print(
            f"{case}: masks={len(masks)} qualifying_pairs={len(pairs)} "
            f"selected={len(selected)} rescue_voxels={int(rescue.sum())}",
            flush=True,
        )

    with (args.output.parent / f"{args.output.name}.selection.csv").open(
        "w", newline=""
    ) as handle:
        fields = [
            "case",
            "first",
            "second",
            "mutual_dice",
            "intersection_voxels",
            "intersection_mm3",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
