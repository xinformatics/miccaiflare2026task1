#!/usr/bin/env python3
"""Public-label diagnostic for automatic ULS+ rescue on missed cases.

The label-free selection rows are intentionally separated from label-informed
oracle rows. This program has no hidden-data or submission-packaging path.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def crop_flat_indices(
    crop: np.ndarray, full_shape: tuple[int, int, int], start_xyz: list[int]
) -> np.ndarray:
    """Map a crop mask to sorted, unique flat source-volume indices."""
    coordinates = np.argwhere(crop > 0)
    if coordinates.size == 0:
        return np.empty(0, dtype=np.int64)
    coordinates += np.asarray(tuple(reversed(start_xyz)), dtype=coordinates.dtype)
    shape = np.asarray(full_shape)
    keep = np.all((coordinates >= 0) & (coordinates < shape), axis=1)
    coordinates = coordinates[keep]
    return np.ravel_multi_index(coordinates.T, full_shape).astype(np.int64, copy=False)


def mask_stats(
    indices: np.ndarray, gt_indices: np.ndarray, target_voxels: int
) -> dict[str, float | int]:
    predicted = int(indices.size)
    overlap = int(np.intersect1d(indices, gt_indices, assume_unique=True).size)
    return {
        "voxels": predicted,
        "overlap_voxels": overlap,
        "dice": 2.0 * overlap / max(predicted + target_voxels, 1),
        "recall": overlap / max(target_voxels, 1),
        "precision": overlap / max(predicted, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads((args.work / "manifest.json").read_text())
    by_case: dict[str, list[dict]] = {}
    for record in records:
        by_case.setdefault(record["case"], []).append(record)

    report: dict[str, object] = {"selection_uses_labels": False, "cases": {}}
    for case, case_records in sorted(by_case.items()):
        label_img = sitk.ReadImage(str(args.labels / f"{case}.nii.gz"))
        gt = sitk.GetArrayFromImage(label_img) > 0
        gt_indices = np.flatnonzero(gt.ravel())
        target_voxels = int(gt_indices.size)
        proposal_img = sitk.ReadImage(case_records[0]["proposal"])
        proposal = sitk.GetArrayFromImage(proposal_img) > 0
        seed_labels, _ = ndimage.label(
            proposal, structure=ndimage.generate_binary_structure(3, 3)
        )
        seed_target_overlap = np.bincount(
            seed_labels[gt].ravel(), minlength=int(seed_labels.max()) + 1
        )

        candidates = []
        masks: dict[str, np.ndarray] = {}
        labels_flat = seed_labels.ravel()
        for record in case_records:
            crop_path = args.work / "predicted_crops" / f"{record['crop_id']}.nii.gz"
            crop = sitk.GetArrayFromImage(sitk.ReadImage(str(crop_path))) > 0
            refined = crop_flat_indices(crop, gt.shape, record["start_xyz"])
            label_id = int(record["label_id"])
            intersection = int(np.count_nonzero(labels_flat[refined] == label_id))
            refined_voxels = int(refined.size)
            seed_voxels = int(record["seed_voxels"])
            seed_gt_voxels = int(seed_target_overlap[label_id])
            features = {
                "crop_id": record["crop_id"],
                "rank": int(record["rank"]),
                "seed_voxels": seed_voxels,
                "seed_mm3": float(record["seed_mm3"]),
                "refined_voxels": refined_voxels,
                "seed_coverage": intersection / max(seed_voxels, 1),
                "refined_overlap": intersection / max(refined_voxels, 1),
                "volume_ratio": refined_voxels / max(seed_voxels, 1),
                "seed_gt_diagnostic": {
                    "overlap_voxels": seed_gt_voxels,
                    "dice": 2.0
                    * seed_gt_voxels
                    / max(seed_voxels + target_voxels, 1),
                    "recall": seed_gt_voxels / max(target_voxels, 1),
                    "precision": seed_gt_voxels / max(seed_voxels, 1),
                },
                "gt_diagnostic": mask_stats(refined, gt_indices, target_voxels),
            }
            candidates.append(features)
            masks[record["crop_id"]] = refined

        pairs = []
        for first, second in itertools.combinations(candidates, 2):
            a = masks[first["crop_id"]]
            b = masks[second["crop_id"]]
            intersection = np.intersect1d(a, b, assume_unique=True)
            mutual = 2.0 * int(intersection.size) / max(int(a.size + b.size), 1)
            pairs.append(
                {
                    "first": first["crop_id"],
                    "second": second["crop_id"],
                    "mutual_dice": mutual,
                    "min_seed_coverage": min(first["seed_coverage"], second["seed_coverage"]),
                    "min_refined_overlap": min(first["refined_overlap"], second["refined_overlap"]),
                    "max_volume_ratio": max(first["volume_ratio"], second["volume_ratio"]),
                    "intersection_gt_diagnostic": mask_stats(
                        intersection, gt_indices, target_voxels
                    ),
                }
            )

        top_mutual = max(pairs, key=lambda row: row["mutual_dice"])
        # A fixed, inference-visible confirmation score. It rewards independent
        # ULS agreement and requires both refined masks to contain their seeds.
        for pair in pairs:
            pair["confirmation_score"] = (
                pair["mutual_dice"]
                * pair["min_seed_coverage"]
                * min(1.0, 10.0 * pair["min_refined_overlap"])
            )
        top_confirmed = max(pairs, key=lambda row: row["confirmation_score"])
        best_individual = max(candidates, key=lambda row: row["gt_diagnostic"]["dice"])
        best_pair_oracle = max(
            pairs, key=lambda row: row["intersection_gt_diagnostic"]["dice"]
        )
        report["cases"][case] = {
            "target_voxels": target_voxels,
            "candidate_count": len(candidates),
            "label_free_top_mutual": top_mutual,
            "label_free_top_confirmed": top_confirmed,
            "label_informed_best_individual_oracle": best_individual,
            "label_informed_best_pair_oracle": best_pair_oracle,
            "candidates": candidates,
            "pairs": pairs,
        }
        print(
            f"{case}: n={len(candidates)} "
            f"top_mutual={top_mutual['intersection_gt_diagnostic']['dice']:.4f} "
            f"top_confirmed={top_confirmed['intersection_gt_diagnostic']['dice']:.4f} "
            f"individual_oracle={best_individual['gt_diagnostic']['dice']:.4f} "
            f"pair_oracle={best_pair_oracle['intersection_gt_diagnostic']['dice']:.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
