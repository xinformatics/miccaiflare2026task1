#!/usr/bin/env python3
"""Audit whether historical 6-connectivity filtered fragmented large lesions."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from fuse_and_filter_masks import connected_components, filter_components


def load_binary(path: Path, reference: nib.Nifti1Image | None = None):
    image = nib.load(str(path))
    if reference is not None and (
        image.shape != reference.shape
        or not np.allclose(image.affine, reference.affine, atol=1e-3)
    ):
        raise ValueError(f"geometry mismatch: {path}")
    values = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(values), (0, 1)).all():
        raise ValueError(f"nonbinary mask: {path}")
    return image, values.astype(bool, copy=False)


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * int((prediction & target).sum()) / denominator


def audit(
    *,
    raw_dir: Path,
    historical_filtered_dir: Path,
    anchor_dir: Path,
    labels_dir: Path,
    minimum_mm3: float,
    large_mm3: float,
    detection_overlap: float,
    output: Path,
) -> dict:
    names = sorted(path.name for path in labels_dir.glob("*.nii.gz"))
    if len(names) != 50:
        raise ValueError(f"expected fixed public 50, found {len(names)}")
    expected = set(names)
    for directory in (raw_dir, historical_filtered_dir, anchor_dir):
        if {path.name for path in directory.glob("*.nii.gz")} != expected:
            raise ValueError(f"inventory mismatch: {directory}")

    rows: list[dict] = []
    dsc6: list[float] = []
    dsc26: list[float] = []
    changed_cases = 0
    added_voxels = 0
    for case_index, name in enumerate(names, 1):
        raw_image, raw = load_binary(raw_dir / name)
        _, historical = load_binary(historical_filtered_dir / name, raw_image)
        _, anchor = load_binary(anchor_dir / name, raw_image)
        _, target = load_binary(labels_dir / name, raw_image)
        voxel_mm3 = float(abs(np.linalg.det(raw_image.affine[:3, :3])))

        filtered6 = filter_components(raw, voxel_mm3, minimum_mm3, connectivity=6)
        if not np.array_equal(filtered6, historical):
            raise ValueError(f"historical 6-connectivity reproduction mismatch: {name}")
        filtered26 = filter_components(raw, voxel_mm3, minimum_mm3, connectivity=26)
        dsc6.append(dice(filtered6, target))
        dsc26.append(dice(filtered26, target))
        difference = filtered26 & ~filtered6
        changed_cases += int(difference.any())
        added_voxels += int(difference.sum())

        labels6, count6 = connected_components(raw, 6)
        sizes6_mm3 = np.bincount(labels6.ravel(), minlength=count6 + 1) * voxel_mm3
        target_labels, target_count = connected_components(target, 26)
        for lesion_id in range(1, target_count + 1):
            lesion = target_labels == lesion_id
            lesion_mm3 = float(lesion.sum() * voxel_mm3)
            if lesion_mm3 < large_mm3:
                continue
            anchor_overlap = float((anchor & lesion).sum() / lesion.sum())
            if anchor_overlap >= detection_overlap:
                continue
            coordinates = np.nonzero(lesion)
            region = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)
            bbox_ids = np.unique(labels6[region])
            bbox_ids = bbox_ids[bbox_ids > 0]
            overlap_ids = np.unique(labels6[lesion])
            overlap_ids = overlap_ids[overlap_ids > 0]
            bbox_volumes = sizes6_mm3[bbox_ids]
            overlap_volumes = sizes6_mm3[overlap_ids]
            overlap6 = float((filtered6 & lesion).sum() / lesion.sum())
            overlap26 = float((filtered26 & lesion).sum() / lesion.sum())
            rows.append(
                {
                    "case": name,
                    "lesion_id": lesion_id,
                    "lesion_mm3": lesion_mm3,
                    "anchor_overlap_fraction": anchor_overlap,
                    "raw_components_in_bbox": int(len(bbox_ids)),
                    "raw_bbox_component_sum_mm3": float(bbox_volumes.sum()),
                    "raw_bbox_largest_component_mm3": float(bbox_volumes.max(initial=0.0)),
                    "raw_components_touching_lesion": int(len(overlap_ids)),
                    "raw_touching_component_sum_mm3": float(overlap_volumes.sum()),
                    "filter6_overlap_fraction": overlap6,
                    "filter26_overlap_fraction": overlap26,
                    "bbox_fragmented_then_filtered_possible": bool(
                        len(bbox_ids) > 1
                        and bbox_volumes.max(initial=0.0) < minimum_mm3
                        and bbox_volumes.sum() >= minimum_mm3
                        and overlap6 < detection_overlap
                    ),
                    "connectivity26_strict_recovery": bool(
                        overlap6 < detection_overlap and overlap26 >= detection_overlap
                    ),
                }
            )
        print(f"[{case_index:02d}/50] {name}", flush=True)
        del raw, historical, anchor, target, filtered6, filtered26, labels6, target_labels
        gc.collect()

    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "role": "fixed_public_fragmentation_forensic_audit",
        "case_count": 50,
        "minimum_component_mm3": minimum_mm3,
        "large_lesion_mm3": large_mm3,
        "detection_overlap_fraction": detection_overlap,
        "historical_connectivity": 6,
        "candidate_connectivity": 26,
        "historical_reproduced_exactly": True,
        "mean_raw_filtered_DSC": {
            "connectivity6": float(np.mean(dsc6)),
            "connectivity26": float(np.mean(dsc26)),
            "gain": float(np.mean(np.asarray(dsc26) - np.asarray(dsc6))),
        },
        "changed_cases": changed_cases,
        "added_voxels": added_voxels,
        "missed_large_lesions": len(rows),
        "bbox_fragmentation_possible": sum(
            row["bbox_fragmented_then_filtered_possible"] for row in rows
        ),
        "connectivity26_strict_recoveries": sum(
            row["connectivity26_strict_recovery"] for row in rows
        ),
        "lesions": rows,
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--historical-filtered-dir", required=True, type=Path)
    parser.add_argument("--anchor-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--minimum-mm3", default=1000.0, type=float)
    parser.add_argument("--large-mm3", default=3000.0, type=float)
    parser.add_argument("--detection-overlap", default=0.1, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(**vars(args))
    print(json.dumps({key: result[key] for key in (
        "mean_raw_filtered_DSC", "changed_cases", "added_voxels",
        "missed_large_lesions", "bbox_fragmentation_possible",
        "connectivity26_strict_recoveries",
    )}, indent=2))


if __name__ == "__main__":
    main()
