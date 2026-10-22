#!/usr/bin/env python3
"""Audit an independent teacher against the fixed public-50 GF anchor.

This is diagnostic-only. It never reads hidden data, creates submissions, or
changes masks. The lesion convention matches the 2026-08-27 Glance/Focus audit:
26-connectivity and a lesion detection threshold of 10% GT-volume overlap.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_mask(path: Path, reference: nib.Nifti1Image | None = None) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    unique = np.unique(data)
    if not np.isin(unique, [0, 1]).all():
        raise ValueError(f"mask is not binary: {path}, values={unique[:20].tolist()}")
    if reference is not None:
        if image.shape != reference.shape:
            raise ValueError(f"shape mismatch: {path}: {image.shape} != {reference.shape}")
        if not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
            raise ValueError(f"affine mismatch: {path}")
        if not np.allclose(image.header.get_zooms()[:3], reference.header.get_zooms()[:3], rtol=0, atol=1e-5):
            raise ValueError(f"spacing mismatch: {path}")
    return image, data.astype(bool, copy=False)


def overlap_fraction(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = int(reference.sum())
    return float(np.count_nonzero(reference & candidate) / denominator) if denominator else 0.0


def dice(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = int(reference.sum()) + int(candidate.sum())
    return float(2 * np.count_nonzero(reference & candidate) / denominator) if denominator else 1.0


def inventory(directory: Path) -> dict[str, Path]:
    return {path.name: path for path in sorted(directory.glob("*.nii.gz"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--prior-lesions", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    detection = float(policy["lesion_detection_overlap_fraction"])
    large_mm3 = float(policy["large_lesion_min_volume_mm3"])
    predictions = inventory(args.predictions)
    labels = inventory(args.labels)
    anchors = inventory(args.anchor)
    if not (len(predictions) == len(labels) == len(anchors) == 50):
        raise RuntimeError(
            f"fixed-public inventory must be 50/50/50, got "
            f"{len(predictions)}/{len(labels)}/{len(anchors)}"
        )
    if set(predictions) != set(labels) or set(anchors) != set(labels):
        raise RuntimeError("prediction, label, and anchor filenames do not match")

    with args.prior_lesions.open(newline="", encoding="utf-8") as stream:
        prior = list(csv.DictReader(stream))
    if len(prior) != 239:
        raise RuntimeError(f"expected 239 prior lesion records, found {len(prior)}")
    prior_by_key = {(row["case"], int(row["component_id"])): row for row in prior}

    lesion_rows: list[dict] = []
    novel_rows: list[dict] = []
    case_rows: list[dict] = []
    for name in sorted(labels):
        reference, target = load_mask(labels[name])
        _, teacher = load_mask(predictions[name], reference)
        _, anchor = load_mask(anchors[name], reference)
        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
        voxel_mm3 = float(np.prod(spacing))
        component_map, component_count = ndimage.label(target, structure=CONNECTIVITY_26)
        teacher_map, teacher_count = ndimage.label(teacher, structure=CONNECTIVITY_26)
        case_rows.append(
            {
                "case": name,
                "teacher_dsc": dice(target, teacher),
                "anchor_dsc": dice(target, anchor),
                "blind_union_dsc": dice(target, anchor | teacher),
                "teacher_voxels": int(teacher.sum()),
                "anchor_voxels": int(anchor.sum()),
                "target_voxels": int(target.sum()),
            }
        )
        gt_component_voxels = np.bincount(
            component_map.ravel(), minlength=component_count + 1
        )
        gt_component_anchor = np.bincount(
            component_map[anchor].ravel(), minlength=component_count + 1
        )
        gt_component_teacher = np.bincount(
            component_map[teacher].ravel(), minlength=component_count + 1
        )
        for component_id in range(1, component_count + 1):
            prior_row = prior_by_key.get((name, component_id))
            if prior_row is None:
                raise RuntimeError(f"prior lesion identity mismatch: {name} component {component_id}")
            voxels = int(gt_component_voxels[component_id])
            anchor_overlap = float(gt_component_anchor[component_id] / voxels)
            teacher_overlap = float(gt_component_teacher[component_id] / voxels)
            expected_anchor = float(prior_row["anchor_overlap"])
            if not np.isclose(anchor_overlap, expected_anchor, rtol=0, atol=1e-12):
                raise RuntimeError(
                    f"anchor audit mismatch for {name} component {component_id}: "
                    f"{anchor_overlap} != {expected_anchor}"
                )
            volume_mm3 = voxels * voxel_mm3
            lesion_rows.append(
                {
                    "case": name,
                    "component_id": component_id,
                    "volume_mm3": volume_mm3,
                    "anchor_overlap": anchor_overlap,
                    "teacher_overlap": teacher_overlap,
                    "anchor_detected": anchor_overlap >= detection,
                    "teacher_detected": teacher_overlap >= detection,
                    "large": volume_mm3 >= large_mm3,
                    "anchor_zero_overlap": anchor_overlap == 0.0,
                    "perfect_addition_macro_dsc_gain": float(
                        prior_row["perfect_addition_macro_dsc_gain"]
                    ),
                }
            )
        # A low-precision teacher may emit thousands of components. Compute
        # component sizes and overlaps in one O(voxels) pass rather than
        # materializing a full-volume Boolean mask once per component.
        component_voxels = np.bincount(
            teacher_map.ravel(), minlength=teacher_count + 1
        )
        component_anchor = np.bincount(
            teacher_map[anchor].ravel(), minlength=teacher_count + 1
        )
        component_target = np.bincount(
            teacher_map[target].ravel(), minlength=teacher_count + 1
        )
        for component_id in range(1, teacher_count + 1):
            voxels = int(component_voxels[component_id])
            anchor_overlap = float(component_anchor[component_id] / voxels)
            if anchor_overlap >= detection:
                continue
            gt_precision = float(component_target[component_id] / voxels)
            novel_rows.append(
                {
                    "case": name,
                    "component_id": component_id,
                    "volume_mm3": voxels * voxel_mm3,
                    "voxels": voxels,
                    "anchor_overlap": anchor_overlap,
                    "gt_precision": gt_precision,
                    "gt_confirmed": gt_precision >= detection,
                }
            )

    misses = [row for row in lesion_rows if not row["anchor_detected"]]
    large_misses = [row for row in misses if row["large"]]
    zero_large = [row for row in large_misses if row["anchor_zero_overlap"]]
    recovered_large = [row for row in large_misses if row["teacher_detected"]]
    recovered_zero = [row for row in zero_large if row["teacher_detected"]]
    novel_large = [row for row in novel_rows if row["volume_mm3"] >= 1000.0]
    teacher_dsc = float(np.mean([row["teacher_dsc"] for row in case_rows]))
    gate = policy["train_unlabel_inference_gate"]
    checks = {
        "standalone_public_dsc": teacher_dsc >= float(gate["minimum_standalone_public_dsc"]),
        "large_anchor_misses_recovered": len(recovered_large)
        >= int(gate["minimum_recovered_large_anchor_misses"]),
        "zero_overlap_large_anchor_misses_recovered": len(recovered_zero)
        >= int(gate["minimum_recovered_zero_overlap_large_anchor_misses"]),
    }
    gate_pass = all(checks.values())
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision_role": "PUBLIC_DIAGNOSTIC_ONLY",
        "cases": len(case_rows),
        "lesions": len(lesion_rows),
        "standalone_mean_dsc": teacher_dsc,
        "anchor_mean_dsc": float(np.mean([row["anchor_dsc"] for row in case_rows])),
        "blind_union_mean_dsc_diagnostic_only": float(
            np.mean([row["blind_union_dsc"] for row in case_rows])
        ),
        "anchor_misses": len(misses),
        "large_anchor_misses": len(large_misses),
        "large_anchor_misses_recovered": len(recovered_large),
        "zero_overlap_large_anchor_misses": len(zero_large),
        "zero_overlap_large_anchor_misses_recovered": len(recovered_zero),
        "recovered_large_lesions": recovered_large,
        "novel_teacher_components": len(novel_rows),
        "novel_teacher_components_ge1000mm3": len(novel_large),
        "novel_teacher_component_precision_ge1000mm3": (
            float(np.mean([row["gt_confirmed"] for row in novel_large])) if novel_large else None
        ),
        "gate_checks": checks,
        "train_unlabel_inference_gate_pass": gate_pass,
        "assets": {
            "policy": {"path": str(args.policy.resolve()), "sha256": sha256(args.policy)},
            "prior_lesions": {
                "path": str(args.prior_lesions.resolve()),
                "sha256": sha256(args.prior_lesions),
            },
            "predictions": str(args.predictions.resolve()),
            "labels": str(args.labels.resolve()),
            "anchor": str(args.anchor.resolve()),
        },
        "constraints": {
            "public_labels_read": True,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
            "blind_union_authorized": False,
            "retraining_authorized": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "lesions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(lesion_rows[0]))
        writer.writeheader()
        writer.writerows(lesion_rows)
    with (args.output / "novel_teacher_components.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(novel_rows[0]) if novel_rows else ["case", "component_id"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(novel_rows)
    with (args.output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    atomic_json(args.output / "summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
