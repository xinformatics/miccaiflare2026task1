#!/usr/bin/env python3
"""Audit fixed-public lesion performance by TotalSegmentator anatomy.

This is a diagnostic only. It reads the fixed public images' precomputed
TotalSegmentator maps, public lesion labels, and an already-scored incumbent
case table. It neither trains a model nor writes predictions/submissions.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from totalsegmentator.map_to_binary import class_map


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_case(value: str) -> str:
    return Path(value).name.removesuffix(".nii.gz")


def organ_group(name: str) -> str:
    if name.startswith("lung_"):
        return "lung"
    if name.startswith("kidney_"):
        return "kidney"
    if name.startswith("adrenal_gland_"):
        return "adrenal_gland"
    if name.startswith("vertebrae_") or name in {"sacrum", "spinal_cord"}:
        return "spine"
    if name.startswith("rib_") or name in {"sternum", "costal_cartilages"}:
        return "chest_wall"
    if name.startswith("iliac_"):
        return "iliac_vessels"
    if name.startswith(("humerus_", "scapula_", "clavicula_", "femur_", "hip_")):
        return "bone"
    if name.startswith(("gluteus_", "autochthon_", "iliopsoas_")):
        return "muscle"
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("validation/Validation-Public-Labels"),
    )
    parser.add_argument(
        "--totalseg",
        type=Path,
        default=Path("external/GF-Screen/public_sweeps/totalseg_fast_public_20260822"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "external/STU-Net/pancancer/public_sweeps/"
            "stunetb_matched_e160_20260826/component_closed_case_metrics"
        ),
    )
    parser.add_argument("--configuration", default="gf_anchor_confirm_either_p70")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nnunet/analysis/incumbent_e160_public_organ_coverage_20260827"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_dir = args.labels.resolve()
    totalseg_dir = args.totalseg.resolve()
    metrics_path = args.metrics.resolve()
    output_dir = args.output.resolve()

    metric_files: list[Path]
    if metrics_path.is_dir():
        metric_files = sorted(metrics_path.glob("*.metrics.json"))
        metrics = {}
        for path in metric_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload["metrics"][args.configuration]
            metrics[canonical_case(payload["case"])] = {
                "dice": float(values[0]),
                "nsd": float(values[1]),
            }
    else:
        metric_files = [metrics_path]
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            metric_rows = list(csv.DictReader(stream))
        metrics = {canonical_case(row["case"]): row for row in metric_rows}
    if len(metrics) != 50:
        raise ValueError(f"expected 50 unique public cases, found {len(metrics)}")

    total_map = class_map["total"]
    rows: list[dict[str, object]] = []
    for case in sorted(metrics):
        label_path = labels_dir / f"{case}.nii.gz"
        anatomy_path = totalseg_dir / f"{case}.nii.gz"
        if not label_path.is_file() or not anatomy_path.is_file():
            raise FileNotFoundError(f"missing label/anatomy for {case}")

        gt_img = nib.load(str(label_path))
        anatomy_img = nib.load(str(anatomy_path))
        gt = np.asanyarray(gt_img.dataobj) > 0
        anatomy = np.asanyarray(anatomy_img.dataobj).astype(np.int16, copy=False)
        if gt.shape != anatomy.shape or not np.allclose(gt_img.affine, anatomy_img.affine):
            raise ValueError(f"geometry mismatch for {case}")
        gt_voxels = int(gt.sum())
        if gt_voxels <= 0:
            raise ValueError(f"empty public lesion label for {case}")

        labels, counts = np.unique(anatomy[gt], return_counts=True)
        label_counts = {int(label): int(count) for label, count in zip(labels, counts)}
        nonzero = [(label, count) for label, count in label_counts.items() if label != 0]
        dominant_label, dominant_count = max(nonzero, key=lambda item: item[1], default=(0, 0))
        dominant_name = total_map.get(dominant_label, "unassigned")
        metric = metrics[case]
        rows.append(
            {
                "case": case,
                "dice": float(metric["dice"]),
                "nsd": float(metric["nsd"]),
                "gt_voxels": gt_voxels,
                "dominant_label": dominant_label,
                "dominant_anatomy": dominant_name,
                "dominant_group": organ_group(dominant_name),
                "dominant_fraction_of_gt": dominant_count / gt_voxels,
                "background_fraction_of_gt": label_counts.get(0, 0) / gt_voxels,
                "anatomy_labels_in_gt": ";".join(
                    f"{label}:{total_map.get(label, 'unknown')}:{count}"
                    for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dominant_group"])].append(row)
    group_summary = {}
    for group, group_rows in sorted(grouped.items()):
        dice = np.asarray([float(row["dice"]) for row in group_rows])
        nsd = np.asarray([float(row["nsd"]) for row in group_rows])
        group_summary[group] = {
            "cases": len(group_rows),
            "case_ids": [str(row["case"]) for row in group_rows],
            "macro_dice": float(dice.mean()),
            "macro_nsd": float(nsd.mean()),
            "zero_dice_cases": int((dice == 0).sum()),
            "median_dominant_fraction_of_gt": float(
                np.median([float(row["dominant_fraction_of_gt"]) for row in group_rows])
            ),
        }

    zero_rows = [row for row in rows if float(row["dice"]) == 0]
    summary = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "purpose": "fixed-public incumbent lesion performance by precomputed TotalSegmentator anatomy",
        "case_count": len(rows),
        "macro_dice": float(np.mean([float(row["dice"]) for row in rows])),
        "macro_nsd": float(np.mean([float(row["nsd"]) for row in rows])),
        "zero_dice_cases": [
            {
                "case": row["case"],
                "dominant_anatomy": row["dominant_anatomy"],
                "dominant_fraction_of_gt": row["dominant_fraction_of_gt"],
            }
            for row in zero_rows
        ],
        "groups": group_summary,
        "inputs": {
            "metrics": {
                "path": str(metrics_path),
                "configuration": args.configuration,
                "file_count": len(metric_files),
                "manifest_sha256": hashlib.sha256(
                    "\n".join(f"{path.name} {sha256(path)}" for path in metric_files).encode()
                ).hexdigest(),
            },
            "labels": str(labels_dir),
            "totalseg": str(totalseg_dir),
        },
        "cases": {"path": str(cases_path), "sha256": sha256(cases_path)},
        "interpretation_guard": (
            "public-only diagnostic; anatomy is an independent pretrained segmentation and may be imperfect; "
            "not a case router, threshold selector, training target, or submission"
        ),
        "public_labels_read": True,
        "hidden_read": False,
        "training_mutated": False,
        "submission_created": False,
        "uploaded": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
