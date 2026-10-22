#!/usr/bin/env python3
"""Component-level error audit for binary lesion segmentations.

This complements case-level DSC/NSD with lesion detection and voxel error
statistics.  It also reports two diagnostic oracle Dice scores: one after
removing every false-positive voxel and one after adding every false-negative
voxel.  The oracles are not deployable; their gap identifies which error type
has the larger remaining headroom.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)
SIZE_BINS_MM3 = (
    ("lt100", 0.0, 100.0),
    ("100to500", 100.0, 500.0),
    ("500to1000", 500.0, 1000.0),
    ("1000to3000", 1000.0, 3000.0),
    ("ge3000", 3000.0, float("inf")),
)
STANDALONE_SKIP_SENTINEL = (
    Path(__file__).resolve().parent.parent
    / "BiomedParse"
    / "SKIP_REDUNDANT_STANDALONE_COMPONENT_AUDITS"
)
STANDALONE_AUDIT_PREFIXES = (
    "component_audit_whole_body_lesion_e",
    "component_audit_tumor_e",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--detection-recall",
        type=float,
        default=0.10,
        help="Minimum fraction of a GT component overlapped to count as detected.",
    )
    return parser.parse_args()


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def dice(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 1.0


def size_bin(volume_mm3: float) -> str:
    for name, lower, upper in SIZE_BINS_MM3:
        if lower <= volume_mm3 < upper:
            return name
    raise AssertionError(volume_mm3)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def foreground_slices(mask: np.ndarray) -> tuple[slice, ...] | None:
    """Return the tight foreground extent without labeling empty CT space."""
    coordinates = np.nonzero(mask)
    if not coordinates[0].size:
        return None
    return tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.detection_recall <= 1.0:
        raise ValueError("--detection-recall must be in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    if STANDALONE_SKIP_SENTINEL.exists() and args.output.name.startswith(
        STANDALONE_AUDIT_PREFIXES
    ):
        summary = {
            "status": "SKIPPED_REDUNDANT_STANDALONE_AUDIT",
            "reason": (
                "Standalone DSC/NSD is already scored; reserve the full "
                "component audit for the selected fusion candidate."
            ),
            "predictions": str(args.predictions.resolve()),
            "sentinel": str(STANDALONE_SKIP_SENTINEL),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return

    rows: list[dict[str, object]] = []
    gt_bin_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"components": 0, "detected": 0, "any_overlap": 0, "voxels": 0, "overlap_voxels": 0}
    )
    pred_bin_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"components": 0, "true_positive": 0, "voxels": 0, "overlap_voxels": 0}
    )

    label_paths = sorted(args.labels.glob("*.nii.gz"))
    if not label_paths:
        raise RuntimeError(f"No NIfTI labels found in {args.labels}")

    for label_path in label_paths:
        pred_path = args.predictions / label_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

        label_img = nib.load(str(label_path))
        pred_img = nib.load(str(pred_path))
        gt = np.asanyarray(label_img.dataobj) > 0
        pred = np.asanyarray(pred_img.dataobj) > 0
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {label_path.name}: {gt.shape} vs {pred.shape}")

        spacing = np.asarray(label_img.header.get_zooms()[:3], dtype=float)
        voxel_mm3 = float(np.prod(spacing))
        gt_voxels = int(np.count_nonzero(gt))
        pred_voxels = int(np.count_nonzero(pred))
        tp = int(np.count_nonzero(pred[gt]))
        fp = pred_voxels - tp
        fn = gt_voxels - tp

        gt_extent = foreground_slices(gt)
        if gt_extent is None:
            gt_crop = gt[tuple(slice(0, 0) for _ in gt.shape)]
            pred_at_gt = pred[tuple(slice(0, 0) for _ in pred.shape)]
        else:
            gt_crop = gt[gt_extent]
            pred_at_gt = pred[gt_extent]
        gt_map, gt_count = ndimage.label(gt_crop, CONNECTIVITY_26)
        gt_sizes = np.bincount(gt_map.ravel())
        gt_overlap = np.bincount(gt_map[pred_at_gt], minlength=len(gt_sizes))
        gt_detected = 0
        gt_any_overlap = 0
        for component_id in range(1, len(gt_sizes)):
            component_voxels = int(gt_sizes[component_id])
            overlap_voxels = int(gt_overlap[component_id])
            bin_name = size_bin(component_voxels * voxel_mm3)
            gt_bin_counts[bin_name]["components"] += 1
            gt_bin_counts[bin_name]["voxels"] += component_voxels
            gt_bin_counts[bin_name]["overlap_voxels"] += overlap_voxels
            if overlap_voxels > 0:
                gt_any_overlap += 1
                gt_bin_counts[bin_name]["any_overlap"] += 1
            if overlap_voxels / component_voxels >= args.detection_recall:
                gt_detected += 1
                gt_bin_counts[bin_name]["detected"] += 1

        pred_extent = foreground_slices(pred)
        if pred_extent is None:
            pred_crop = pred[tuple(slice(0, 0) for _ in pred.shape)]
            gt_at_pred = gt[tuple(slice(0, 0) for _ in gt.shape)]
        else:
            pred_crop = pred[pred_extent]
            gt_at_pred = gt[pred_extent]
        pred_map, pred_count = ndimage.label(pred_crop, CONNECTIVITY_26)
        pred_sizes = np.bincount(pred_map.ravel())
        pred_overlap = np.bincount(pred_map[gt_at_pred], minlength=len(pred_sizes))
        pred_true = 0
        for component_id in range(1, len(pred_sizes)):
            component_voxels = int(pred_sizes[component_id])
            overlap_voxels = int(pred_overlap[component_id])
            bin_name = size_bin(component_voxels * voxel_mm3)
            pred_bin_counts[bin_name]["components"] += 1
            pred_bin_counts[bin_name]["voxels"] += component_voxels
            pred_bin_counts[bin_name]["overlap_voxels"] += overlap_voxels
            if overlap_voxels > 0:
                pred_true += 1
                pred_bin_counts[bin_name]["true_positive"] += 1

        rows.append(
            {
                "case": label_path.name,
                "dice": dice(tp, fp, fn),
                "precision": safe_ratio(tp, tp + fp),
                "recall": safe_ratio(tp, tp + fn),
                "tp_voxels": tp,
                "fp_voxels": fp,
                "fn_voxels": fn,
                "fp_mm3": fp * voxel_mm3,
                "fn_mm3": fn * voxel_mm3,
                "gt_components": gt_count,
                "gt_components_any_overlap": gt_any_overlap,
                "gt_components_detected": gt_detected,
                "lesion_detection_recall": safe_ratio(gt_detected, gt_count),
                "pred_components": pred_count,
                "pred_components_overlapping_gt": pred_true,
                "component_precision": safe_ratio(pred_true, pred_count),
                "fp_free_oracle_dice": dice(tp, 0, fn),
                "fn_free_oracle_dice": dice(tp + fn, fp, 0),
            }
        )
        if len(rows) % 10 == 0 or len(rows) == len(label_paths):
            print(f"processed {len(rows)}/{len(label_paths)} cases", flush=True)

    fieldnames = list(rows[0])
    with (args.output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tp_total = sum(int(row["tp_voxels"]) for row in rows)
    fp_total = sum(int(row["fp_voxels"]) for row in rows)
    fn_total = sum(int(row["fn_voxels"]) for row in rows)
    gt_components_total = sum(int(row["gt_components"]) for row in rows)
    gt_detected_total = sum(int(row["gt_components_detected"]) for row in rows)
    pred_components_total = sum(int(row["pred_components"]) for row in rows)
    pred_true_total = sum(int(row["pred_components_overlapping_gt"]) for row in rows)

    gt_by_size = {}
    pred_by_size = {}
    for name, _, _ in SIZE_BINS_MM3:
        gt_stats = gt_bin_counts[name]
        pred_stats = pred_bin_counts[name]
        gt_by_size[name] = {
            **gt_stats,
            "detection_recall": safe_ratio(gt_stats["detected"], gt_stats["components"]),
            "any_overlap_rate": safe_ratio(gt_stats["any_overlap"], gt_stats["components"]),
            "voxel_recall": safe_ratio(gt_stats["overlap_voxels"], gt_stats["voxels"]),
        }
        pred_by_size[name] = {
            **pred_stats,
            "component_precision": safe_ratio(pred_stats["true_positive"], pred_stats["components"]),
            "voxel_precision": safe_ratio(pred_stats["overlap_voxels"], pred_stats["voxels"]),
        }

    summary = {
        "configuration": {
            "implementation": str(Path(__file__).resolve()),
            "implementation_sha256": file_sha256(Path(__file__).resolve()),
            "labels": str(args.labels.resolve()),
            "predictions": str(args.predictions.resolve()),
            "detection_recall": args.detection_recall,
            "connectivity": 26,
        },
        "cases": len(rows),
        "macro": {
            "dice": float(np.mean([float(row["dice"]) for row in rows])),
            "precision": float(np.mean([float(row["precision"] or 0.0) for row in rows])),
            "recall": float(np.mean([float(row["recall"] or 0.0) for row in rows])),
            "fp_free_oracle_dice": float(np.mean([float(row["fp_free_oracle_dice"]) for row in rows])),
            "fn_free_oracle_dice": float(np.mean([float(row["fn_free_oracle_dice"]) for row in rows])),
        },
        "micro": {
            "dice": dice(tp_total, fp_total, fn_total),
            "precision": safe_ratio(tp_total, tp_total + fp_total),
            "recall": safe_ratio(tp_total, tp_total + fn_total),
            "tp_voxels": tp_total,
            "fp_voxels": fp_total,
            "fn_voxels": fn_total,
        },
        "components": {
            "gt_total": gt_components_total,
            "gt_detected": gt_detected_total,
            "detection_recall": safe_ratio(gt_detected_total, gt_components_total),
            "pred_total": pred_components_total,
            "pred_overlapping_gt": pred_true_total,
            "precision": safe_ratio(pred_true_total, pred_components_total),
        },
        "gt_by_size_mm3": gt_by_size,
        "prediction_by_size_mm3": pred_by_size,
        "worst_dice_cases": [
            {key: row[key] for key in ("case", "dice", "precision", "recall", "fp_mm3", "fn_mm3")}
            for row in sorted(rows, key=lambda item: float(item["dice"]))[:15]
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
