#!/usr/bin/env python3
"""Simulate partial labels and audit conservative dual-teacher completion.

For each fully labelled validation volume, the largest connected lesion is kept
as the synthetic provided annotation. All other lesions are hidden. Candidate
components come from a GF mask outside the trusted annotation box and must have
the requested overlap with an independent support mask (for example,
BiomedParse). This script never writes predictions and never uses hidden data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--gf", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context-mm", default=16.0, type=float)
    parser.add_argument(
        "--analysis-spacing-mm",
        default="2,2,3",
        help="Minimum coarse-grid spacing used only for this bounded audit.",
    )
    parser.add_argument("--min-mm3", default="0,100,500,1000")
    parser.add_argument("--max-mm3", default="2000,10000,inf")
    parser.add_argument("--support-fractions", default="0.05,0.10,0.25,0.50")
    return parser.parse_args()


def values(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",")]


def load_mask(path: Path, minimum_spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    native_spacing = np.asarray(image.header.get_zooms()[:3])
    steps = np.maximum(1, np.ceil(minimum_spacing / native_spacing).astype(int))
    slices = tuple(slice(None, None, int(step)) for step in steps)
    return np.asanyarray(image.dataobj[slices]) > 0, native_spacing * steps


def expanded_box(component_slice: tuple[slice, ...], shape: tuple[int, ...], spacing: np.ndarray, context_mm: float) -> tuple[slice, ...]:
    low = np.asarray([axis.start for axis in component_slice], dtype=int)
    high = np.asarray([axis.stop for axis in component_slice], dtype=int)
    margin = np.ceil(context_mm / spacing).astype(int)
    low = np.maximum(0, low - margin)
    high = np.minimum(shape, high + margin)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def main() -> None:
    args = parse_args()
    minima = values(args.min_mm3)
    maxima = values(args.max_mm3)
    fractions = values(args.support_fractions)
    analysis_spacing = np.asarray(values(args.analysis_spacing_mm), dtype=np.float64)
    if analysis_spacing.shape != (3,) or np.any(analysis_spacing <= 0):
        raise ValueError("--analysis-spacing-mm must contain three positive values")
    configurations = [(minimum, maximum, fraction) for minimum in minima for maximum in maxima for fraction in fractions if minimum <= maximum]
    totals = {
        config: {
            "cases": 0,
            "pseudo_tp": 0,
            "pseudo_voxels": 0,
            "hidden_voxels": 0,
            "pred_hit": 0,
            "pred_count": 0,
            "target_hit": 0,
            "target_count": 0,
            "completed_dice": [],
            "annotated_dice": [],
        }
        for config in configurations
    }

    label_paths = sorted(args.labels.glob("*.nii.gz"))
    if not label_paths:
        raise SystemExit(f"No labels found in {args.labels}")
    for label_path in label_paths:
        gf_path = args.gf / label_path.name
        support_path = args.support / label_path.name
        if not gf_path.is_file() or not support_path.is_file():
            raise FileNotFoundError(f"Missing teacher mask for {label_path.name}")
        target, spacing = load_mask(label_path, analysis_spacing)
        gf, gf_spacing = load_mask(gf_path, analysis_spacing)
        support, support_spacing = load_mask(support_path, analysis_spacing)
        if target.shape != gf.shape or target.shape != support.shape:
            raise ValueError(f"Shape mismatch for {label_path.name}")
        if not np.allclose(spacing, gf_spacing) or not np.allclose(spacing, support_spacing):
            raise ValueError(f"Spacing mismatch for {label_path.name}")

        target_labels = np.empty(target.shape, dtype=np.uint16)
        target_count = int(ndimage.label(target, structure=STRUCTURE, output=target_labels))
        if target_count >= np.iinfo(target_labels.dtype).max:
            raise ValueError(f"Too many target components in {label_path.name}: {target_count}")
        target_sizes = np.bincount(target_labels.ravel(), minlength=target_count + 1)
        annotated_id = int(np.argmax(target_sizes[1:]) + 1) if target_count else 0
        annotated_voxels = int(target_sizes[annotated_id]) if annotated_id else 0
        objects = ndimage.find_objects(target_labels)
        if annotated_id:
            box_slice = expanded_box(objects[annotated_id - 1], target.shape, spacing, args.context_mm)
        else:
            box_slice = tuple(slice(0, 0) for _ in target.shape)

        hidden = target.copy()
        hidden[box_slice] = False
        gf[box_slice] = False
        candidates = np.empty(target.shape, dtype=np.uint16)
        count = int(ndimage.label(gf, structure=STRUCTURE, output=candidates))
        if count >= np.iinfo(candidates.dtype).max:
            raise ValueError(f"Too many GF components in {label_path.name}: {count}")
        voxel_mm3 = float(np.prod(spacing))
        candidate_sizes_vox = np.bincount(candidates.ravel(), minlength=count + 1)
        candidate_support_vox = np.bincount(
            candidates[support].ravel(), minlength=count + 1
        )
        candidate_sizes_mm3 = candidate_sizes_vox.astype(np.float64) * voxel_mm3
        candidate_support_fraction = np.divide(
            candidate_support_vox,
            candidate_sizes_vox,
            out=np.zeros(count + 1, dtype=np.float64),
            where=candidate_sizes_vox > 0,
        )
        candidate_hidden_vox = np.bincount(
            candidates[hidden].ravel(), minlength=count + 1
        )
        hidden_target_ids = np.unique(target_labels[hidden])
        hidden_target_ids = hidden_target_ids[hidden_target_ids > 0]
        pair_region = np.logical_and(hidden, candidates > 0)
        pair_target_ids = target_labels[pair_region].astype(np.uint32, copy=False)
        pair_candidate_ids = candidates[pair_region].astype(np.uint32, copy=False)
        if len(pair_target_ids):
            pair_codes = np.unique(pair_target_ids * np.uint32(count + 1) + pair_candidate_ids)
            pair_targets = pair_codes // np.uint32(count + 1)
            pair_candidates = pair_codes % np.uint32(count + 1)
        else:
            pair_targets = np.empty(0, dtype=np.uint32)
            pair_candidates = np.empty(0, dtype=np.uint32)
        target_voxels = int(target.sum())
        hidden_voxels = int(hidden.sum())
        annotated_case_dice = (
            1.0 if target_voxels == 0 else 2.0 * annotated_voxels / (annotated_voxels + target_voxels)
        )

        for config in configurations:
            minimum, maximum, fraction = config
            selected = np.flatnonzero(
                (candidate_sizes_mm3 >= minimum)
                & (candidate_sizes_mm3 <= maximum)
                & (candidate_support_fraction >= fraction)
            )
            selected = selected[selected > 0]
            selected_lookup = np.zeros(count + 1, dtype=bool)
            selected_lookup[selected] = True
            pseudo_voxels = int(candidate_sizes_vox[selected].sum())
            pseudo_tp = int(candidate_hidden_vox[selected].sum())
            completed_denominator = annotated_voxels + pseudo_voxels + target_voxels
            completed_case_dice = (
                1.0
                if completed_denominator == 0
                else 2.0 * (annotated_voxels + pseudo_tp) / completed_denominator
            )
            record = totals[config]
            record["cases"] += 1
            record["pseudo_tp"] += pseudo_tp
            record["pseudo_voxels"] += pseudo_voxels
            record["hidden_voxels"] += hidden_voxels
            record["pred_hit"] += int(np.count_nonzero(candidate_hidden_vox[selected]))
            record["pred_count"] += int(len(selected))
            hit_targets = np.unique(pair_targets[selected_lookup[pair_candidates]])
            record["target_hit"] += int(len(hit_targets))
            record["target_count"] += int(len(hidden_target_ids))
            record["completed_dice"].append(completed_case_dice)
            record["annotated_dice"].append(annotated_case_dice)

    rows = []
    for (minimum, maximum, fraction), record in totals.items():
        pseudo_precision = record["pseudo_tp"] / record["pseudo_voxels"] if record["pseudo_voxels"] else 1.0
        hidden_recall = record["pseudo_tp"] / record["hidden_voxels"] if record["hidden_voxels"] else 1.0
        component_precision = record["pred_hit"] / record["pred_count"] if record["pred_count"] else 1.0
        component_recall = record["target_hit"] / record["target_count"] if record["target_count"] else 1.0
        annotated_dice = float(np.mean(record["annotated_dice"]))
        completed_dice = float(np.mean(record["completed_dice"]))
        rows.append({
            "min_mm3": minimum,
            "max_mm3": maximum,
            "support_fraction": fraction,
            "pseudo_voxel_precision": pseudo_precision,
            "hidden_voxel_recall": hidden_recall,
            "candidate_component_precision": component_precision,
            "hidden_component_recall": component_recall,
            "annotated_only_macro_dice": annotated_dice,
            "completed_macro_dice": completed_dice,
            "macro_dice_gain": completed_dice - annotated_dice,
            "candidate_components": record["pred_count"],
            "hidden_components": record["target_count"],
        })
    rows.sort(key=lambda row: (row["completed_macro_dice"], row["pseudo_voxel_precision"]), reverse=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "labels": str(args.labels.resolve()),
        "gf": str(args.gf.resolve()),
        "support": str(args.support.resolve()),
        "context_mm": args.context_mm,
        "minimum_analysis_spacing_mm": analysis_spacing.tolist(),
        "cases": len(label_paths),
        "best_by_completed_dice": rows[0],
        "high_precision_candidates": [row for row in rows if row["pseudo_voxel_precision"] >= 0.90][:10],
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
