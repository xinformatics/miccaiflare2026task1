#!/usr/bin/env python3
"""Public-only sweep of the STU probability used to confirm whole anchor components."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / "external" / "GF-Screen"))
from fast_official_lesion_metrics import (  # noqa: E402
    prepare_surface_reference,
    surface_dice_with_reference,
)

sys.path.insert(0, str(HERE))
from analyze_stunet_gf_ensemble import (  # noqa: E402
    load_compact_probability,
    probability_name,
)


def load_case(item: tuple[str, str, str, str, str, tuple[float, ...]]):
    name, labels_dir, anchor_dir, probability_dir, freetumor_dir, thresholds = item
    target_image = nib.load(str(Path(labels_dir) / name))
    target = np.asanyarray(target_image.dataobj).astype(bool, copy=False)
    anchor_image = nib.load(str(Path(anchor_dir) / name))
    freetumor_image = nib.load(str(Path(freetumor_dir) / name))
    if (
        anchor_image.shape != target_image.shape
        or freetumor_image.shape != target_image.shape
        or not np.allclose(anchor_image.affine, target_image.affine, atol=1e-4)
        or not np.allclose(freetumor_image.affine, target_image.affine, atol=1e-4)
    ):
        raise ValueError(f"geometry mismatch: {name}")
    anchor = np.asanyarray(anchor_image.dataobj).astype(bool, copy=False)
    freetumor = np.asanyarray(freetumor_image.dataobj).astype(bool, copy=False)
    probability = load_compact_probability(
        Path(probability_dir) / probability_name(name), target_image.shape
    )
    spacing = tuple(float(value) for value in target_image.header.get_zooms()[:3])
    return name, target, anchor, freetumor, probability, spacing, thresholds


def anchor_component_table(
    anchor: np.ndarray, probability: np.ndarray, freetumor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the exact whole-component confirmation state once per case."""

    labels, count = ndimage.label(anchor, structure=np.ones((3, 3, 3)))
    component_ids = np.arange(1, count + 1)
    maximum_probability = np.zeros(count + 1, dtype=np.float32)
    if count:
        maximum_probability[1:] = ndimage.maximum(
            probability, labels=labels, index=component_ids
        )
    freetumor_ids = np.zeros(count + 1, dtype=bool)
    freetumor_ids[np.unique(labels[anchor & freetumor])] = True
    freetumor_ids[0] = False
    return labels, maximum_probability, freetumor_ids


def confirmed_mask(
    labels: np.ndarray,
    maximum_probability: np.ndarray,
    freetumor_ids: np.ndarray,
    threshold: float,
) -> np.ndarray:
    keep = (maximum_probability > threshold) | freetumor_ids
    keep[0] = False
    return keep[labels]


def score_dsc_one(item: tuple[str, str, str, str, str, tuple[float, ...]]):
    name, target, anchor, freetumor, probability, _, thresholds = load_case(item)
    labels, maximum_probability, freetumor_ids = anchor_component_table(
        anchor, probability, freetumor
    )
    scores = {}
    voxels = {}
    target_voxels = int(target.sum())
    for threshold in thresholds:
        mask = confirmed_mask(
            labels, maximum_probability, freetumor_ids, threshold
        )
        mask_voxels = int(mask.sum())
        intersection = int(np.logical_and(target, mask).sum())
        scores[threshold] = float(
            2.0 * intersection / (target_voxels + mask_voxels)
        )
        voxels[threshold] = int(mask.sum())
    return name, scores, voxels


def score_nsd_one(item: tuple[str, str, str, str, str, tuple[float, ...]]):
    name, target, anchor, freetumor, probability, spacing, thresholds = load_case(item)
    labels, maximum_probability, freetumor_ids = anchor_component_table(
        anchor, probability, freetumor
    )
    reference = prepare_surface_reference(target, spacing)
    scores = {}
    for threshold in thresholds:
        mask = confirmed_mask(
            labels, maximum_probability, freetumor_ids, threshold
        )
        scores[threshold] = (
            surface_dice_with_reference(reference, mask, spacing)
            if mask.any()
            else 0.0
        )
    return name, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--stunet-probabilities", required=True, type=Path)
    parser.add_argument("--freetumor", required=True, type=Path)
    parser.add_argument("--threshold", action="append", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dsc-only", action="store_true")
    args = parser.parse_args()

    thresholds = tuple(sorted(set(args.threshold)))
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("thresholds must lie strictly between zero and one")
    names = sorted(path.name for path in args.labels.glob("*.nii.gz"))
    if len(names) != 50:
        raise RuntimeError(f"expected fixed public 50, got {len(names)}")
    items = [
        (
            name,
            str(args.labels),
            str(args.anchor),
            str(args.stunet_probabilities),
            str(args.freetumor),
            thresholds,
        )
        for name in names
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(score_dsc_one, items))

    metrics = {}
    for threshold in thresholds:
        values = np.asarray([row[1][threshold] for row in rows], dtype=float)
        metrics[f"p{int(round(threshold * 100)):02d}"] = {
            "threshold": threshold,
            "Lesion_DSC": float(values.mean()),
            "Lesion_NSD": None,
        }
    ranked = sorted(
        thresholds,
        key=lambda threshold: metrics[f"p{int(round(threshold * 100)):02d}"]["Lesion_DSC"],
        reverse=True,
    )
    if args.dsc_only:
        best = f"p{int(round(ranked[0] * 100)):02d}"
        payload = {
            "schema_version": 1,
            "status": "COMPLETE",
            "role": "fixed_public_STU_anchor_confirmation_DSC_threshold_sweep",
            "case_count": len(rows),
            "metrics": metrics,
            "best": {"configuration": best, **metrics[best]},
            "public_labels_read": True,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("case", "threshold", "Lesion_DSC", "Lesion_NSD", "voxels"))
            for name, scores, voxels in rows:
                for threshold in thresholds:
                    writer.writerow((name, threshold, scores[threshold], "", voxels[threshold]))
        print(json.dumps(payload, indent=2))
        return
    nsd_thresholds = tuple(ranked[:3])
    nsd_items = [(*item[:-1], nsd_thresholds) for item in items]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        nsd_rows = list(executor.map(score_nsd_one, nsd_items))
    nsd_by_case = {name: scores for name, scores in nsd_rows}
    for threshold in nsd_thresholds:
        key = f"p{int(round(threshold * 100)):02d}"
        metrics[key]["Lesion_NSD"] = float(
            np.mean([nsd_by_case[name][threshold] for name, _, _ in rows])
        )
    best = max(
        (f"p{int(round(threshold * 100)):02d}" for threshold in nsd_thresholds),
        key=lambda key: (metrics[key]["Lesion_DSC"], metrics[key]["Lesion_NSD"]),
    )
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "role": "fixed_public_STU_anchor_confirmation_threshold_sweep",
        "case_count": len(rows),
        "metrics": metrics,
        "best": {"configuration": best, **metrics[best]},
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "threshold", "Lesion_DSC", "Lesion_NSD", "voxels"))
        for name, scores, voxels in rows:
            for threshold in thresholds:
                writer.writerow(
                    (
                        name,
                        threshold,
                        scores[threshold],
                        nsd_by_case.get(name, {}).get(threshold, ""),
                        voxels[threshold],
                    )
                )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
