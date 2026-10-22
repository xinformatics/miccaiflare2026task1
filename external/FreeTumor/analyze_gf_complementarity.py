#!/usr/bin/env python3
"""Measure whether an independent teacher contains DSC-relevant GF misses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "GF-Screen"))
from fast_official_lesion_metrics import score_case  # noqa: E402


FILTERS_MM3 = (0, 500, 1000, 3000)
HIGH_VALUE_MM3 = 3000.0
DETECTION_OVERLAP_FRACTION = 0.10


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def connected_components(
    mask: np.ndarray, connectivity: int = 26
) -> tuple[np.ndarray, int]:
    if connectivity not in {6, 26}:
        raise ValueError("connectivity must be 6 or 26")
    structure = (
        ndimage.generate_binary_structure(3, 1)
        if connectivity == 6
        else np.ones((3, 3, 3), bool)
    )
    return ndimage.label(mask.astype(bool), structure=structure)


def filter_components(
    mask: np.ndarray, voxel_mm3: float, minimum_mm3: float
) -> np.ndarray:
    # Match the exact challenge deployment filter implementation.
    labels, count = connected_components(mask, connectivity=6)
    if count == 0 or minimum_mm3 <= 0:
        return mask.astype(bool, copy=False)
    sizes = np.bincount(labels.ravel()) * voxel_mm3
    keep = sizes >= minimum_mm3
    keep[0] = False
    return keep[labels]


def oracle_supported_components(candidate: np.ndarray, gt: np.ndarray) -> np.ndarray:
    labels, count = connected_components(candidate, connectivity=6)
    if count == 0:
        return np.zeros_like(candidate, dtype=bool)
    touched = np.unique(labels[gt.astype(bool)])
    keep = np.zeros(count + 1, dtype=bool)
    keep[touched[touched > 0]] = True
    return keep[labels]


def large_miss_recovery(
    gt: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    voxel_mm3: float,
) -> list[dict]:
    labels, count = connected_components(gt)
    rows = []
    for component in range(1, count + 1):
        lesion = labels == component
        voxels = int(lesion.sum())
        volume = voxels * voxel_mm3
        if volume < HIGH_VALUE_MM3:
            continue
        anchor_fraction = float(np.count_nonzero(anchor & lesion) / voxels)
        if anchor_fraction >= DETECTION_OVERLAP_FRACTION:
            continue
        candidate_fraction = float(np.count_nonzero(candidate & lesion) / voxels)
        rows.append(
            {
                "component": component,
                "volume_mm3": volume,
                "anchor_overlap_fraction": anchor_fraction,
                "candidate_overlap_fraction": candidate_fraction,
                "candidate_recovered": (
                    candidate_fraction >= DETECTION_OVERLAP_FRACTION
                ),
            }
        )
    return rows


def evaluate(
    labels_dir: Path,
    anchor_dir: Path,
    candidate_dir: Path,
) -> dict:
    names = sorted(path.name for path in labels_dir.glob("*.nii.gz"))
    if not names:
        raise FileNotFoundError(f"no public labels in {labels_dir}")
    expected = set(names)
    for description, directory in (
        ("anchor", anchor_dir),
        ("candidate", candidate_dir),
    ):
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected:
            raise ValueError(
                f"{description} inventory mismatch: "
                f"missing={sorted(expected-observed)} extra={sorted(observed-expected)}"
            )

    score_rows: dict[str, list[tuple[float, float]]] = {
        "anchor": [],
        "candidate": [],
        "oracle_supported_union": [],
        **{f"union_mm{minimum}": [] for minimum in FILTERS_MM3},
    }
    large_misses = []
    closed_cases = {"FLARE23Ts_0009.nii.gz", "FLARE23Ts_0029.nii.gz", "FLARE23Ts_0079.nii.gz"}
    for name in names:
        gt_image = nib.load(str(labels_dir / name))
        gt = np.asanyarray(gt_image.dataobj).astype(bool, copy=False)
        anchor = np.asanyarray(nib.load(str(anchor_dir / name)).dataobj).astype(
            bool, copy=False
        )
        candidate = np.asanyarray(
            nib.load(str(candidate_dir / name)).dataobj
        ).astype(bool, copy=False)
        if gt.shape != anchor.shape or gt.shape != candidate.shape:
            raise ValueError(f"shape mismatch for {name}")
        spacing = gt_image.header.get_zooms()[:3]
        voxel_mm3 = float(np.prod(spacing))
        masks = {
            "anchor": anchor,
            "candidate": candidate,
            "oracle_supported_union": anchor
            | oracle_supported_components(candidate, gt),
        }
        for minimum in FILTERS_MM3:
            masks[f"union_mm{minimum}"] = anchor | filter_components(
                candidate, voxel_mm3, minimum
            )
        for key, mask in masks.items():
            score_rows[key].append(score_case(gt, mask, spacing))
        for row in large_miss_recovery(gt, anchor, candidate, voxel_mm3):
            large_misses.append(
                {
                    "case": name.removesuffix(".nii.gz"),
                    "closed_to_gf_postprocessing": name in closed_cases,
                    **row,
                }
            )

    metrics = {}
    for key, rows in score_rows.items():
        values = np.asarray(rows, dtype=float)
        metrics[key] = {
            "Lesion_DSC": float(values[:, 0].mean()),
            "Lesion_NSD": float(values[:, 1].mean()),
        }
    anchor_dsc = metrics["anchor"]["Lesion_DSC"]
    anchor_nsd = metrics["anchor"]["Lesion_NSD"]
    direct = []
    for minimum in FILTERS_MM3:
        key = f"union_mm{minimum}"
        direct.append(
            {
                "configuration": key,
                **metrics[key],
                "DSC_gain": metrics[key]["Lesion_DSC"] - anchor_dsc,
                "NSD_gain": metrics[key]["Lesion_NSD"] - anchor_nsd,
            }
        )
    best_direct = max(
        direct,
        key=lambda row: (row["Lesion_DSC"], row["Lesion_NSD"]),
    )
    recovered = [row for row in large_misses if row["candidate_recovered"]]
    recovered_closed = [
        row for row in recovered if row["closed_to_gf_postprocessing"]
    ]
    oracle = metrics["oracle_supported_union"]
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "case_count": len(names),
        "metrics": metrics,
        "direct_union_grid": direct,
        "best_direct_union": best_direct,
        "oracle_supported_union_gain": {
            "DSC": oracle["Lesion_DSC"] - anchor_dsc,
            "NSD": oracle["Lesion_NSD"] - anchor_nsd,
        },
        "large_anchor_misses_ge3000": large_misses,
        "large_anchor_miss_count": len(large_misses),
        "large_misses_recovered_by_candidate": len(recovered),
        "closed_gf_cases_recovered_by_candidate": len(recovered_closed),
        "recovered_cases": sorted({row["case"] for row in recovered}),
        "recovered_closed_cases": sorted(
            {row["case"] for row in recovered_closed}
        ),
        "independent_proposal_signal": bool(recovered),
        "direct_union_public_gate": bool(
            best_direct["Lesion_DSC"] >= 0.6154316379500749
            and best_direct["Lesion_NSD"] >= 0.5104225107769019
        ),
        "role": "public_label_diagnostic_only_not_submission_authority",
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = evaluate(args.labels, args.anchor, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
