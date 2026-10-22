#!/usr/bin/env python3
"""Public-label oracle for proposal-component volume boundaries.

This diagnostic measures three increasingly optimistic ceilings as the maximum
physical proposal-component volume is relaxed: keeping whole touched proposal
components, deleting their false-positive voxels, and reconstructing every
ground-truth component touched by an admitted proposal. It writes no masks and
must never be used to select voxels on hidden data.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean

import cc3d
import nibabel as nib
import numpy as np


DEFAULT_MAXIMA_MM3 = (1000.0, 2000.0, 5000.0, 20000.0, 50000.0, 100000.0, 250000.0, 1e12)


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.count_nonzero(prediction & target))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def analyze_case(
    task: tuple[str, str, str, float, tuple[float, ...]],
) -> dict[str, object]:
    name, target_path, proposal_path, minimum_mm3, maxima_mm3 = task
    target_nii = nib.load(target_path)
    proposal_nii = nib.load(proposal_path)
    if target_nii.shape[:3] != proposal_nii.shape[:3] or not np.allclose(
        target_nii.affine, proposal_nii.affine, atol=1e-3, rtol=0
    ):
        raise ValueError(f"geometry mismatch: {name}")
    target = np.asanyarray(target_nii.dataobj).astype(bool)
    proposal = np.ascontiguousarray(np.asanyarray(proposal_nii.dataobj).astype(bool))
    proposal_labels, count = cc3d.connected_components(
        proposal.astype(np.uint8), connectivity=26, return_N=True
    )
    proposal_counts = np.asarray(
        cc3d.statistics(proposal_labels)["voxel_counts"], dtype=np.int64
    )
    target_labels = cc3d.connected_components(target.astype(np.uint8), connectivity=26)
    voxel_mm3 = float(np.prod(np.linalg.norm(target_nii.affine[:3, :3], axis=0)))
    volumes_mm3 = proposal_counts * voxel_mm3
    overlap = np.bincount(proposal_labels[target], minlength=count + 1)

    row: dict[str, object] = {"case": name.removesuffix(".nii.gz")}
    for maximum_mm3 in maxima_mm3:
        admitted = (volumes_mm3 >= minimum_mm3) & (volumes_mm3 <= maximum_mm3)
        admitted[0] = False
        prediction = admitted[proposal_labels]
        covered = prediction & target
        touched_target_ids = np.unique(target_labels[prediction])
        touched_target_ids = touched_target_ids[touched_target_ids > 0]
        reconstructed = (
            np.isin(target_labels, touched_target_ids)
            if touched_target_ids.size
            else np.zeros_like(target, dtype=bool)
        )
        key = f"max_{int(maximum_mm3):012d}"
        row[f"{key}_candidates"] = int(np.count_nonzero(admitted))
        row[f"{key}_positive_candidates"] = int(
            np.count_nonzero(admitted & (overlap > 0))
        )
        row[f"{key}_whole_component_dsc"] = dice(prediction, target)
        row[f"{key}_voxel_oracle_dsc"] = dice(covered, target)
        row[f"{key}_target_reconstruction_dsc"] = dice(reconstructed, target)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-mm3", type=float, default=0.0)
    parser.add_argument(
        "--maxima-mm3", type=float, nargs="+", default=DEFAULT_MAXIMA_MM3
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    maxima_mm3 = tuple(sorted(set(args.maxima_mm3)))
    if (
        args.workers < 1
        or args.min_mm3 < 0
        or not maxima_mm3
        or any(value <= args.min_mm3 for value in maxima_mm3)
    ):
        parser.error("invalid worker or component-volume limits")

    names = sorted(path.name for path in args.gt.glob("*.nii.gz"))
    if len(names) != 50:
        raise ValueError(f"expected 50 public cases, found {len(names)}")
    if {path.name for path in args.proposal.glob("*.nii.gz")} != set(names):
        raise ValueError("proposal cases do not exactly match public labels")
    tasks = [
        (
            name,
            str(args.gt / name),
            str(args.proposal / name),
            args.min_mm3,
            maxima_mm3,
        )
        for name in names
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(analyze_case, tasks), 1):
            rows.append(row)
            print(f"[{index:02d}/50] {row['case']}", flush=True)

    report = {
        "warning": "Public-label-informed upper bound only; never use for hidden inference or submission.",
        "cases": len(rows),
        "proposal": str(args.proposal.resolve()),
        "minimum_mm3": args.min_mm3,
        "maxima": {},
    }
    for maximum_mm3 in maxima_mm3:
        key = f"max_{int(maximum_mm3):012d}"
        report["maxima"][str(int(maximum_mm3))] = {
            "mean_candidates_per_case": fmean(
                int(row[f"{key}_candidates"]) for row in rows
            ),
            "positive_candidates": sum(
                int(row[f"{key}_positive_candidates"]) for row in rows
            ),
            "whole_component_dsc": fmean(
                float(row[f"{key}_whole_component_dsc"]) for row in rows
            ),
            "voxel_oracle_dsc": fmean(
                float(row[f"{key}_voxel_oracle_dsc"]) for row in rows
            ),
            "target_reconstruction_dsc": fmean(
                float(row[f"{key}_target_reconstruction_dsc"]) for row in rows
            ),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
