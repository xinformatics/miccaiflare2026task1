#!/usr/bin/env python3
"""Diagnostic global threshold sweep over fixed grouped-OOF probabilities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.components.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)

    case_stats = {}
    for label_path in sorted(args.labels.glob("*.nii.gz")):
        case = label_path.name.removesuffix(".nii.gz")
        target = np.asanyarray(nib.load(str(label_path)).dataobj).astype(bool)
        base = np.asanyarray(nib.load(str(args.base / label_path.name)).dataobj).astype(bool)
        case_stats[case] = {
            "base_voxels": int(base.sum()),
            "target_voxels": int(target.sum()),
            "base_overlap": int(np.count_nonzero(base & target)),
        }

    total_true = sum(int(row["is_true_component"]) for row in rows)
    results = []
    for threshold in np.linspace(0.05, 0.95, 19):
        scores = []
        anchor_scores = []
        selected = true_selected = wins = catastrophic = recovered_zero = 0
        for case, stats in sorted(case_stats.items()):
            chosen = [
                row
                for row in by_case.get(case, [])
                if float(row["oof_probability"]) >= threshold
            ]
            selected += len(chosen)
            true_selected += sum(int(row["is_true_component"]) for row in chosen)
            predicted = stats["base_voxels"] + sum(
                int(row["component_voxels"]) for row in chosen
            )
            overlap = stats["base_overlap"] + sum(
                int(row["target_overlap_voxels"]) for row in chosen
            )
            score = 2.0 * overlap / max(predicted + stats["target_voxels"], 1)
            anchor = 2.0 * stats["base_overlap"] / max(
                stats["base_voxels"] + stats["target_voxels"], 1
            )
            scores.append(score)
            anchor_scores.append(anchor)
            wins += int(score > anchor)
            catastrophic += int(score < anchor - 0.05)
            recovered_zero += int(anchor == 0 and score > 0)
        results.append(
            {
                "threshold": float(threshold),
                "macro_DSC": float(np.mean(scores)),
                "DSC_gain": float(np.mean(scores) - np.mean(anchor_scores)),
                "selected_components": selected,
                "selected_component_precision": true_selected / max(selected, 1),
                "selected_component_recall": true_selected / max(total_true, 1),
                "win_rate": wins / len(scores),
                "catastrophic_regression_rate": catastrophic / len(scores),
                "recovered_anchor_zero_cases": recovered_zero,
            }
        )
    report = {
        "warning": "Public-label diagnostic over fixed grouped-OOF probabilities.",
        "selection_uses_public_labels": True,
        "best_by_DSC": max(results, key=lambda row: row["macro_DSC"]),
        "thresholds": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
