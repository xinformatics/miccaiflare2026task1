#!/usr/bin/env python3
"""Screen a bounded, inference-observable per-case Focus threshold rule.

Very large low-confidence masks are a recurrent GF-Screen failure mode. The
rule retains the locked 0.50/1000 mm3 output by default and switches a case to
a stricter precomputed threshold only when (1) its baseline predicted burden
is large and (2) little of that burden survives the stricter threshold. No CT,
ground-truth feature, filename, or cohort metadata enters the routing decision.
Ground truth is used only through existing metric CSVs to rank the bounded grid.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["Name"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate names in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("public_sweeps/adaptive_burden_thresholds_20260820"),
    )
    parser.add_argument("--high-thresholds", default="0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--burden-mm3", default="50000,100000,200000,400000,800000")
    parser.add_argument("--max-retention", default="0.10,0.25,0.50,0.75")
    args = parser.parse_args()

    baseline_tag = "lesion_0.50_mm1000"
    baseline_dir = args.root / baseline_tag
    baseline = read_metrics(args.root / f"{baseline_tag}.metrics.csv")
    names = sorted(baseline)
    volume_mm3: dict[str, dict[str, float]] = {"0.50": {}}

    for name in names:
        path = baseline_dir / f"{name}.nii.gz"
        image = nib.load(path)
        voxel_mm3 = float(np.prod(image.header.get_zooms()[:3]))
        volume_mm3["0.50"][name] = float(baseline[name]["pred_fg_voxels"]) * voxel_mm3

    thresholds = [float(value) for value in args.high_thresholds.split(",")]
    burdens = [float(value) for value in args.burden_mm3.split(",")]
    retentions = [float(value) for value in args.max_retention.split(",")]
    metric_sets: dict[str, dict[str, dict[str, str]]] = {"0.50": baseline}
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        tag = f"lesion_{key}_mm1000"
        metrics = read_metrics(args.root / f"{tag}.metrics.csv")
        if set(metrics) != set(names):
            raise ValueError(f"case mismatch for {tag}")
        metric_sets[key] = metrics
        volume_mm3[key] = {}
        for name in names:
            path = args.root / tag / f"{name}.nii.gz"
            image = nib.load(path)
            voxel_mm3 = float(np.prod(image.header.get_zooms()[:3]))
            volume_mm3[key][name] = float(metrics[name]["pred_fg_voxels"]) * voxel_mm3

    candidates = []
    selected_rows: dict[str, list[dict[str, object]]] = {}
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        for burden in burdens:
            for max_retention in retentions:
                rows = []
                switched = 0
                for name in names:
                    base_volume = volume_mm3["0.50"][name]
                    retention = volume_mm3[key][name] / max(base_volume, 1e-6)
                    switch = base_volume >= burden and retention <= max_retention
                    source = key if switch else "0.50"
                    metric = metric_sets[source][name]
                    switched += int(switch)
                    rows.append({
                        "Name": name,
                        "Lesion_DSC": float(metric["Lesion_DSC"]),
                        "Lesion_NSD": float(metric["Lesion_NSD"]),
                        "pred_fg_voxels": int(metric["pred_fg_voxels"]),
                        "selected_threshold": float(source),
                        "baseline_volume_mm3": base_volume,
                        "high_threshold_retention": retention,
                    })
                dsc = float(np.mean([row["Lesion_DSC"] for row in rows]))
                nsd = float(np.mean([row["Lesion_NSD"] for row in rows]))
                tag = f"high{key}_burden{int(burden)}_ret{max_retention:.2f}"
                candidates.append({
                    "tag": tag,
                    "high_threshold": threshold,
                    "min_baseline_burden_mm3": burden,
                    "max_high_threshold_retention": max_retention,
                    "switched_cases": switched,
                    "Lesion_DSC": dsc,
                    "Lesion_NSD": nsd,
                    "combined": (dsc + nsd) / 2.0,
                })
                selected_rows[tag] = rows

    candidates.sort(key=lambda row: row["combined"], reverse=True)
    best = candidates[0]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(candidates)
    with (args.output / "best.metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "Name", "Lesion_DSC", "Lesion_NSD", "pred_fg_voxels",
                "selected_threshold", "baseline_volume_mm3", "high_threshold_retention",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_rows[best["tag"]])
    (args.output / "best.json").write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
