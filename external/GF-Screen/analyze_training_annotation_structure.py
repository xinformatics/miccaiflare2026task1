#!/usr/bin/env python3
"""Stratified audit of lesion-label structure across training cohorts.

The challenge labels only a primary lesion in general, but its merged public
datasets need not all have identical annotation completeness.  This script is
diagnostic only: it samples cases deterministically within each coarse source
cohort and reports connected-component and enclosing-box statistics.  It never
reads validation data and does not modify labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from utils.data_utils import _cohort_key


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def cohort(record: dict[str, str], region: str) -> str:
    return _cohort_key(record, region)


def inspect_case(task: tuple[str, str, str]) -> dict[str, object]:
    group, name, path = task
    image = nib.load(path)
    mask = np.asanyarray(image.dataobj).astype(np.float32, copy=False) > 0.5
    labels, count = ndimage.label(mask, structure=CONNECTIVITY_26)
    component_sizes = np.bincount(labels.ravel())[1:]
    foreground = int(mask.sum())
    voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    if foreground:
        coordinates = np.argwhere(mask)
        extents = coordinates.max(0) - coordinates.min(0) + 1
        box_voxels = int(np.prod(extents, dtype=np.int64))
        box_occupancy = foreground / box_voxels
    else:
        box_voxels = 0
        box_occupancy = 0.0
    largest_fraction = (
        float(component_sizes.max() / foreground)
        if foreground and component_sizes.size else 0.0
    )
    return {
        "cohort": group,
        "name": name,
        "label": path,
        "components": int(count),
        "foreground_voxels": foreground,
        "foreground_mm3": foreground * voxel_volume,
        "largest_component_fraction": largest_fraction,
        "enclosing_box_voxels": box_voxels,
        "enclosing_box_occupancy": box_occupancy,
    }


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        action="append",
        nargs=2,
        metavar=("REGION", "PATH"),
        default=None,
        help="region name and Decathlon JSON; may be repeated",
    )
    parser.add_argument("--per-cohort", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/training_annotation_structure_20260820"),
    )
    args = parser.parse_args()
    json_specs = args.json or [
        ["abdomen", "jsons/local/abdomen.json"],
        ["chest", "jsons/local/chest.json"],
    ]
    if args.per_cohort < 1 or args.workers < 1:
        parser.error("--per-cohort and --workers must be positive")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for region, json_path in json_specs:
        with Path(json_path).open() as stream:
            dataset = json.load(stream)
        for record in dataset["training"]:
            grouped[cohort(record, region)].append(record)

    rng = np.random.default_rng(args.seed)
    tasks: list[tuple[str, str, str]] = []
    sampled_counts: dict[str, tuple[int, int]] = {}
    for group in sorted(grouped):
        records = grouped[group]
        sample_count = min(args.per_cohort, len(records))
        selected = np.sort(rng.choice(len(records), size=sample_count, replace=False))
        sampled_counts[group] = (sample_count, len(records))
        tasks.extend(
            (group, records[index]["name"], records[index]["label"])
            for index in selected
        )

    context = mp.get_context("spawn")
    with context.Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(inspect_case, tasks, chunksize=2))
    rows.sort(key=lambda row: (str(row["cohort"]), str(row["name"])))

    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["cohort"])].append(row)
    summary = []
    for group in sorted(by_group):
        group_rows = by_group[group]
        components = [int(row["components"]) for row in group_rows]
        largest = [float(row["largest_component_fraction"]) for row in group_rows]
        occupancy = [float(row["enclosing_box_occupancy"]) for row in group_rows]
        sampled, available = sampled_counts[group]
        summary.append({
            "cohort": group,
            "sampled_cases": sampled,
            "available_cases": available,
            "nonempty_fraction": float(np.mean(np.asarray(components) > 0)),
            "multicomponent_fraction": float(np.mean(np.asarray(components) > 1)),
            "components_mean": float(np.mean(components)),
            "components_p50": quantile(components, 0.50),
            "components_p90": quantile(components, 0.90),
            "largest_component_fraction_p50": quantile(largest, 0.50),
            "enclosing_box_occupancy_p50": quantile(occupancy, 0.50),
        })
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = {
        "seed": args.seed,
        "per_cohort_limit": args.per_cohort,
        "sampled_cases": len(rows),
        "cohorts": len(summary),
        "summary": summary,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print((args.output / "summary.csv").read_text(), end="")


if __name__ == "__main__":
    main()
