#!/usr/bin/env python3
"""Summarize positive proposal geometry by cohort from a refiner patch index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("geometry values must be nonempty and finite")
    return {
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(values.max()),
    }


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("labels must be binary")
        if label:
            grouped[str(row["cohort"])].append(row)
    if not grouped:
        raise ValueError("index contains no positive candidates")
    reports = []
    for cohort, positive in sorted(grouped.items()):
        volumes = np.asarray([float(row["seed_mm3"]) for row in positive])
        overlaps = np.asarray(
            [float(row["seed_overlap_fraction"]) for row in positive]
        )
        intersections = np.asarray(
            [int(row["intersection_voxels"]) for row in positive], dtype=np.float64
        )
        if np.any(volumes <= 0) or np.any(overlaps < 0) or np.any(overlaps > 1):
            raise ValueError(f"invalid positive geometry in {cohort}")
        cases = {str(row["case"]) for row in positive}
        components = {
            (str(row["case"]), int(row["component_id"])) for row in positive
        }
        reports.append(
            {
                "cohort": cohort,
                "positive_cases": len(cases),
                "positive_components": len(components),
                "positive_tiles": len(positive),
                "tiles_per_positive_component": float(
                    len(positive) / len(components)
                ),
                "seed_mm3": quantiles(volumes),
                "seed_overlap_fraction": quantiles(overlaps),
                "intersection_voxels": quantiles(intersections),
                "target_resample_fallbacks": sum(
                    int(row.get("target_resample_fallback", 0)) for row in positive
                ),
            }
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "index": str(args.index.resolve()),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "selection": "all positive patch-index rows; no threshold retuning",
        "cohorts": summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
