#!/usr/bin/env python3
"""Audit superior/inferior foreground-location balance in Dataset501.

Only preprocessed training properties are read. The sampled foreground
locations are mapped back through crop/resampling metadata into physical
superior-inferior coordinates and normalized within each cropped body extent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--properties", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source(identifier: str) -> str:
    if "_" not in identifier:
        raise ValueError(f"identifier has no source separator: {identifier}")
    return identifier.rsplit("_", 1)[0]


def new_shape(old_shape: np.ndarray, old_spacing: np.ndarray, target_spacing: np.ndarray) -> np.ndarray:
    return np.maximum(1, np.rint(old_shape * old_spacing / target_spacing).astype(int))


def foreground_locations(properties: dict) -> np.ndarray:
    for key, value in properties.get("class_locations", {}).items():
        if str(key) in {"1", "(1,)"}:
            if not len(value):
                return np.empty((0, 3), dtype=np.float64)
            return np.asarray(value)[:, 1:4].astype(np.float64)
    raise ValueError("properties contain no class-1 location key")


def normalized_physical_z(properties: dict) -> float:
    old_shape = np.asarray(properties["shape_after_cropping_and_before_resampling"], dtype=float)
    old_spacing = np.asarray(properties["spacing"], dtype=float)
    target_spacing = np.asarray([2.4, 1.6, 1.6], dtype=float)
    resampled_shape = new_shape(old_shape, old_spacing, target_spacing).astype(float)
    locations = foreground_locations(properties)
    center_resampled = np.median(locations, axis=0)
    center_cropped = center_resampled * np.maximum(old_shape - 1, 1) / np.maximum(resampled_shape - 1, 1)
    bbox = np.asarray(properties["bbox_used_for_cropping"], dtype=float)
    center_zyx = center_cropped + bbox[:, 0]

    stuff = properties["sitk_stuff"]
    origin = np.asarray(stuff["origin"], dtype=float)
    spacing_xyz = np.asarray(stuff["spacing"], dtype=float)
    direction = np.asarray(stuff["direction"], dtype=float).reshape(3, 3)

    def physical_z(index_zyx: np.ndarray) -> float:
        index_xyz = index_zyx[::-1]
        return float((origin + direction @ (index_xyz * spacing_xyz))[2])

    corner_z = []
    bounds = [(bbox[axis, 0], bbox[axis, 1] - 1) for axis in range(3)]
    for z in bounds[0]:
        for y in bounds[1]:
            for x in bounds[2]:
                corner_z.append(physical_z(np.asarray([z, y, x], dtype=float)))
    inferior, superior = min(corner_z), max(corner_z)
    if superior <= inferior:
        raise ValueError("cropped physical superior-inferior extent is empty")
    return float(np.clip((physical_z(center_zyx) - inferior) / (superior - inferior), 0, 1))


def main() -> None:
    args = parse_args()
    plans = json.loads(args.plans.read_text(encoding="utf-8"))
    spacing = plans["configurations"]["3d_fullres_ResEncS6"]["spacing"]
    if not np.allclose(spacing, [2.4, 1.6, 1.6], rtol=0, atol=1e-12):
        raise SystemExit(f"unexpected target spacing: {spacing}")
    paths = sorted(args.properties.glob("*.pkl"))
    if len(paths) != 17575:
        raise SystemExit(f"expected 17575 property files, found {len(paths)}")

    rows = []
    empty_cases = []
    direction_counts: Counter[tuple[float, ...]] = Counter()
    for index, path in enumerate(paths, start=1):
        with path.open("rb") as stream:
            properties = pickle.load(stream)
        identifier = path.stem
        direction = tuple(float(value) for value in properties["sitk_stuff"]["direction"])
        direction_counts[direction] += 1
        locations = foreground_locations(properties)
        if not len(locations):
            empty_cases.append(identifier)
            continue
        rows.append({
            "identifier": identifier,
            "source": source(identifier),
            "body_relative_superior_inferior": normalized_physical_z(properties),
            "foreground_proxy_voxels": len(locations),
        })
        if index % 2000 == 0:
            print(f"training location audit {index}/{len(paths)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    cases_path = args.output / "cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    values = np.asarray([row["body_relative_superior_inferior"] for row in rows], dtype=float)
    edges = np.linspace(0, 1, 6)
    counts, _ = np.histogram(values, bins=edges)
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(float(row["body_relative_superior_inferior"]))
    source_summary = {
        name: {
            "cases": len(items),
            "median": float(np.median(items)),
            "fraction_below_0_25": float(np.mean(np.asarray(items) < 0.25)),
            "fraction_above_0_75": float(np.mean(np.asarray(items) > 0.75)),
        }
        for name, items in sorted(by_source.items())
    }
    digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "PASS",
        "dataset": "Dataset501_PanCancer",
        "dataset_case_count": len(paths),
        "positive_case_count": len(rows),
        "empty_foreground_case_count": len(empty_cases),
        "empty_foreground_cases": empty_cases,
        "source_count": len(by_source),
        "target_spacing_mm": spacing,
        "physical_axis": "superior_inferior_normalized_in_cropped_body; 0=inferior, 1=superior",
        "quantiles": {
            str(q): float(np.quantile(values, q)) for q in (0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)
        },
        "five_bin_edges": edges.tolist(),
        "five_bin_counts": counts.tolist(),
        "five_bin_fractions": (counts / len(values)).tolist(),
        "fraction_below_0_25": float(np.mean(values < 0.25)),
        "fraction_above_0_75": float(np.mean(values > 0.75)),
        "direction_counts": {str(key): value for key, value in direction_counts.items()},
        "sources": source_summary,
        "cases": {"path": str(cases_path.resolve()), "sha256": digest},
        "public_read": False,
        "hidden_read": False,
        "training_mutated": False,
        "submission_created": False,
        "uploaded": False
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
