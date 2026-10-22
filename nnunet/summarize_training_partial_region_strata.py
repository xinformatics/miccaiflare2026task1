#!/usr/bin/env python3
"""Summarize source and center strata from the frozen training-context audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


POLICY = "shape_linf_band_12mm"
QUANTILES = (0.0, 0.1, 0.5, 0.9, 1.0)


def summarize(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot summarize an empty stratum")
    return {
        "patches": int(len(array)),
        "background_to_foreground_ratio_quantiles": {
            str(q): float(np.quantile(array, q)) for q in QUANTILES
        },
        "fraction_below_5": float(np.mean(array < 5.0)),
    }


def analyze(rows_path: Path, expected_sha256: str) -> dict[str, object]:
    observed_sha = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    if observed_sha != expected_sha256:
        raise ValueError("patch-metric SHA256 mismatch")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    identifiers = {str(row["identifier"]) for row in rows}
    sources = {str(row["source"]) for row in rows}
    if len(rows) != 1120 or len(identifiers) != 280 or len(sources) != 28:
        raise ValueError("unexpected frozen audit inventory")
    per_identifier: dict[str, list[str]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    by_center: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["policies"][POLICY]["background_to_foreground_ratio"])
        identifier = str(row["identifier"])
        center = str(row["center_kind"])
        per_identifier[identifier].append(center)
        by_source[str(row["source"])].append(value)
        by_center[center].append(value)
    expected_centers = sorted(["foreground_voxel"] * 3 + ["component_uniform"])
    if any(sorted(items) != expected_centers for items in per_identifier.values()):
        raise ValueError("case-level sampler mix changed")
    source_summary = {name: summarize(values) for name, values in sorted(by_source.items())}
    center_summary = {name: summarize(values) for name, values in sorted(by_center.items())}
    below = [
        name
        for name, item in source_summary.items()
        if item["background_to_foreground_ratio_quantiles"]["0.5"] < 5.0
    ]
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE_DESCRIPTIVE_NO_POLICY_CHANGE",
        "input": {"path": str(rows_path.resolve()), "sha256": observed_sha},
        "policy": POLICY,
        "sources": source_summary,
        "center_kinds": center_summary,
        "sources_with_median_ratio_below_5": below,
        "interpretation": [
            "This source breakdown was inspected after the global gate and is descriptive, not a new selection rule.",
            "A fixed physical boundary shell has a lower shell-volume/lesion-volume ratio for large lesions; a low ratio does not mean the lesion boundary lacks 12 mm of trusted context.",
            "The foreground-voxel and component-uniform sampler branches have nearly matched distributions, so center selection does not explain the source differences.",
            "Do not expand low-ratio patches adaptively without trained fixed-public evidence because doing so increases exposure to omitted lesions.",
        ],
        "decision": "KEEP_PREDECLARED_FIXED_12MM_SHAPE_BAND_CANDIDATE",
        "public_labels_read": False,
        "hidden_read": False,
        "training_mutated": False,
        "submission_created": False,
        "uploaded": False,
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = analyze(args.rows, args.expected_sha256)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
