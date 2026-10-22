#!/usr/bin/env python3
"""Freeze one contour threshold from a predeclared full-volume OOF sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from summarize_candidate_refiner_oof import deployment_segmentation_thresholds


def select(rows: list[dict[str, str]], center: float, expected_cases: int = 210) -> dict:
    expected = deployment_segmentation_thresholds(center)
    if len(rows) != len(expected):
        raise ValueError("segmentation calibration must contain exactly three rows")
    thresholds = [float(row["segmentation_threshold"]) for row in rows]
    if sorted(thresholds) != expected or len(set(thresholds)) != len(expected):
        raise ValueError(
            f"segmentation threshold grid mismatch: expected={expected} "
            f"observed={sorted(thresholds)}"
        )
    classification = {float(row["classification_threshold"]) for row in rows}
    modes = {str(row["deployment_mode"]) for row in rows}
    if len(classification) != 1 or modes != {"refined"}:
        raise ValueError("segmentation sweep must fix classification and refined mode")
    for row in rows:
        for field in ("DSC", "NSD"):
            value = float(row[field])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid {field}: {value}")
    selected = max(
        rows,
        key=lambda row: (
            float(row["DSC"]),
            float(row["NSD"]),
            -abs(float(row["segmentation_threshold"]) - center),
        ),
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": (
            f"exact-{expected_cases} case-held-out full-volume OOF; no public or hidden data"
        ),
        "patch_oof_center": float(center),
        "predeclared_thresholds": expected,
        "candidates": rows,
        "selected_segmentation_threshold": float(
            selected["segmentation_threshold"]
        ),
        "selected_variant": selected["variant"],
        "public_read": False,
        "hidden_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--patch-oof-center", type=float, required=True)
    parser.add_argument("--expected-cases", type=int, default=210)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if args.expected_cases < 1:
        parser.error("expected cases must be positive")
    payload = select(rows, args.patch_oof_center, args.expected_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f'{payload["selected_segmentation_threshold"]:.2f}')


if __name__ == "__main__":
    main()
