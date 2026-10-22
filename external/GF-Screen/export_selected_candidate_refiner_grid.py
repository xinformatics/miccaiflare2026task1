#!/usr/bin/env python3
"""Export one hash-stable materialization configuration from OOF calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


SOURCES = {
    "epoch105",
    "currentgf",
    "autopet_weak",
    "lndb_weak",
    *(f"lung_cell{index}" for index in range(8)),
}
MODES = {"refined", "refined_or_seed", "seed", "seed_crop", "seed_union", "prune_base"}


def probability_map(value: object, label: str) -> dict[str, float]:
    mapping = dict(value or {})
    if set(mapping) - SOURCES:
        raise ValueError(f"unexpected {label} sources: {sorted(set(mapping) - SOURCES)}")
    output = {str(name): float(threshold) for name, threshold in mapping.items()}
    if any(not math.isfinite(value) or not 0 < value < 1 for value in output.values()):
        raise ValueError(f"invalid {label} threshold")
    return dict(sorted(output.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.tag or "/" in args.tag or "\\" in args.tag:
        parser.error("tag must be a nonempty path-safe name")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    selected = dict(calibration.get("selected", {}))
    classification = float(
        selected.get("classification_threshold", calibration["classification_threshold"])
    )
    segmentation = float(
        selected.get("segmentation_threshold", calibration["segmentation_threshold"])
    )
    mode = str(selected.get("deployment_mode", calibration.get("deployment_mode", "refined")))
    if not 0 < classification < 1 or not 0 < segmentation < 1 or mode not in MODES:
        raise ValueError("invalid selected global threshold or deployment mode")
    source_classification = probability_map(
        selected.get(
            "source_classification_thresholds",
            calibration.get("source_classification_thresholds", {}),
        ),
        "source classification",
    )
    source_segmentation = probability_map(
        selected.get(
            "source_segmentation_thresholds",
            calibration.get("source_segmentation_thresholds", {}),
        ),
        "source segmentation",
    )
    source_modes = {
        str(name): str(value)
        for name, value in dict(
            selected.get(
                "source_deployment_modes",
                calibration.get("source_deployment_modes", {}),
            )
            or {}
        ).items()
    }
    if set(source_modes) - SOURCES or any(value not in MODES for value in source_modes.values()):
        raise ValueError("invalid selected source deployment modes")
    payload = {
        "schema_version": 1,
        "selection_basis": "single deployment point frozen by exact full-volume OOF calibration",
        "calibration": str(args.calibration.resolve()),
        "configurations": [
            {
                "tag": args.tag,
                "classification_threshold": classification,
                "segmentation_threshold": segmentation,
                "mode": mode,
                "source_classification_thresholds": source_classification,
                "source_segmentation_thresholds": source_segmentation,
                "source_deployment_modes": dict(sorted(source_modes.items())),
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
