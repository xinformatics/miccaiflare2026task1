#!/usr/bin/env python3
"""Gate a predeclared source-routed contour policy on exact held-out OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(
    baseline: dict,
    candidate: dict,
    minimum_dice_gain: float = 0.002,
    maximum_group_drop: float = 0.002,
    probability_tolerance: float = 1e-8,
) -> tuple[str, dict[str, object], dict[str, bool]]:
    before_ap = float(baseline["classification"]["average_precision"])
    after_ap = float(candidate["classification"]["average_precision"])
    before_dice = float(baseline["segmentation"]["dice"])
    after_dice = float(candidate["segmentation"]["dice"])
    baseline_groups = baseline.get("source_segmentation_groups", {})
    candidate_groups = candidate.get("source_segmentation_groups", {})
    required_groups = ("mature", "weak_sources", "lung_cells")
    group_gains = {
        name: float(candidate_groups[name]["dice"])
        - float(baseline_groups[name]["dice"])
        for name in required_groups
        if name in baseline_groups and name in candidate_groups
    }
    metrics: dict[str, object] = {
        "baseline_average_precision": before_ap,
        "candidate_average_precision": after_ap,
        "average_precision_delta": after_ap - before_ap,
        "baseline_segmentation_dice": before_dice,
        "candidate_segmentation_dice": after_dice,
        "segmentation_dice_gain": after_dice - before_dice,
        "group_segmentation_dice_gains": group_gains,
    }
    checks = {
        "classification_identical": abs(after_ap - before_ap)
        <= probability_tolerance,
        "meaningful_global_contour_gain": after_dice - before_dice
        >= minimum_dice_gain,
        "required_source_groups_present": set(group_gains) == set(required_groups),
        "source_group_nonregression": set(group_gains) == set(required_groups)
        and min(group_gains.values()) >= -maximum_group_drop,
    }
    decision = (
        "QUALIFY_SOURCE_ROUTED_FULL_VOLUME"
        if all(checks.values())
        else "KEEP_GLOBAL_CONTOUR"
    )
    return decision, metrics, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-dice-gain", type=float, default=0.002)
    parser.add_argument("--maximum-group-drop", type=float, default=0.002)
    parser.add_argument("--probability-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if min(
        args.minimum_dice_gain,
        args.maximum_group_drop,
        args.probability_tolerance,
    ) < 0:
        parser.error("gate tolerances must be nonnegative")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision, metrics, checks = decide(
        baseline,
        candidate,
        args.minimum_dice_gain,
        args.maximum_group_drop,
        args.probability_tolerance,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": sha256(args.baseline),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "metrics": metrics,
        "checks": checks,
        "thresholds": {
            "minimum_dice_gain": args.minimum_dice_gain,
            "maximum_group_drop": args.maximum_group_drop,
            "probability_tolerance": args.probability_tolerance,
        },
        "next_stage": (
            "matched source-routed full-volume exact-707 OOF; no public evaluation"
            if decision == "QUALIFY_SOURCE_ROUTED_FULL_VOLUME"
            else "none"
        ),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
