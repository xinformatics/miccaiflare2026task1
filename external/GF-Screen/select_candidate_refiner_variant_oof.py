#!/usr/bin/env python3
"""Gate a controlled exact-707 loss/input ablation after five-fold patch OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(summary: dict) -> dict[str, float]:
    classification = summary["classification"]
    segmentation = summary["segmentation"]
    ap = float(classification["average_precision"])
    dice = float(segmentation["dice"])
    values = {
        "average_precision": ap,
        "precision": float(classification["precision"]),
        "recall": float(classification["recall"]),
        "segmentation_dice": dice,
        "composite": (ap + dice) / 2.0,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite OOF metric")
    return values


def eligible(summary: dict) -> bool:
    classification = summary["classification"]
    return bool(
        float(classification["average_precision"])
        >= float(classification["prevalence"]) + 0.15
        and float(classification["precision"]) >= 0.50
        and float(classification["recall"]) >= 0.20
    )


def decide(
    baseline: dict,
    candidate: dict,
    minimum_segmentation_gain: float = 0.005,
    minimum_composite_gain: float = 0.002,
    maximum_classification_drop: float = 0.010,
) -> tuple[str, dict[str, float], dict[str, bool]]:
    before, after = metrics(baseline), metrics(candidate)
    gains = {key: after[key] - before[key] for key in before}
    checks = {
        "candidate_classifier_eligible": eligible(candidate),
        "segmentation_gain": gains["segmentation_dice"]
        >= minimum_segmentation_gain,
        "composite_gain": gains["composite"] >= minimum_composite_gain,
        "classification_preserved": gains["average_precision"]
        >= -maximum_classification_drop,
    }
    decision = (
        "QUALIFY_VARIANT_FULL_VOLUME" if all(checks.values()) else "KEEP_PRIMARY"
    )
    return decision, gains, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--variant", choices=("source_condition", "small_lesion"), required=True)
    parser.add_argument("--minimum-segmentation-gain", type=float, default=0.005)
    parser.add_argument("--minimum-composite-gain", type=float, default=0.002)
    parser.add_argument("--maximum-classification-drop", type=float, default=0.010)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.minimum_segmentation_gain,
        args.minimum_composite_gain,
        args.maximum_classification_drop,
    ) < 0:
        parser.error("gate tolerances must be nonnegative")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision, gains, checks = decide(
        baseline,
        candidate,
        args.minimum_segmentation_gain,
        args.minimum_composite_gain,
        args.maximum_classification_drop,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "variant": args.variant,
        "decision": decision,
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": sha256(args.baseline),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "baseline_metrics": metrics(baseline),
        "candidate_metrics": metrics(candidate),
        "gains": gains,
        "checks": checks,
        "thresholds": {
            "minimum_segmentation_gain": args.minimum_segmentation_gain,
            "minimum_composite_gain": args.minimum_composite_gain,
            "maximum_classification_drop": args.maximum_classification_drop,
        },
        "next_stage": (
            "matched exact-707 full-volume OOF only"
            if decision == "QUALIFY_VARIANT_FULL_VOLUME"
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
