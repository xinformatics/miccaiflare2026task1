#!/usr/bin/env python3
"""Gate decoupled confidence/contour checkpoints for full-volume OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(summary: dict) -> dict[str, float]:
    return {
        "average_precision": float(summary["classification"]["average_precision"]),
        "precision": float(summary["classification"]["precision"]),
        "recall": float(summary["classification"]["recall"]),
        "segmentation_dice": float(summary["segmentation"]["dice"]),
    }


def classification_eligible(summary: dict) -> bool:
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
    minimum_primary_gain: float = 0.002,
    maximum_metric_drop: float = 1e-6,
) -> tuple[str, dict[str, float], dict[str, bool]]:
    before, after = metrics(baseline), metrics(candidate)
    gains = {key: after[key] - before[key] for key in before}
    checks = {
        "candidate_classifier_eligible": classification_eligible(candidate),
        "average_precision_nonregression": gains["average_precision"]
        >= -maximum_metric_drop,
        "segmentation_nonregression": gains["segmentation_dice"]
        >= -maximum_metric_drop,
        "meaningful_primary_gain": max(
            gains["average_precision"], gains["segmentation_dice"]
        )
        >= minimum_primary_gain,
    }
    decision = (
        "QUALIFY_DECOUPLED_FULL_VOLUME"
        if all(checks.values())
        else "KEEP_COUPLED"
    )
    return decision, gains, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-primary-gain", type=float, default=0.002)
    parser.add_argument("--maximum-metric-drop", type=float, default=1e-6)
    args = parser.parse_args()
    if args.minimum_primary_gain < 0 or args.maximum_metric_drop < 0:
        parser.error("gate tolerances must be nonnegative")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision, gains, checks = decide(
        baseline,
        candidate,
        args.minimum_primary_gain,
        args.maximum_metric_drop,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
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
            "minimum_primary_gain": args.minimum_primary_gain,
            "maximum_metric_drop": args.maximum_metric_drop,
        },
        "next_stage": (
            "matched full-volume OOF only; no public evaluation"
            if decision == "QUALIFY_DECOUPLED_FULL_VOLUME"
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
