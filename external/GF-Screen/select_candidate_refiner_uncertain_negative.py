#!/usr/bin/env python3
"""Gate annotation-negative downweighting on one matched grouped fold."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from select_candidate_refiner_source_conditioning import file_sha256, summarize


def decide(
    baseline: dict[str, dict[str, float | int]],
    candidate: dict[str, dict[str, float | int]],
    minimum_classification_gain: float = 0.005,
    minimum_composite_gain: float = 0.002,
    maximum_segmentation_drop: float = 0.005,
) -> tuple[str, dict[str, float], dict[str, bool]]:
    gains = {
        "segmentation_dice": float(candidate["segmentation"]["segmentation_dice"])
        - float(baseline["segmentation"]["segmentation_dice"]),
        "selection_score": float(candidate["composite"]["selection_score"])
        - float(baseline["composite"]["selection_score"]),
        "average_precision": float(candidate["classification"]["average_precision"])
        - float(baseline["classification"]["average_precision"]),
    }
    gates = {
        "classification_gain": gains["average_precision"]
        >= minimum_classification_gain,
        "composite_gain": gains["selection_score"] >= minimum_composite_gain,
        "segmentation_preserved": gains["segmentation_dice"]
        >= -maximum_segmentation_drop,
    }
    decision = (
        "EXPAND_UNCERTAIN_NEGATIVE_025"
        if all(gates.values())
        else "KEEP_HARD_NEGATIVES"
    )
    return decision, gains, gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--candidate-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-classification-gain", type=float, default=0.005)
    parser.add_argument("--minimum-composite-gain", type=float, default=0.002)
    parser.add_argument("--maximum-segmentation-drop", type=float, default=0.005)
    args = parser.parse_args()
    if min(
        args.minimum_classification_gain,
        args.minimum_composite_gain,
        args.maximum_segmentation_drop,
    ) < 0:
        parser.error("gate tolerances must be nonnegative")
    baseline_history = json.loads(
        args.baseline_history.read_text(encoding="utf-8")
    )
    candidate_history = json.loads(
        args.candidate_history.read_text(encoding="utf-8")
    )
    if not isinstance(baseline_history, list) or not isinstance(
        candidate_history, list
    ):
        raise ValueError("training histories must be JSON lists")
    baseline = summarize(baseline_history)
    candidate = summarize(candidate_history)
    decision, gains, gates = decide(
        baseline,
        candidate,
        args.minimum_classification_gain,
        args.minimum_composite_gain,
        args.maximum_segmentation_drop,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "intervention": {
            "classification_negative_weight": 0.25,
            "interpretation": "annotation-negative candidates are uncertain/unlabeled",
        },
        "baseline_history": str(args.baseline_history.resolve()),
        "baseline_history_sha256": file_sha256(args.baseline_history),
        "candidate_history": str(args.candidate_history.resolve()),
        "candidate_history_sha256": file_sha256(args.candidate_history),
        "baseline": baseline,
        "candidate": candidate,
        "gains": gains,
        "gates": gates,
        "thresholds": {
            "minimum_classification_gain": args.minimum_classification_gain,
            "minimum_composite_gain": args.minimum_composite_gain,
            "maximum_segmentation_drop": args.maximum_segmentation_drop,
        },
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
