#!/usr/bin/env python3
"""Gate a source-conditioned refiner using one matched held-out fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_metric(record: dict[str, object], key: str) -> float:
    value = float(record[key])
    if not __import__("math").isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def best_record(
    history: list[dict[str, object]], key: str
) -> dict[str, object]:
    if not history:
        raise ValueError("training history is empty")
    required = {
        "epoch",
        "segmentation_dice",
        "average_precision",
        "selection_score",
    }
    for record in history:
        missing = required - set(record)
        if missing:
            raise ValueError(f"training history record misses {sorted(missing)}")
        for metric in required - {"epoch"}:
            finite_metric(record, metric)
    tie_breakers = {
        "segmentation_dice": ("segmentation_dice", "selection_score", "average_precision"),
        "selection_score": ("selection_score", "segmentation_dice", "average_precision"),
        "average_precision": ("average_precision", "selection_score", "segmentation_dice"),
    }
    if key not in tie_breakers:
        raise ValueError(f"unsupported selection key: {key}")
    return max(
        history,
        key=lambda record: tuple(
            finite_metric(record, metric) for metric in tie_breakers[key]
        )
        + (-int(record["epoch"]),),
    )


def summarize(history: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for name, key in (
        ("segmentation", "segmentation_dice"),
        ("composite", "selection_score"),
        ("classification", "average_precision"),
    ):
        record = best_record(history, key)
        output[name] = {
            "epoch": int(record["epoch"]),
            "segmentation_dice": finite_metric(record, "segmentation_dice"),
            "average_precision": finite_metric(record, "average_precision"),
            "selection_score": finite_metric(record, "selection_score"),
        }
    return output


def decide(
    baseline: dict[str, dict[str, float | int]],
    candidate: dict[str, dict[str, float | int]],
    minimum_segmentation_gain: float = 0.02,
    minimum_composite_gain: float = 0.005,
    maximum_classification_drop: float = 0.01,
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
        "segmentation_gain": gains["segmentation_dice"] >= minimum_segmentation_gain,
        "composite_gain": gains["selection_score"] >= minimum_composite_gain,
        "classification_preserved": gains["average_precision"]
        >= -maximum_classification_drop,
    }
    return (
        "EXPAND_SOURCE_CONDITIONING" if all(gates.values()) else "KEEP_UNCONDITIONED",
        gains,
        gates,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--candidate-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-segmentation-gain", type=float, default=0.02)
    parser.add_argument("--minimum-composite-gain", type=float, default=0.005)
    parser.add_argument("--maximum-classification-drop", type=float, default=0.01)
    args = parser.parse_args()
    if (
        args.minimum_segmentation_gain < 0
        or args.minimum_composite_gain < 0
        or args.maximum_classification_drop < 0
    ):
        parser.error("gate tolerances must be nonnegative")
    baseline_history = json.loads(args.baseline_history.read_text(encoding="utf-8"))
    candidate_history = json.loads(args.candidate_history.read_text(encoding="utf-8"))
    if not isinstance(baseline_history, list) or not isinstance(candidate_history, list):
        raise ValueError("training histories must be JSON lists")
    baseline = summarize(baseline_history)
    candidate = summarize(candidate_history)
    decision, gains, gates = decide(
        baseline,
        candidate,
        args.minimum_segmentation_gain,
        args.minimum_composite_gain,
        args.maximum_classification_drop,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "baseline_history": str(args.baseline_history.resolve()),
        "baseline_history_sha256": file_sha256(args.baseline_history),
        "candidate_history": str(args.candidate_history.resolve()),
        "candidate_history_sha256": file_sha256(args.candidate_history),
        "baseline": baseline,
        "candidate": candidate,
        "gains": gains,
        "gates": gates,
        "thresholds": {
            "minimum_segmentation_gain": args.minimum_segmentation_gain,
            "minimum_composite_gain": args.minimum_composite_gain,
            "maximum_classification_drop": args.maximum_classification_drop,
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
