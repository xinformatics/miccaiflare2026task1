#!/usr/bin/env python3
"""Apply the predeclared union-native fold-0 gate against transfer weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch


EXPECTED_FAMILIES = {"composite", "segmentation", "classification"}


def validate_candidate(
    checkpoint: dict,
    path: Path,
    family: str,
    plan: dict,
    index_hash: str,
    expected_fold: int | None = None,
) -> dict:
    training = plan["future_training"]
    fold = int(training["first_fold"] if expected_fold is None else expected_fold)
    config = checkpoint.get("config", {})
    expected = {
        "fold": fold,
        "folds": int(training["folds"]),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "workers": int(training["workers"]),
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "base_channels": 32,
        "sampling_temperature": float(training["sampling_temperature"]),
        "seed": int(training["seed"]),
    }
    observed = {
        "fold": int(checkpoint.get("fold", -1)),
        "folds": int(checkpoint.get("folds", -1)),
        **{key: config.get(key) for key in expected if key not in ("fold", "folds")},
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{family} configuration mismatch: {mismatches}")
    if checkpoint.get("index_sha256") != index_hash:
        raise ValueError(f"{family} union-native index mismatch")
    metrics = checkpoint.get("metrics", {})
    values = {
        "case_balanced_segmentation_dice": float(
            metrics["case_balanced_segmentation_dice"]
        ),
        "selection_score": float(metrics["selection_score"]),
        "average_precision": float(metrics["average_precision"]),
    }
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
        raise ValueError(f"invalid {family} metric")
    if not math.isclose(
        float(metrics.get("segmentation_dice", -1)),
        values["case_balanced_segmentation_dice"],
        abs_tol=1e-8,
    ):
        raise ValueError(f"{family} checkpoint was not selected with case-balanced metrics")
    return {
        "family": family,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "epoch": int(checkpoint.get("epoch", -1)),
        "metrics": values,
    }


def select(plan: dict, baseline: dict, candidates: list[dict]) -> dict:
    if {candidate["family"] for candidate in candidates} != EXPECTED_FAMILIES:
        raise ValueError("union-native pilot requires all three checkpoint families")
    baseline_metrics = baseline["selected_metrics"]
    gate = plan["fold0_gate_all_required"]
    records = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        gains = {
            "case_balanced_segmentation_dice": metrics[
                "case_balanced_segmentation_dice"
            ]
            - float(baseline_metrics["case_balanced_segmentation_dice"]),
            "selection_score": metrics["selection_score"]
            - float(baseline_metrics["selection_score"]),
            "average_precision": metrics["average_precision"]
            - float(baseline_metrics["average_precision"]),
        }
        checks = {
            "segmentation_dice_gain": gains["case_balanced_segmentation_dice"]
            >= float(gate["case_balanced_segmentation_dice_gain_min"]),
            "selection_score_gain": gains["selection_score"]
            >= float(gate["case_balanced_selection_score_gain_min"]),
            "average_precision_regression": gains["average_precision"]
            >= -float(gate["average_precision_regression_max"]),
        }
        records.append({**candidate, "gains": gains, "checks": checks, "passed": all(checks.values())})
    passing = [record for record in records if record["passed"]]
    selected = (
        max(
            passing,
            key=lambda record: (
                record["metrics"]["case_balanced_segmentation_dice"],
                record["metrics"]["selection_score"],
                record["metrics"]["average_precision"],
                record["family"],
            ),
        )
        if passing
        else max(records, key=lambda record: record["metrics"]["selection_score"])
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": "CONFIRM_ON_UNTOUCHED_FOLD" if passing else "KEEP_TRANSFER",
        "passed": bool(passing),
        "selected_family": selected["family"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_metrics": selected["metrics"],
        "selected_gains": selected["gains"],
        "baseline_family": baseline["selected_family"],
        "baseline_checkpoint": baseline["selected_checkpoint"],
        "baseline_metrics": baseline_metrics,
        "gate": gate,
        "candidate_records": records,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    index_hash = hashlib.sha256(args.index.read_bytes()).hexdigest()
    if baseline.get("evaluation_index_sha256") != index_hash:
        raise ValueError("baseline and union-native index differ")
    candidates = []
    families = set()
    for value in args.checkpoint:
        family, separator, raw_path = value.partition("=")
        if not separator or family in families:
            raise ValueError("checkpoint must be unique FAMILY=PATH")
        families.add(family); path = Path(raw_path)
        candidates.append(
            validate_candidate(
                torch.load(path, map_location="cpu", weights_only=False),
                path, family, plan, index_hash,
            )
        )
    result = select(plan, baseline, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
