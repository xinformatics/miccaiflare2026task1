#!/usr/bin/env python3
"""Require untouched union-native fold-1 confirmation before expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from select_candidate_refiner_union_native_pilot import (
    EXPECTED_FAMILIES,
    validate_candidate,
)


def select(plan: dict, fold0: dict, baseline: dict, candidates: list[dict]) -> dict:
    if fold0.get("decision") != "CONFIRM_ON_UNTOUCHED_FOLD" or fold0.get("passed") is not True:
        raise ValueError("fold-0 union-native gate did not pass")
    if int(baseline.get("fold", -1)) != 1:
        raise ValueError("confirmation baseline is not untouched fold 1")
    if {candidate["family"] for candidate in candidates} != EXPECTED_FAMILIES:
        raise ValueError("confirmation requires all three fold-1 checkpoint families")
    gate = plan["future_untouched_fold1_gate_if_fold0_passes"]
    baseline_metrics = baseline["selected_metrics"]
    fold0_gains = fold0["selected_gains"]
    records = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        fold1_gains = {
            "case_balanced_segmentation_dice": metrics[
                "case_balanced_segmentation_dice"
            ]
            - float(baseline_metrics["case_balanced_segmentation_dice"]),
            "selection_score": metrics["selection_score"]
            - float(baseline_metrics["selection_score"]),
            "average_precision": metrics["average_precision"]
            - float(baseline_metrics["average_precision"]),
        }
        means = {
            name: float(np.mean([float(fold0_gains[name]), float(fold1_gains[name])]))
            for name in ("case_balanced_segmentation_dice", "selection_score", "average_precision")
        }
        checks = {
            "fold1_segmentation_gain": fold1_gains["case_balanced_segmentation_dice"]
            >= float(gate["fold1_case_balanced_segmentation_dice_gain_min"]),
            "fold1_selection_gain": fold1_gains["selection_score"]
            >= float(gate["fold1_selection_score_gain_min"]),
            "fold1_average_precision_regression": fold1_gains["average_precision"]
            >= -float(gate["average_precision_regression_max_each_fold"]),
            "two_fold_segmentation_gain": means["case_balanced_segmentation_dice"]
            >= float(gate["two_fold_mean_case_balanced_segmentation_dice_gain_min"]),
            "two_fold_selection_gain": means["selection_score"]
            >= float(gate["two_fold_mean_selection_score_gain_min"]),
        }
        records.append({
            **candidate,
            "fold1_gains": fold1_gains,
            "two_fold_mean_gains": means,
            "checks": checks,
            "passed": all(checks.values()),
        })
    passing = [record for record in records if record["passed"]]
    selected = (
        max(
            passing,
            key=lambda record: (
                record["two_fold_mean_gains"]["case_balanced_segmentation_dice"],
                record["two_fold_mean_gains"]["selection_score"],
                record["metrics"]["case_balanced_segmentation_dice"],
                record["family"],
            ),
        )
        if passing
        else max(records, key=lambda record: record["metrics"]["selection_score"])
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": "EXPAND_FIVE_FOLD" if passing else "KEEP_TRANSFER",
        "passed": bool(passing),
        "fold0_selected_family": fold0["selected_family"],
        "fold0_selected_checkpoint": fold0["selected_checkpoint"],
        "fold0_selected_gains": fold0_gains,
        "fold1_selected_family": selected["family"],
        "fold1_selected_checkpoint": selected["checkpoint"],
        "fold1_selected_metrics": selected["metrics"],
        "fold1_selected_gains": selected["fold1_gains"],
        "two_fold_mean_gains": selected["two_fold_mean_gains"],
        "fold1_baseline_family": baseline["selected_family"],
        "fold1_baseline_checkpoint": baseline["selected_checkpoint"],
        "fold1_baseline_metrics": baseline_metrics,
        "gate": gate,
        "candidate_records": records,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    fold0 = json.loads(args.fold0.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    index_hash = hashlib.sha256(args.index.read_bytes()).hexdigest()
    if baseline.get("evaluation_index_sha256") != index_hash:
        raise ValueError("fold-1 baseline and union-native index differ")
    candidates = []
    families = set()
    for value in args.checkpoint:
        family, separator, raw_path = value.partition("=")
        if not separator or family in families:
            raise ValueError("checkpoint must be unique FAMILY=PATH")
        families.add(family); path = Path(raw_path)
        candidates.append(validate_candidate(
            torch.load(path, map_location="cpu", weights_only=False),
            path, family, plan, index_hash, expected_fold=1,
        ))
    result = select(plan, fold0, baseline, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
