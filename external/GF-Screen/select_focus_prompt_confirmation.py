#!/usr/bin/env python3
"""Validate prompt-Focus folds 0/1 and apply the predeclared expansion gate."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from focus_validation_metrics import aggregate_positive_dice
from train_candidate_refiner import grouped_fold_rows


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def decide(records: list[dict], plan: dict) -> dict:
    if [int(record["fold"]) for record in records] != [0, 1]:
        raise ValueError("confirmation records must be folds 0 and 1")
    fold0_gate = plan["fold0_gate"]
    fold1_gate = plan["future_fold1_gate_if_fold0_passes"]
    fold0_pass = (
        float(records[0]["focus_case_balanced_dice"])
        >= float(fold0_gate["case_balanced_segmentation_dice_min"])
        and float(records[0]["negative_nonempty_rate"])
        <= float(fold0_gate["negative_nonempty_rate_max"])
        and float(records[0]["selection_score"])
        >= float(fold0_gate["selection_score_min"])
    )
    fold1_pass = (
        float(records[1]["focus_case_balanced_dice"])
        >= float(fold1_gate["case_balanced_segmentation_dice_min"])
        and float(records[1]["negative_nonempty_rate"])
        <= float(fold1_gate["negative_nonempty_rate_max"])
    )
    mean_gain = float(np.mean([float(record["dice_gain"]) for record in records]))
    mean_pass = mean_gain >= float(fold1_gate["two_fold_mean_gain_over_base32_min"])
    passed = fold0_pass and fold1_pass and mean_pass
    return {
        "decision": "EXPAND_FIVE_FOLD" if passed else "KEEP_BASE32",
        "passed": passed,
        "fold0_gate_passed": fold0_pass,
        "fold1_gate_passed": fold1_pass,
        "two_fold_mean_gain": mean_gain,
        "two_fold_mean_gate_passed": mean_pass,
    }


def summarize(plan: dict, index: Path, runs: list[Path], small: list[Path]) -> dict:
    if len(runs) != 2 or len(small) != 2:
        raise ValueError("exactly two prompt-Focus runs and small-model predictions required")
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    if digest != plan["index_sha256"]:
        raise ValueError("patch-index hash mismatch")
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    training = plan["training"]
    records = []
    for expected_fold, (run, small_path) in enumerate(zip(runs, small)):
        checkpoint_path = run / "checkpoint_best.pt"
        prediction_path = run / "validation_predictions_best.npz"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("fold", -1)) != expected_fold:
            raise ValueError("prompt-Focus fold order mismatch")
        if checkpoint.get("index_sha256") != digest:
            raise ValueError("prompt-Focus index hash mismatch")
        if checkpoint.get("pretrained_sha256") != plan["pretrained_sha256"]:
            raise ValueError("prompt-Focus pretrained hash mismatch")
        config = checkpoint.get("config", {})
        expected = {
            "epochs": int(training["epochs"]),
            "batch_size": int(training["batch_size"]),
            "workers": int(training["workers"]),
            "lr": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "loss_recipe": training["loss_recipe"],
            "sampling_temperature": float(training["sampling_temperature"]),
            "samples_per_epoch": int(training["samples_per_epoch"]),
            "freeze_encoder": True,
            "input_mode": "soft_seed",
            "folds": int(training["folds"]),
            "seed": int(training["seed"]),
        }
        mismatches = {
            key: {"expected": value, "observed": config.get(key)}
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"prompt-Focus configuration mismatch: {mismatches}")
        _, validation_rows = grouped_fold_rows(
            rows, int(training["folds"]), expected_fold, int(training["seed"])
        )
        focus_arrays = load_npz(prediction_path)
        focus_summary = aggregate_positive_dice(
            focus_arrays["positive_patch_id"],
            focus_arrays["segmentation_dice_grid"],
            focus_arrays["segmentation_thresholds"],
            validation_rows,
        )
        small_arrays = load_npz(small_path)
        small_summary = aggregate_positive_dice(
            small_arrays["positive_patch_id"],
            small_arrays["segmentation_dice_grid"],
            small_arrays["segmentation_thresholds"],
            validation_rows,
        )
        metrics = checkpoint["metrics"]
        focus_dice = float(focus_summary["case_balanced_dice"])
        if not np.isclose(focus_dice, float(metrics["segmentation_dice"]), atol=1e-7):
            raise ValueError("prompt-Focus checkpoint/prediction Dice mismatch")
        small_dice = float(small_summary["case_balanced_dice"])
        records.append(
            {
                "fold": expected_fold,
                "focus_run": str(run.resolve()),
                "focus_checkpoint": str(checkpoint_path.resolve()),
                "focus_prediction": str(prediction_path.resolve()),
                "small_prediction": str(small_path.resolve()),
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "focus_case_balanced_dice": focus_dice,
                "small_case_balanced_dice": small_dice,
                "dice_gain": focus_dice - small_dice,
                "negative_nonempty_rate": float(metrics["negative_nonempty_rate"]),
                "selection_score": float(metrics["selection_score"]),
                "positive_cases": int(focus_summary["positive_cases"]),
            }
        )
    result = decide(records, plan)
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "fold_records": records,
        **result,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--focus-run", type=Path, action="append", required=True)
    parser.add_argument("--small-prediction", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        json.loads(args.plan.read_text(encoding="utf-8")),
        args.index,
        args.focus_run,
        args.small_prediction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
