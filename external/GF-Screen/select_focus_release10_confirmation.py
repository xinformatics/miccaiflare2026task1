#!/usr/bin/env python3
"""Apply the predeclared two-fold gate to the extended Focus pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from focus_validation_metrics import aggregate_positive_dice
from train_candidate_refiner import grouped_fold_rows


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def summarize(
    plan: dict,
    index: Path,
    focus_runs: list[Path],
    small_predictions: list[Path],
) -> dict:
    if len(focus_runs) != 2 or len(small_predictions) != 2:
        raise ValueError("exactly two Focus runs and two small predictions are required")
    index_digest = hashlib.sha256(index.read_bytes()).hexdigest()
    fixed = plan["fixed_recipe"]
    if index_digest != fixed["index_sha256"]:
        raise ValueError("confirmation index digest mismatch")
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for run, small_path in zip(focus_runs, small_predictions):
        checkpoint_path = run / "checkpoint_best.pt"
        prediction_path = run / "validation_predictions_best.npz"
        for required in (checkpoint_path, prediction_path, small_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        fold = int(checkpoint.get("fold", -1))
        if fold not in (0, 1):
            raise ValueError(f"unexpected confirmation fold: {fold}")
        if checkpoint.get("index_sha256") != index_digest:
            raise ValueError("Focus checkpoint index mismatch")
        if checkpoint.get("pretrained_sha256") != fixed["pretrained_sha256"]:
            raise ValueError("Focus checkpoint pretrained mismatch")
        config = checkpoint.get("config", {})
        config_keys = {
            "epochs": "epochs",
            "batch_size": "batch_size",
            "loss_recipe": "loss_recipe",
            "lr": "learning_rate",
            "weight_decay": "weight_decay",
            "sampling_temperature": "sampling_temperature",
            "samples_per_epoch": "samples_per_epoch",
            "freeze_encoder": "freeze_encoder",
            "folds": "folds",
            "seed": "seed",
        }
        for config_key, plan_key in config_keys.items():
            if config.get(config_key) != fixed[plan_key]:
                raise ValueError(f"Focus configuration mismatch: {config_key}")
        _, validation_rows = grouped_fold_rows(
            rows, int(fixed["folds"]), fold, int(fixed["seed"])
        )
        focus = load_prediction(prediction_path)
        small = load_prediction(small_path)
        focus_summary = aggregate_positive_dice(
            focus["positive_patch_id"],
            focus["segmentation_dice_grid"],
            focus["segmentation_thresholds"],
            validation_rows,
        )
        small_summary = aggregate_positive_dice(
            small["positive_patch_id"],
            small["segmentation_dice_grid"],
            small["segmentation_thresholds"],
            validation_rows,
        )
        metrics = checkpoint["metrics"]
        if not np.isclose(
            float(metrics["segmentation_dice"]),
            float(focus_summary["case_balanced_dice"]),
            atol=1e-7,
        ):
            raise ValueError("Focus checkpoint and prediction Dice mismatch")
        if not np.isclose(
            float(metrics["segmentation_threshold"]),
            float(focus_summary["threshold"]),
            atol=1e-7,
        ):
            raise ValueError("Focus checkpoint and prediction threshold mismatch")
        focus_dice = float(focus_summary["case_balanced_dice"])
        small_dice = float(small_summary["case_balanced_dice"])
        records.append(
            {
                "fold": fold,
                "focus_run": str(run.resolve()),
                "focus_checkpoint": str(checkpoint_path.resolve()),
                "focus_prediction": str(prediction_path.resolve()),
                "small_prediction": str(small_path.resolve()),
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "focus_case_balanced_dice": focus_dice,
                "small_case_balanced_dice": small_dice,
                "dice_gain": focus_dice - small_dice,
                "focus_negative_nonempty_rate": float(
                    metrics["negative_nonempty_rate"]
                ),
                "positive_cases": int(focus_summary["positive_cases"]),
            }
        )
    records.sort(key=lambda record: int(record["fold"]))
    if [record["fold"] for record in records] != [0, 1]:
        raise ValueError("confirmation folds must be exactly 0 and 1")
    rule = plan["promotion_rule_fixed_before_fold1_training"]
    mean_gain = float(np.mean([record["dice_gain"] for record in records]))
    fold1_gain = float(records[1]["dice_gain"])
    negative_ok = all(
        float(record["focus_negative_nonempty_rate"])
        <= float(rule["maximum_negative_nonempty_rate_each_fold"])
        for record in records
    )
    qualifies = (
        fold1_gain >= float(rule["minimum_fold1_dice_gain"])
        and mean_gain >= float(rule["minimum_two_fold_mean_dice_gain"])
        and negative_ok
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "plan": plan,
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "fold_records": records,
        "fold1_dice_gain": fold1_gain,
        "two_fold_mean_dice_gain": mean_gain,
        "negative_gate_passed": negative_ok,
        "decision": "EXPAND_FIVE_FOLD" if qualifies else "KEEP_SMALL_REFINER",
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
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    payload = summarize(plan, args.index, args.focus_run, args.small_prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
