#!/usr/bin/env python3
"""Apply the predeclared training-only prompt-conditioned Focus pilot gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path

import torch


def select(plan: dict, checkpoint: dict, checkpoint_path: Path) -> dict:
    training = plan["training"]
    config = checkpoint.get("config", {})
    expected = {
        "fold": int(training["fold"]),
        "folds": int(training["folds"]),
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
        "seed": int(training["seed"]),
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"training configuration mismatch: {mismatches}")
    if checkpoint.get("index_sha256") != plan["index_sha256"]:
        raise ValueError("patch-index hash mismatch")
    if checkpoint.get("pretrained_sha256") != plan["pretrained_sha256"]:
        raise ValueError("pretrained checkpoint hash mismatch")
    if "soft_seed" not in str(checkpoint.get("architecture", "")):
        raise ValueError("checkpoint is not prompt-conditioned Focus")

    metrics = checkpoint.get("metrics", {})
    names = (
        "case_balanced_segmentation_dice",
        "negative_nonempty_rate",
        "selection_score",
    )
    observed = {name: float(metrics[name]) for name in names}
    if not all(math.isfinite(value) for value in observed.values()):
        raise ValueError("non-finite prompt-Focus metric")
    if float(metrics.get("segmentation_dice", float("nan"))) != observed[
        "case_balanced_segmentation_dice"
    ]:
        raise ValueError("checkpoint selection is not case-balanced")
    gate = plan["fold0_gate"]
    checks = {
        "segmentation_dice": observed["case_balanced_segmentation_dice"]
        >= float(gate["case_balanced_segmentation_dice_min"]),
        "negative_nonempty_rate": observed["negative_nonempty_rate"]
        <= float(gate["negative_nonempty_rate_max"]),
        "selection_score": observed["selection_score"]
        >= float(gate["selection_score_min"]),
    }
    passed = all(checks.values())
    return {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "decision": "CONFIRM_ON_UNTOUCHED_FOLD" if passed else "KEEP_BASE32",
        "passed": passed,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "metrics": observed,
        "checks": checks,
        "gate": gate,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(
        json.loads(args.plan.read_text(encoding="utf-8")),
        torch.load(args.checkpoint, map_location="cpu", weights_only=False),
        args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
