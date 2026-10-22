#!/usr/bin/env python3
"""Apply the predeclared, training-only base-48 capacity-pilot gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path

import torch


METRICS = (
    "segmentation_dice",
    "average_precision",
    "selection_score",
)


def select(plan: dict, checkpoint: dict, checkpoint_path: Path) -> dict:
    candidate = plan["candidate"]
    config = checkpoint.get("config", {})
    expected = {
        "base_channels": int(candidate["base_channels"]),
        "epochs": int(candidate["epochs"]),
        "batch_size": int(candidate["batch_size"]),
        "lr": float(candidate["learning_rate"]),
        "weight_decay": float(candidate["weight_decay"]),
        "sampling_temperature": float(candidate["sampling_temperature"]),
        "seed": int(candidate["seed"]),
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"training configuration mismatch: {mismatches}")
    if int(checkpoint.get("fold", -1)) != 0 or int(checkpoint.get("folds", -1)) != 5:
        raise ValueError("pilot must be grouped fold 0 of 5")
    if checkpoint.get("index_sha256") != plan["index_sha256"]:
        raise ValueError("patch-index hash mismatch")
    if int(checkpoint.get("epoch", -1)) > expected["epochs"]:
        raise ValueError("checkpoint epoch exceeds the predeclared pilot")

    metrics = checkpoint.get("metrics", {})
    if metrics.get("aggregation") != "patches weighted to equal total weight per case":
        raise ValueError("checkpoint does not use case-balanced selection")
    observed = {name: float(metrics[name]) for name in METRICS}
    if not all(math.isfinite(value) for value in observed.values()):
        raise ValueError("non-finite pilot metric")
    gate = plan["promotion_gate"]
    checks = {
        "segmentation_dice": observed["segmentation_dice"]
        >= float(gate["case_balanced_segmentation_dice_min"]),
        "average_precision": observed["average_precision"]
        >= float(gate["case_balanced_average_precision_min"]),
        "selection_score": observed["selection_score"]
        >= float(gate["case_balanced_selection_score_min"]),
    }
    passed = all(checks.values())
    return {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "decision": "CONFIRM_ON_UNTOUCHED_FOLD" if passed else "KEEP_BASE32",
        "passed": passed,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "index_sha256": checkpoint["index_sha256"],
        "metrics": observed,
        "gate": {
            "segmentation_dice_min": float(
                gate["case_balanced_segmentation_dice_min"]
            ),
            "average_precision_min": float(
                gate["case_balanced_average_precision_min"]
            ),
            "selection_score_min": float(
                gate["case_balanced_selection_score_min"]
            ),
        },
        "checks": checks,
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
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = select(plan, checkpoint, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
