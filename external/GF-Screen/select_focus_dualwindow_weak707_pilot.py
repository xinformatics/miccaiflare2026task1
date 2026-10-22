#!/usr/bin/env python3
"""Apply the predeclared local-only gate for the 707-case dual-window Focus pilot."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--focus-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    focus = torch.load(args.focus_checkpoint, map_location="cpu", weights_only=False)
    baseline = torch.load(
        args.baseline_checkpoint, map_location="cpu", weights_only=False
    )
    training = plan["training"]
    expected = {
        "fold": 0,
        "folds": 5,
        "epochs": 10,
        "batch_size": 8,
        "workers": 2,
        "lr": 3e-5,
        "weight_decay": 1e-5,
        "loss_recipe": "release_dice_ce",
        "sampling_temperature": 0.5,
        "samples_per_epoch": 2048,
        "freeze_encoder": True,
        "input_mode": "soft_lung_seed",
        "segmentation_threshold_max": 0.95,
        "seed": 20260823,
    }
    mismatches = {
        key: {"expected": value, "observed": focus.get("config", {}).get(key)}
        for key, value in expected.items()
        if focus.get("config", {}).get(key) != value
    }
    if mismatches:
        raise ValueError(f"Focus training configuration mismatch: {mismatches}")
    if training != expected:
        raise ValueError("plan training recipe does not match the frozen selector")
    index_hash = plan["index_sha256"]
    if focus.get("index_sha256") != index_hash:
        raise ValueError("Focus patch-index hash mismatch")
    if baseline.get("index_sha256") != index_hash:
        raise ValueError("B32 patch-index hash mismatch")
    if int(baseline.get("fold", -1)) != 0 or int(baseline.get("folds", -1)) != 5:
        raise ValueError("B32 checkpoint is not grouped fold 0/5")
    if int(baseline.get("config", {}).get("base_channels", -1)) != 32:
        raise ValueError("comparison checkpoint is not the B32 baseline")
    if not math.isclose(
        float(baseline.get("metrics", {}).get("segmentation_dice", float("nan"))),
        float(baseline.get("best_segmentation", float("nan"))),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("B32 checkpoint was not selected for segmentation")
    if "soft_lung_seed" not in str(focus.get("architecture", "")):
        raise ValueError("Focus checkpoint is not the three-channel architecture")
    if focus.get("pretrained_sha256") != plan["pretrained_sha256"]:
        raise ValueError("released Focus checkpoint hash mismatch")
    preflight_path = Path(plan["preflight"])
    if sha256(preflight_path) != plan["preflight_sha256"]:
        raise ValueError("Focus checkpoint preflight changed after planning")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise ValueError("Focus checkpoint preflight did not pass")
    if preflight.get("checkpoint_sha256") != plan["pretrained_sha256"]:
        raise ValueError("Focus preflight/checkpoint identity mismatch")
    if sha256(args.baseline_checkpoint) != plan["baseline_checkpoint_sha256"]:
        raise ValueError("B32 comparison checkpoint changed after planning")

    focus_metrics = focus["metrics"]
    baseline_metrics = baseline["metrics"]
    history = focus.get("history", [])
    if not history or int(history[0].get("epoch", -1)) != 0:
        raise ValueError("Focus checkpoint lacks its initialization baseline")
    released_metrics = history[0]
    observed = {
        "focus_case_dice": float(focus_metrics["case_balanced_segmentation_dice"]),
        "focus_cohort_dice": float(
            focus_metrics["cohort_balanced_segmentation_dice"]
        ),
        "focus_negative_nonempty_rate": float(
            focus_metrics["negative_nonempty_rate"]
        ),
        "b32_case_dice": float(baseline_metrics["segmentation_dice"]),
        "b32_cohort_dice": float(
            baseline_metrics["cohort_balanced_segmentation_dice"]
        ),
        "released_case_dice": float(
            released_metrics["case_balanced_segmentation_dice"]
        ),
    }
    if not all(math.isfinite(value) for value in observed.values()):
        raise ValueError("non-finite pilot metric")
    observed["case_gain_over_b32"] = (
        observed["focus_case_dice"] - observed["b32_case_dice"]
    )
    observed["cohort_gain_over_b32"] = (
        observed["focus_cohort_dice"] - observed["b32_cohort_dice"]
    )
    observed["case_gain_over_released"] = (
        observed["focus_case_dice"] - observed["released_case_dice"]
    )
    gate = plan["fold0_gate"]
    checks = {
        "case_gain_over_b32": observed["case_gain_over_b32"]
        >= float(gate["minimum_case_dice_gain_over_b32"]),
        "cohort_gain_over_b32": observed["cohort_gain_over_b32"]
        >= float(gate["minimum_cohort_dice_gain_over_b32"]),
        "case_gain_over_released": observed["case_gain_over_released"]
        >= float(gate["minimum_case_dice_gain_over_released"]),
        "negative_nonempty_rate": observed["focus_negative_nonempty_rate"]
        <= float(gate["maximum_negative_nonempty_rate"]),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "decision": "PROMOTE_TO_FIVEFOLD" if passed else "KEEP_B32",
        "passed": passed,
        "plan": str(args.plan.resolve()),
        "focus_checkpoint": str(args.focus_checkpoint.resolve()),
        "focus_checkpoint_sha256": sha256(args.focus_checkpoint),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
        "metrics": observed,
        "gate": gate,
        "checks": checks,
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
