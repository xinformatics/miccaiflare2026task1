#!/usr/bin/env python3
"""Gate discriminative-LR Focus adaptation on one grouped held-out fold."""

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
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-batch-size", type=int, choices=(4, 8), default=4
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    candidate = torch.load(
        args.candidate_checkpoint, map_location="cpu", weights_only=False
    )
    baseline = torch.load(
        args.baseline_checkpoint, map_location="cpu", weights_only=False
    )
    frozen = torch.load(
        args.frozen_checkpoint, map_location="cpu", weights_only=False
    )
    training = plan["training"]
    expected = {
        "fold": 0,
        "folds": 5,
        "epochs": 10,
        "batch_size": args.expected_batch_size,
        "workers": 2,
        "lr": 3e-5,
        "encoder_lr_scale": 0.1,
        "weight_decay": 1e-5,
        "loss_recipe": "release_dice_ce",
        "sampling_temperature": 0.5,
        "samples_per_epoch": 2048,
        "freeze_encoder": False,
        "input_mode": "soft_lung_seed",
        "segmentation_threshold_max": 0.95,
        "seed": 20260823,
    }
    if training != expected:
        raise ValueError("plan training recipe does not match the frozen selector")
    config = candidate.get("config", {})
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Focus training configuration mismatch: {mismatches}")
    index_hash = plan["index_sha256"]
    for name, checkpoint in (
        ("candidate", candidate),
        ("baseline", baseline),
        ("frozen", frozen),
    ):
        if checkpoint.get("index_sha256") != index_hash:
            raise ValueError(f"{name} patch-index hash mismatch")
        if int(checkpoint.get("fold", -1)) != 0 or int(
            checkpoint.get("folds", -1)
        ) != 5:
            raise ValueError(f"{name} checkpoint is not grouped fold 0/5")
    if candidate.get("pretrained_sha256") != plan["pretrained_sha256"]:
        raise ValueError("released Focus checkpoint mismatch")
    if sha256(args.baseline_checkpoint) != plan["baseline_checkpoint_sha256"]:
        raise ValueError("B32 comparison checkpoint changed after planning")
    if sha256(args.frozen_checkpoint) != plan["frozen_checkpoint_sha256"]:
        raise ValueError("frozen Focus comparison checkpoint changed after planning")
    if sha256(Path(plan["frozen_decision"])) != plan["frozen_decision_sha256"]:
        raise ValueError("frozen Focus decision changed after planning")
    if int(baseline.get("config", {}).get("base_channels", -1)) != 32:
        raise ValueError("comparison checkpoint is not the B32 baseline")
    if not math.isclose(
        float(baseline.get("metrics", {}).get("segmentation_dice", float("nan"))),
        float(baseline.get("best_segmentation", float("nan"))),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("B32 checkpoint was not selected for segmentation")
    frozen_config = frozen.get("config", {})
    if frozen_config.get("freeze_encoder") is not True or frozen_config.get(
        "input_mode"
    ) != "soft_lung_seed":
        raise ValueError("frozen comparator is not the dual-window Focus pilot")
    if "soft_lung_seed" not in str(candidate.get("architecture", "")):
        raise ValueError("candidate is not the dual-window Focus architecture")
    history = candidate.get("history", [])
    if not history or int(history[0].get("epoch", -1)) != 0:
        raise ValueError("candidate lacks its released initialization baseline")

    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    frozen_metrics = frozen["metrics"]
    observed = {
        "candidate_case_dice": float(
            candidate_metrics["case_balanced_segmentation_dice"]
        ),
        "candidate_cohort_dice": float(
            candidate_metrics["cohort_balanced_segmentation_dice"]
        ),
        "candidate_negative_nonempty_rate": float(
            candidate_metrics["negative_nonempty_rate"]
        ),
        "b32_case_dice": float(baseline_metrics["segmentation_dice"]),
        "b32_cohort_dice": float(
            baseline_metrics["cohort_balanced_segmentation_dice"]
        ),
        "frozen_case_dice": float(
            frozen_metrics["case_balanced_segmentation_dice"]
        ),
        "frozen_cohort_dice": float(
            frozen_metrics["cohort_balanced_segmentation_dice"]
        ),
    }
    if not all(math.isfinite(value) for value in observed.values()):
        raise ValueError("non-finite pilot metric")
    for comparator in ("b32", "frozen"):
        observed[f"case_gain_over_{comparator}"] = (
            observed["candidate_case_dice"] - observed[f"{comparator}_case_dice"]
        )
        observed[f"cohort_gain_over_{comparator}"] = (
            observed["candidate_cohort_dice"]
            - observed[f"{comparator}_cohort_dice"]
        )
    gate = plan["fold0_gate"]
    checks = {
        "case_gain_over_b32": observed["case_gain_over_b32"]
        >= float(gate["minimum_case_dice_gain_over_b32"]),
        "cohort_gain_over_b32": observed["cohort_gain_over_b32"]
        >= float(gate["minimum_cohort_dice_gain_over_b32"]),
        "case_gain_over_frozen": observed["case_gain_over_frozen"]
        >= float(gate["minimum_case_dice_gain_over_frozen"]),
        "cohort_gain_over_frozen": observed["cohort_gain_over_frozen"]
        >= float(gate["minimum_cohort_dice_gain_over_frozen"]),
        "negative_nonempty_rate": observed["candidate_negative_nonempty_rate"]
        <= float(gate["maximum_negative_nonempty_rate"]),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "decision": "PROMOTE_TO_FIVEFOLD" if passed else "KEEP_FROZEN_OR_B32",
        "passed": passed,
        "plan": str(args.plan.resolve()),
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "candidate_checkpoint_sha256": sha256(args.candidate_checkpoint),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
        "frozen_checkpoint": str(args.frozen_checkpoint.resolve()),
        "frozen_checkpoint_sha256": sha256(args.frozen_checkpoint),
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
