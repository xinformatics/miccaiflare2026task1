#!/usr/bin/env python3
"""Gate a source-specific B32/Focus five-fold expansion on held-out fold 0.

This is an escape hatch for a heterogeneous contour model: a global Focus
average may regress even when Focus is materially better for weak or lung-cell
proposals.  The comparison uses the identical grouped held-out fold and keeps
the mature proposal families on B32.  It is local-only and cannot read public
or hidden images or create a submission.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from candidate_refiner_sampling import semantic_source_group
from focus_validation_metrics import aggregate_positive_dice
from train_candidate_refiner import grouped_fold_rows


EXPECTED_TRAINING = {
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
EXPECTED_GATE = {
    "target_groups": ["weak_sources", "lung_cells"],
    "minimum_case_dice_gain_over_b32": 0.02,
    "minimum_cohort_dice_gain_over_b32": 0.01,
    "maximum_focus_negative_nonempty_rate": 0.95,
    "threshold_selection": "separate group-optimal held-out case-balanced Dice",
    "mature_route": "b32",
}
EXPECTED_SOURCES = {
    "mature": {"epoch105", "currentgf"},
    "weak_sources": {"autopet_weak", "lndb_weak"},
    "lung_cells": {f"lung_cell{index}" for index in range(8)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contour_group(source_name: str) -> str:
    source = semantic_source_group(source_name)
    if source in {"epoch105", "currentgf"}:
        return "mature"
    if source in {"autopet_weak", "lndb_weak"}:
        return "weak_sources"
    return source


def _close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=2e-6):
        raise ValueError(
            f"{label} does not reproduce checkpoint metric: "
            f"observed={observed} expected={expected}"
        )


def _load_prediction(path: Path, require_negative: bool) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "positive_patch_id",
            "segmentation_thresholds",
            "segmentation_dice_grid",
        }
        if require_negative:
            required.update({"negative_patch_id", "negative_nonempty_grid"})
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"prediction archive lacks {sorted(missing)}: {path}")
        result: dict[str, object] = {
            "positive_ids": [str(value) for value in archive["positive_patch_id"]],
            "thresholds": archive["segmentation_thresholds"].astype(np.float64),
            "dice_grid": archive["segmentation_dice_grid"].astype(np.float64),
        }
        if require_negative:
            result["negative_ids"] = [
                str(value) for value in archive["negative_patch_id"]
            ]
            result["negative_grid"] = archive["negative_nonempty_grid"].astype(
                np.int64
            )
    positive_ids = result["positive_ids"]
    thresholds = result["thresholds"]
    dice_grid = result["dice_grid"]
    assert isinstance(positive_ids, list)
    assert isinstance(thresholds, np.ndarray)
    assert isinstance(dice_grid, np.ndarray)
    if len(positive_ids) != len(set(positive_ids)):
        raise ValueError(f"duplicate positive patch IDs: {path}")
    if dice_grid.shape != (len(positive_ids), len(thresholds)):
        raise ValueError(f"invalid positive Dice grid: {path}")
    if require_negative:
        negative_ids = result["negative_ids"]
        negative_grid = result["negative_grid"]
        assert isinstance(negative_ids, list)
        assert isinstance(negative_grid, np.ndarray)
        if len(negative_ids) != len(set(negative_ids)):
            raise ValueError(f"duplicate negative patch IDs: {path}")
        if negative_grid.shape != (len(negative_ids), len(thresholds)):
            raise ValueError(f"invalid negative activation grid: {path}")
        if np.any((negative_grid != 0) & (negative_grid != 1)):
            raise ValueError(f"negative activation grid is not binary: {path}")
    return result


def select(
    plan_path: Path,
    focus_checkpoint_path: Path,
    focus_prediction_path: Path,
    baseline_checkpoint_path: Path,
    baseline_prediction_path: Path,
) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("training") != EXPECTED_TRAINING:
        raise ValueError("plan training recipe does not match the frozen selector")
    if plan.get("mixed_source_gate") != EXPECTED_GATE:
        raise ValueError("plan mixed-source gate does not match the frozen selector")
    index_path = Path(plan["index"])
    if sha256(index_path) != plan.get("index_sha256"):
        raise ValueError("patch index changed after planning")
    if sha256(baseline_checkpoint_path) != plan.get(
        "baseline_checkpoint_sha256"
    ):
        raise ValueError("B32 checkpoint changed after planning")
    if sha256(baseline_prediction_path) != plan.get(
        "baseline_prediction_sha256"
    ):
        raise ValueError("B32 prediction archive changed after planning")

    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    cases = {str(row["case"]) for row in rows}
    if len(cases) != 707:
        raise ValueError(f"pilot index is not exact 707-case: {len(cases)}")
    _, held_out = grouped_fold_rows(rows, 5, 0, 20260823)
    expected_positive = {
        str(row["patch_id"]) for row in held_out if int(row["label"]) == 1
    }
    expected_negative = {
        str(row["patch_id"]) for row in held_out if int(row["label"]) == 0
    }
    if not expected_positive or not expected_negative:
        raise ValueError("fold 0 does not contain both labels")
    observed_sources = {
        str(row["source_name"]) for row in held_out if str(row["source_name"])
    }
    expected_sources = set().union(*EXPECTED_SOURCES.values())
    if observed_sources != expected_sources:
        raise ValueError("fold-0 source inventory mismatch")

    focus = torch.load(
        focus_checkpoint_path, map_location="cpu", weights_only=False
    )
    baseline = torch.load(
        baseline_checkpoint_path, map_location="cpu", weights_only=False
    )
    focus_config = focus.get("config", {})
    mismatches = {
        key: {"expected": value, "observed": focus_config.get(key)}
        for key, value in EXPECTED_TRAINING.items()
        if focus_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Focus training configuration mismatch: {mismatches}")
    index_hash = plan["index_sha256"]
    if focus.get("index_sha256") != index_hash:
        raise ValueError("Focus patch-index hash mismatch")
    if baseline.get("index_sha256") != index_hash:
        raise ValueError("B32 patch-index hash mismatch")
    if int(focus.get("fold", -1)) != 0 or int(focus.get("folds", -1)) != 5:
        raise ValueError("Focus checkpoint is not grouped fold 0/5")
    if int(baseline.get("fold", -1)) != 0 or int(
        baseline.get("folds", -1)
    ) != 5:
        raise ValueError("B32 checkpoint is not grouped fold 0/5")
    if int(baseline.get("config", {}).get("base_channels", -1)) != 32:
        raise ValueError("comparison checkpoint is not the B32 baseline")
    if "soft_lung_seed" not in str(focus.get("architecture", "")):
        raise ValueError("Focus checkpoint is not the dual-window architecture")
    if focus.get("pretrained_sha256") != plan.get("pretrained_sha256"):
        raise ValueError("released Focus checkpoint identity mismatch")

    focus_prediction = _load_prediction(focus_prediction_path, True)
    baseline_prediction = _load_prediction(baseline_prediction_path, False)
    for name, prediction in (
        ("Focus", focus_prediction),
        ("B32", baseline_prediction),
    ):
        ids = prediction["positive_ids"]
        assert isinstance(ids, list)
        if set(ids) != expected_positive:
            raise ValueError(f"{name} positive held-out coverage mismatch")
    focus_negative_ids = focus_prediction["negative_ids"]
    assert isinstance(focus_negative_ids, list)
    if set(focus_negative_ids) != expected_negative:
        raise ValueError("Focus negative held-out coverage mismatch")

    def ordered_grid(prediction: dict[str, object], ids: set[str]) -> tuple[list[str], np.ndarray]:
        prediction_ids = prediction["positive_ids"]
        grid = prediction["dice_grid"]
        assert isinstance(prediction_ids, list)
        assert isinstance(grid, np.ndarray)
        positions = [index for index, patch_id in enumerate(prediction_ids) if patch_id in ids]
        return [prediction_ids[index] for index in positions], grid[np.asarray(positions)]

    focus_thresholds = focus_prediction["thresholds"]
    focus_dice = focus_prediction["dice_grid"]
    focus_negative_grid = focus_prediction["negative_grid"]
    baseline_thresholds = baseline_prediction["thresholds"]
    baseline_dice = baseline_prediction["dice_grid"]
    assert isinstance(focus_thresholds, np.ndarray)
    assert isinstance(focus_dice, np.ndarray)
    assert isinstance(focus_negative_grid, np.ndarray)
    assert isinstance(baseline_thresholds, np.ndarray)
    assert isinstance(baseline_dice, np.ndarray)
    focus_global = aggregate_positive_dice(
        focus_prediction["positive_ids"],
        focus_dice,
        focus_thresholds,
        held_out,
        focus_negative_grid.sum(axis=0),
        len(expected_negative),
    )
    baseline_global = aggregate_positive_dice(
        baseline_prediction["positive_ids"],
        baseline_dice,
        baseline_thresholds,
        held_out,
    )
    focus_metrics = focus["metrics"]
    baseline_metrics = baseline["metrics"]
    for observed, expected, label in (
        (focus_global["threshold"], focus_metrics["segmentation_threshold"], "Focus threshold"),
        (focus_global["case_balanced_dice"], focus_metrics["case_balanced_segmentation_dice"], "Focus case Dice"),
        (focus_global["cohort_balanced_dice"], focus_metrics["cohort_balanced_segmentation_dice"], "Focus cohort Dice"),
        (focus_global["negative_nonempty_rate"], focus_metrics["negative_nonempty_rate"], "Focus negative rate"),
        (baseline_global["threshold"], baseline_metrics["segmentation_threshold"], "B32 threshold"),
        (baseline_global["case_balanced_dice"], baseline_metrics["segmentation_dice"], "B32 case Dice"),
        (baseline_global["cohort_balanced_dice"], baseline_metrics["cohort_balanced_segmentation_dice"], "B32 cohort Dice"),
    ):
        _close(float(observed), float(expected), label)

    metadata = {str(row["patch_id"]): row for row in held_out}
    focus_negative_positions = {
        patch_id: position
        for position, patch_id in enumerate(focus_negative_ids)
    }
    target_groups = set(EXPECTED_GATE["target_groups"])
    eligible_groups: list[str] = []
    group_records: dict[str, dict[str, object]] = {}
    for group in EXPECTED_SOURCES:
        sources = EXPECTED_SOURCES[group]
        positive_ids = {
            patch_id
            for patch_id in expected_positive
            if contour_group(str(metadata[patch_id]["source_name"])) == group
        }
        negative_ids = {
            patch_id
            for patch_id in expected_negative
            if contour_group(str(metadata[patch_id]["source_name"])) == group
        }
        group_rows = [
            row
            for row in held_out
            if contour_group(str(row["source_name"])) == group
        ]
        if not positive_ids or not negative_ids:
            raise ValueError(f"group lacks both labels in fold 0: {group}")
        if {str(row["source_name"]) for row in group_rows} != sources:
            raise ValueError(f"source membership mismatch for {group}")
        focus_ids, focus_group_grid = ordered_grid(focus_prediction, positive_ids)
        b32_ids, b32_group_grid = ordered_grid(baseline_prediction, positive_ids)
        focus_group = aggregate_positive_dice(
            focus_ids, focus_group_grid, focus_thresholds, group_rows
        )
        b32_group = aggregate_positive_dice(
            b32_ids, b32_group_grid, baseline_thresholds, group_rows
        )
        focus_threshold_index = int(
            np.argmin(
                np.abs(focus_thresholds - float(focus_group["threshold"]))
            )
        )
        negative_positions = np.asarray(
            [focus_negative_positions[patch_id] for patch_id in sorted(negative_ids)],
            dtype=np.int64,
        )
        focus_negative_rate = float(
            focus_negative_grid[negative_positions, focus_threshold_index].mean()
        )
        case_gain = float(
            focus_group["case_balanced_dice"] - b32_group["case_balanced_dice"]
        )
        cohort_gain = float(
            focus_group["cohort_balanced_dice"]
            - b32_group["cohort_balanced_dice"]
        )
        checks = {
            "target_group": group in target_groups,
            "case_gain": case_gain
            >= float(EXPECTED_GATE["minimum_case_dice_gain_over_b32"]),
            "cohort_gain": cohort_gain
            >= float(EXPECTED_GATE["minimum_cohort_dice_gain_over_b32"]),
            "focus_negative_nonempty_rate": focus_negative_rate
            <= float(EXPECTED_GATE["maximum_focus_negative_nonempty_rate"]),
        }
        qualifies = all(checks.values())
        if qualifies:
            eligible_groups.append(group)
        group_records[group] = {
            "sources": sorted(sources),
            "focus_threshold": float(focus_group["threshold"]),
            "focus_case_dice": float(focus_group["case_balanced_dice"]),
            "focus_cohort_dice": float(focus_group["cohort_balanced_dice"]),
            "focus_negative_nonempty_rate": focus_negative_rate,
            "b32_threshold": float(b32_group["threshold"]),
            "b32_case_dice": float(b32_group["case_balanced_dice"]),
            "b32_cohort_dice": float(b32_group["cohort_balanced_dice"]),
            "case_gain": case_gain,
            "cohort_gain": cohort_gain,
            "checks": checks,
            "qualifies": qualifies,
        }

    passed = bool(eligible_groups)
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "decision": (
            "PROMOTE_MIXED_TO_FIVEFOLD" if passed else "KEEP_B32_MIXED"
        ),
        "passed": passed,
        "eligible_focus_groups": sorted(eligible_groups),
        "fixed_b32_groups": ["mature"],
        "groups": group_records,
        "gate": EXPECTED_GATE,
        "coverage": {
            "cases_total": len(cases),
            "held_out_cases": len({str(row["case"]) for row in held_out}),
            "positive_patches": len(expected_positive),
            "negative_patches": len(expected_negative),
            "sources": sorted(observed_sources),
        },
        "global_metrics": {
            "focus_case_dice": float(focus_global["case_balanced_dice"]),
            "focus_cohort_dice": float(focus_global["cohort_balanced_dice"]),
            "focus_negative_nonempty_rate": float(
                focus_global["negative_nonempty_rate"]
            ),
            "b32_case_dice": float(baseline_global["case_balanced_dice"]),
            "b32_cohort_dice": float(baseline_global["cohort_balanced_dice"]),
        },
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256(plan_path),
        "focus_checkpoint": str(focus_checkpoint_path.resolve()),
        "focus_checkpoint_sha256": sha256(focus_checkpoint_path),
        "focus_prediction": str(focus_prediction_path.resolve()),
        "focus_prediction_sha256": sha256(focus_prediction_path),
        "baseline_checkpoint": str(baseline_checkpoint_path.resolve()),
        "baseline_checkpoint_sha256": sha256(baseline_checkpoint_path),
        "baseline_prediction": str(baseline_prediction_path.resolve()),
        "baseline_prediction_sha256": sha256(baseline_prediction_path),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--focus-checkpoint", type=Path, required=True)
    parser.add_argument("--focus-prediction", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = select(
        args.plan,
        args.focus_checkpoint,
        args.focus_prediction,
        args.baseline_checkpoint,
        args.baseline_prediction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
