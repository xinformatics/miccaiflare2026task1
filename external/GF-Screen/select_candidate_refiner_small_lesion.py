#!/usr/bin/env python3
"""Gate the opt-in small-lesion loss on one matched grouped fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from candidate_refiner_sampling import semantic_source_group
from select_candidate_refiner_source_conditioning import summarize
from train_candidate_refiner import grouped_fold_rows


SMALL_SOURCE_GROUPS = ("lndb_weak", "lung_cells")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_history(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"history is not a list: {path}")
    return value


def best_segmentation_record(history: list[dict[str, object]]) -> dict[str, object]:
    if not history:
        raise ValueError("training history is empty")
    for record in history:
        if "segmentation_threshold" not in record:
            raise ValueError("history misses segmentation_threshold")
    return max(
        history,
        key=lambda row: (
            float(row["segmentation_dice"]),
            float(row["selection_score"]),
            float(row["average_precision"]),
            -int(row["epoch"]),
        ),
    )


def source_dice_at_deployment_threshold(
    prediction_path: Path,
    validation_rows: list[dict[str, str]],
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    with np.load(prediction_path, allow_pickle=False) as values:
        required = {
            "positive_patch_id",
            "segmentation_thresholds",
            "segmentation_dice_grid",
        }
        if not required.issubset(values.files):
            raise ValueError(f"prediction archive misses {sorted(required-set(values.files))}")
        patch_ids = [str(value) for value in values["positive_patch_id"]]
        thresholds = np.asarray(values["segmentation_thresholds"], dtype=np.float64)
        grid = np.asarray(values["segmentation_dice_grid"], dtype=np.float64)
    if len(patch_ids) != len(set(patch_ids)) or grid.shape != (
        len(patch_ids),
        len(thresholds),
    ):
        raise ValueError("invalid positive Dice grid")
    rows = {str(row["patch_id"]): row for row in validation_rows}
    if len(rows) != len(validation_rows):
        raise ValueError("validation patch IDs are not unique")
    expected = {patch_id for patch_id, row in rows.items() if int(row["label"]) == 1}
    if set(patch_ids) != expected:
        raise ValueError(
            f"positive validation coverage mismatch: expected={len(expected)} "
            f"observed={len(patch_ids)}"
        )
    threshold_index = int(np.argmin(np.abs(thresholds - float(threshold))))
    if abs(float(thresholds[threshold_index]) - float(threshold)) > 1e-5:
        raise ValueError("deployment threshold is absent from prediction grid")

    by_group_case: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for patch_index, patch_id in enumerate(patch_ids):
        row = rows[patch_id]
        group = semantic_source_group(row["source_name"])
        by_group_case[group][str(row["case"])].append(
            float(grid[patch_index, threshold_index])
        )

    result: dict[str, dict[str, float | int]] = {}
    pooled: dict[str, list[float]] = defaultdict(list)
    for group in SMALL_SOURCE_GROUPS:
        case_values = by_group_case.get(group, {})
        if not case_values:
            raise ValueError(f"held-out fold has no positive patches for {group}")
        means = [float(np.mean(values)) for values in case_values.values()]
        result[group] = {
            "case_balanced_dice": float(np.mean(means)),
            "positive_cases": len(means),
            "positive_patches": sum(len(values) for values in case_values.values()),
        }
        for case, values in case_values.items():
            pooled[case].extend(values)
    pooled_means = [float(np.mean(values)) for values in pooled.values()]
    result["pooled_small_sources"] = {
        "case_balanced_dice": float(np.mean(pooled_means)),
        "positive_cases": len(pooled_means),
        "positive_patches": sum(len(values) for values in pooled.values()),
    }
    return result


def decide(
    baseline_summary: dict[str, dict[str, float | int]],
    candidate_summary: dict[str, dict[str, float | int]],
    baseline_sources: dict[str, dict[str, float | int]],
    candidate_sources: dict[str, dict[str, float | int]],
    minimum_overall_segmentation_gain: float = 0.005,
    minimum_small_source_gain: float = 0.010,
    minimum_composite_gain: float = 0.0,
    maximum_classification_drop: float = 0.010,
) -> tuple[str, dict[str, float], dict[str, bool]]:
    gains = {
        "segmentation_dice": float(candidate_summary["segmentation"]["segmentation_dice"])
        - float(baseline_summary["segmentation"]["segmentation_dice"]),
        "selection_score": float(candidate_summary["composite"]["selection_score"])
        - float(baseline_summary["composite"]["selection_score"]),
        "average_precision": float(candidate_summary["classification"]["average_precision"])
        - float(baseline_summary["classification"]["average_precision"]),
        "pooled_small_source_dice": float(
            candidate_sources["pooled_small_sources"]["case_balanced_dice"]
        )
        - float(baseline_sources["pooled_small_sources"]["case_balanced_dice"]),
        "lndb_weak_dice": float(candidate_sources["lndb_weak"]["case_balanced_dice"])
        - float(baseline_sources["lndb_weak"]["case_balanced_dice"]),
        "lung_cells_dice": float(candidate_sources["lung_cells"]["case_balanced_dice"])
        - float(baseline_sources["lung_cells"]["case_balanced_dice"]),
    }
    if not all(math.isfinite(value) for value in gains.values()):
        raise ValueError("non-finite ablation gain")
    gates = {
        "overall_segmentation_gain": gains["segmentation_dice"]
        >= minimum_overall_segmentation_gain,
        "small_source_gain": gains["pooled_small_source_dice"]
        >= minimum_small_source_gain,
        "composite_nonregression": gains["selection_score"] >= minimum_composite_gain,
        "classification_preserved": gains["average_precision"]
        >= -maximum_classification_drop,
    }
    decision = "EXPAND_SMALL_LESION_LOSS" if all(gates.values()) else "KEEP_LEGACY_LOSS"
    return decision, gains, gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--candidate-history", type=Path, required=True)
    parser.add_argument("--baseline-prediction", type=Path, required=True)
    parser.add_argument("--candidate-prediction", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--minimum-overall-segmentation-gain", type=float, default=0.005)
    parser.add_argument("--minimum-small-source-gain", type=float, default=0.010)
    parser.add_argument("--minimum-composite-gain", type=float, default=0.0)
    parser.add_argument("--maximum-classification-drop", type=float, default=0.010)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.folds < 2 or not 0 <= args.fold < args.folds:
        parser.error("invalid fold")
    if min(
        args.minimum_overall_segmentation_gain,
        args.minimum_small_source_gain,
        args.minimum_composite_gain,
        args.maximum_classification_drop,
    ) < 0:
        parser.error("gate tolerances must be nonnegative")
    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _, validation_rows = grouped_fold_rows(rows, args.folds, args.fold, args.seed)
    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if source_audit.get("index_sha256") != sha256(args.index):
        raise ValueError("source audit is not bound to the patch index")
    if source_audit.get("recommendations", {}).get("test_small_lesion_loss") is not True:
        raise ValueError("source audit did not recommend the small-lesion ablation")
    baseline_history = load_history(args.baseline_history)
    candidate_history = load_history(args.candidate_history)
    baseline_record = best_segmentation_record(baseline_history)
    candidate_record = best_segmentation_record(candidate_history)
    baseline_sources = source_dice_at_deployment_threshold(
        args.baseline_prediction,
        validation_rows,
        float(baseline_record["segmentation_threshold"]),
    )
    candidate_sources = source_dice_at_deployment_threshold(
        args.candidate_prediction,
        validation_rows,
        float(candidate_record["segmentation_threshold"]),
    )
    baseline_summary, candidate_summary = summarize(baseline_history), summarize(candidate_history)
    decision, gains, gates = decide(
        baseline_summary,
        candidate_summary,
        baseline_sources,
        candidate_sources,
        args.minimum_overall_segmentation_gain,
        args.minimum_small_source_gain,
        args.minimum_composite_gain,
        args.maximum_classification_drop,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "fold": args.fold,
        "folds": args.folds,
        "seed": args.seed,
        "index": str(args.index.resolve()),
        "index_sha256": sha256(args.index),
        "source_audit": str(args.source_audit.resolve()),
        "source_audit_sha256": sha256(args.source_audit),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "baseline_small_sources": baseline_sources,
        "candidate_small_sources": candidate_sources,
        "gains": gains,
        "gates": gates,
        "thresholds": {
            "minimum_overall_segmentation_gain": args.minimum_overall_segmentation_gain,
            "minimum_small_source_gain": args.minimum_small_source_gain,
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
