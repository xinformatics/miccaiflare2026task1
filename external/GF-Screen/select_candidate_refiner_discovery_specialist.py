#!/usr/bin/env python3
"""Gate a routed weak/cell contour specialist on one exact held-out fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from train_candidate_refiner import (
    apply_source_group_segmentation_metrics,
    grouped_fold_rows,
)


DISCOVERY_GROUPS = ("autopet_weak", "lndb_weak", "lung_cells")
WEAK_GROUPS = ("autopet_weak", "lndb_weak")
CELL_GROUPS = ("lung_cells",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def segmentation_metrics(
    predictions: dict[str, np.ndarray],
    validation_rows: list[dict[str, str]],
    groups: tuple[str, ...],
) -> dict[str, object]:
    return apply_source_group_segmentation_metrics(
        {}, predictions, validation_rows, groups
    )


def select_specialist(
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
    minimum_combined_gain: float,
    maximum_group_drop: float,
) -> tuple[str, dict[str, object]]:
    gains = {
        name: float(candidate[name]["routed_segmentation_dice"])
        - float(baseline[name]["routed_segmentation_dice"])
        for name in ("discovery", "weak", "lung_cells")
    }
    checks = {
        "combined_discovery_gain": gains["discovery"] >= minimum_combined_gain,
        "weak_source_nonregression": gains["weak"] >= -maximum_group_drop,
        "lung_cell_nonregression": gains["lung_cells"] >= -maximum_group_drop,
    }
    return (
        "EXPAND_DISCOVERY_SPECIALIST" if all(checks.values()) else "KEEP_PRIMARY",
        {"gains": gains, "checks": checks},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-prediction", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-prediction", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--minimum-combined-gain", type=float, default=0.010)
    parser.add_argument("--maximum-group-drop", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_combined_gain < 0 or args.maximum_group_drop < 0:
        parser.error("gain/drop gates must be nonnegative")
    for path in (
        args.index,
        args.baseline_checkpoint,
        args.baseline_prediction,
        args.candidate_checkpoint,
        args.candidate_prediction,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _, validation_rows = grouped_fold_rows(rows, args.folds, args.fold, args.seed)
    index_sha = sha256(args.index)
    baseline_checkpoint = torch.load(
        args.baseline_checkpoint, map_location="cpu", weights_only=False
    )
    candidate_checkpoint = torch.load(
        args.candidate_checkpoint, map_location="cpu", weights_only=False
    )
    for name, checkpoint in (
        ("baseline", baseline_checkpoint),
        ("candidate", candidate_checkpoint),
    ):
        if checkpoint.get("index_sha256") != index_sha:
            raise ValueError(f"{name} checkpoint/index identity mismatch")
        if int(checkpoint.get("fold", -1)) != args.fold:
            raise ValueError(f"{name} checkpoint fold mismatch")
    candidate_config = candidate_checkpoint.get("config", {})
    if tuple(candidate_config.get("train_source_groups", [])) != DISCOVERY_GROUPS:
        raise ValueError("candidate was not trained exclusively on discovery groups")
    if (
        tuple(candidate_config.get("segmentation_selection_source_groups", []))
        != DISCOVERY_GROUPS
    ):
        raise ValueError("candidate routed-checkpoint selection groups mismatch")

    predictions = {
        "baseline": load_predictions(args.baseline_prediction),
        "candidate": load_predictions(args.candidate_prediction),
    }
    group_specs = {
        "discovery": DISCOVERY_GROUPS,
        "weak": WEAK_GROUPS,
        "lung_cells": CELL_GROUPS,
    }
    metrics = {
        model: {
            name: segmentation_metrics(prediction, validation_rows, groups)
            for name, groups in group_specs.items()
        }
        for model, prediction in predictions.items()
    }
    checkpoint_metric = float(
        candidate_checkpoint.get("metrics", {}).get(
            "routed_segmentation_dice", float("nan")
        )
    )
    recomputed_metric = float(
        metrics["candidate"]["discovery"]["routed_segmentation_dice"]
    )
    if not np.isfinite(checkpoint_metric) or abs(checkpoint_metric - recomputed_metric) > 1e-8:
        raise ValueError("candidate checkpoint/prediction routed metric mismatch")
    decision, comparison = select_specialist(
        metrics["baseline"],
        metrics["candidate"],
        args.minimum_combined_gain,
        args.maximum_group_drop,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "experiment": "discovery-only contour specialist routed around mature GF",
        "fold": args.fold,
        "folds": args.folds,
        "seed": args.seed,
        "index": str(args.index.resolve()),
        "index_sha256": index_sha,
        "baseline": {
            "checkpoint": str(args.baseline_checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.baseline_checkpoint),
            "prediction": str(args.baseline_prediction.resolve()),
            "prediction_sha256": sha256(args.baseline_prediction),
            "metrics": metrics["baseline"],
        },
        "candidate": {
            "checkpoint": str(args.candidate_checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.candidate_checkpoint),
            "prediction": str(args.candidate_prediction.resolve()),
            "prediction_sha256": sha256(args.candidate_prediction),
            "metrics": metrics["candidate"],
        },
        "gates": {
            "minimum_combined_gain": args.minimum_combined_gain,
            "maximum_group_drop": args.maximum_group_drop,
            **comparison,
        },
        "mature_route": "unchanged primary contour checkpoint",
        "classification_route": "unchanged primary classification checkpoint",
        "labels_read": True,
        "public_or_hidden_data_read": False,
        "submission_created": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
