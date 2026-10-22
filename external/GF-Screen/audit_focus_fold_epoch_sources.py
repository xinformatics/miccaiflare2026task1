#!/usr/bin/env python3
"""Audit one retained Focus fold/epoch by proposal-family source group.

This is a diagnostic companion to ``summarize_focus_epoch_routes.py``. It
reuses the same archive validation and aggregation code but does not select a
checkpoint, trigger inference, read public/hidden data, or create a submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import torch

from summarize_focus_epoch_routes import (
    ROUTE_GROUPS,
    group_metrics,
    load_npz,
    sha256,
    validate_prediction,
)
from train_candidate_refiner import grouped_fold_rows


def audit(
    index: Path,
    checkpoint_path: Path,
    prediction_path: Path,
    folds: int,
    fold: int,
    seed: int,
    epoch: int,
) -> dict[str, object]:
    if folds < 2 or not 0 <= fold < folds or epoch < 0:
        raise ValueError("invalid fold or epoch")
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty Focus patch index")
    _, held_out_rows = grouped_fold_rows(rows, folds, fold, seed)
    index_digest = sha256(index)
    # Only metadata is inspected here. Memory mapping avoids eagerly copying
    # hundreds of MiB of model/optimizer tensors during live diagnostics.
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config = checkpoint.get("config", {})
    if (
        int(checkpoint.get("epoch", -1)) != epoch
        or int(checkpoint.get("fold", -1)) != fold
        or int(checkpoint.get("folds", -1)) != folds
        or int(config.get("seed", -1)) != seed
        or checkpoint.get("index_sha256") != index_digest
    ):
        raise ValueError("Focus checkpoint identity mismatch")
    arrays = load_npz(prediction_path)
    validate_prediction(arrays, held_out_rows)
    groups: dict[str, dict[str, object]] = {}
    for group in ROUTE_GROUPS:
        metrics = group_metrics(arrays, held_out_rows, group)
        groups[group] = {
            "threshold": float(metrics["threshold"]),
            "selection_score": float(metrics["selection_score"]),
            "case_balanced_dice": float(metrics["case_balanced_dice"]),
            "cohort_balanced_dice": float(metrics["cohort_balanced_dice"]),
            "patch_mean_dice": float(metrics["patch_mean_dice"]),
            "negative_nonempty_rate": float(metrics["negative_nonempty_rate"]),
            "positive_cases": int(metrics["positive_cases"]),
            "positive_cohorts": int(metrics["positive_cohorts"]),
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "one case-grouped internal held-out fold; diagnostic only",
        "fold": fold,
        "folds": folds,
        "epoch": epoch,
        "seed": seed,
        "held_out_cases": len({str(row["case"]) for row in held_out_rows}),
        "index": str(index.resolve()),
        "index_sha256": index_digest,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "prediction": str(prediction_path.resolve()),
        "prediction_sha256": sha256(prediction_path),
        "groups": groups,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(
        args.index,
        args.checkpoint,
        args.prediction,
        args.folds,
        args.fold,
        args.seed,
        args.epoch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
