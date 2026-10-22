#!/usr/bin/env python3
"""Reproducible source-stratified audit of one grouped-fold prediction file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from summarize_candidate_refiner_oof import (
    pool_source_classification_groups,
    pool_source_segmentation_groups,
    source_classification_metrics,
    source_segmentation_metrics,
)
from train_candidate_refiner import grouped_fold_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _, validation_rows = grouped_fold_rows(
        rows, args.folds, args.fold, args.seed
    )
    expected = {str(row["patch_id"]): row for row in validation_rows}
    if len(expected) != len(validation_rows):
        raise ValueError("validation index contains duplicate patch IDs")
    with np.load(args.prediction, allow_pickle=False) as archive:
        patch_ids = [str(value) for value in archive["patch_id"]]
        labels = archive["label"].astype(np.uint8)
        probabilities = archive["classification_probability"].astype(np.float64)
        positive_ids = [str(value) for value in archive["positive_patch_id"]]
        segmentation_grid = archive["segmentation_dice_grid"].astype(np.float64)
        thresholds = archive["segmentation_thresholds"].astype(np.float64)
    if set(patch_ids) != set(expected) or len(patch_ids) != len(expected):
        raise ValueError("prediction does not exactly cover the grouped validation fold")
    classification = source_classification_metrics(
        expected, patch_ids, labels, probabilities
    )
    segmentation = source_segmentation_metrics(
        expected, positive_ids, segmentation_grid, thresholds
    )
    group_specification = [
        "mature=epoch105,currentgf",
        "weak_sources=autopet_weak,lndb_weak",
        "lung_cells=" + ",".join(f"lung_cell{i}" for i in range(8))
    ]
    classification_groups = pool_source_classification_groups(
        classification,
        expected,
        patch_ids,
        labels,
        probabilities,
        group_specification,
    )
    segmentation_groups = pool_source_segmentation_groups(
        segmentation,
        expected,
        positive_ids,
        segmentation_grid,
        thresholds,
        group_specification,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "one case-grouped held-out fold; no public or hidden data",
        "fold": args.fold,
        "folds": args.folds,
        "seed": args.seed,
        "epoch": args.epoch,
        "cases": len({row["case"] for row in validation_rows}),
        "patches": len(validation_rows),
        "positive_patches": int(labels.sum()),
        "negative_patches": int((labels == 0).sum()),
        "index": str(args.index.resolve()),
        "index_sha256": sha256(args.index),
        "prediction": str(args.prediction.resolve()),
        "prediction_sha256": sha256(args.prediction),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_classification": classification,
        "source_classification_groups": classification_groups,
        "source_segmentation": segmentation,
        "source_segmentation_groups": segmentation_groups,
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
