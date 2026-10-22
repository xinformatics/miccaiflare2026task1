#!/usr/bin/env python3
"""Freeze a GF Focus contour threshold from grouped held-out patch predictions."""

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
from candidate_refiner_sampling import semantic_source_group
from train_candidate_refiner import grouped_fold_rows


def contour_route_group(source_name: str) -> str:
    """Pool proposal identities into the three deployment contour families."""

    source = semantic_source_group(source_name)
    if source in {"epoch105", "currentgf"}:
        return "mature"
    if source in {"autopet_weak", "lndb_weak"}:
        return "weak_sources"
    if source == "lung_cells":
        return "lung_cells"
    return source


def summarize(
    index: Path,
    checkpoints: list[Path],
    predictions: list[Path],
    folds: int,
    seed: int,
) -> dict:
    if len(checkpoints) != folds or len(predictions) != folds:
        raise ValueError("one checkpoint and prediction archive are required per fold")
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    all_dice: list[np.ndarray] = []
    negative_nonempty = None
    negative_total = 0
    thresholds = None
    fold_rows = []
    loss_recipe = None
    batch_size = None
    input_mode = None
    training_recipe = None
    observed_positive_ids: set[str] = set()
    all_positive_ids: list[str] = []
    observed_negative_ids: set[str] = set()
    all_negative_ids: list[str] = []
    all_negative_grids: list[np.ndarray] = []
    for fold, (checkpoint_path, prediction_path) in enumerate(
        zip(checkpoints, predictions)
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("fold", -1)) != fold:
            raise ValueError(f"checkpoint fold mismatch: {checkpoint_path}")
        if int(checkpoint.get("folds", -1)) != folds:
            raise ValueError(f"checkpoint fold-count mismatch: {checkpoint_path}")
        if checkpoint.get("index_sha256") != index_sha:
            raise ValueError(f"checkpoint index mismatch: {checkpoint_path}")
        checkpoint_recipe = checkpoint.get("loss_recipe")
        if checkpoint_recipe not in {"boundary_balanced", "release_dice_ce"}:
            raise ValueError(f"invalid Focus loss recipe: {checkpoint_path}")
        if loss_recipe is None:
            loss_recipe = checkpoint_recipe
        elif checkpoint_recipe != loss_recipe:
            raise ValueError("Focus checkpoints use different loss recipes")
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_batch = int(checkpoint_config.get("batch_size", 0))
        if checkpoint_batch not in {4, 8}:
            raise ValueError(f"invalid Focus batch size: {checkpoint_path}")
        if batch_size is None:
            batch_size = checkpoint_batch
        elif checkpoint_batch != batch_size:
            raise ValueError("Focus checkpoints use different batch sizes")
        checkpoint_input = str(checkpoint_config.get("input_mode", "soft"))
        if checkpoint_input not in {"soft", "soft_seed", "soft_lung_seed"}:
            raise ValueError(f"invalid Focus input mode: {checkpoint_path}")
        if input_mode is None:
            input_mode = checkpoint_input
        elif checkpoint_input != input_mode:
            raise ValueError("Focus checkpoints use different input modes")
        checkpoint_recipe = {
            "epochs": int(checkpoint_config.get("epochs", 0)),
            "batch_size": checkpoint_batch,
            "lr": float(checkpoint_config.get("lr", 0.0)),
            "weight_decay": float(checkpoint_config.get("weight_decay", 0.0)),
            "freeze_encoder": bool(checkpoint_config.get("freeze_encoder", False)),
            "encoder_lr_scale": float(
                checkpoint_config.get("encoder_lr_scale", 1.0)
            ),
            "input_mode": checkpoint_input,
            "samples_per_epoch": int(
                checkpoint_config.get("samples_per_epoch", 0)
            ),
            "sampling_temperature": float(
                checkpoint_config.get("sampling_temperature", 0.5)
            ),
            "seed": int(checkpoint_config.get("seed", seed)),
            "pretrained_sha256": checkpoint.get("pretrained_sha256"),
        }
        if training_recipe is None:
            training_recipe = checkpoint_recipe
        elif checkpoint_recipe != training_recipe:
            raise ValueError("Focus checkpoints use different training recipes")
        _, held_out = grouped_fold_rows(rows, folds, fold, seed)
        expected_positive = {
            row["patch_id"] for row in held_out if int(row["label"]) == 1
        }
        expected_negative = sum(int(row["label"]) == 0 for row in held_out)
        with np.load(prediction_path, allow_pickle=False) as archive:
            fold_ids = [str(value) for value in archive["positive_patch_id"]]
            fold_thresholds = archive["segmentation_thresholds"].astype(np.float64)
            fold_dice = archive["segmentation_dice_grid"].astype(np.float64)
            fold_negative = archive["negative_nonempty_by_threshold"].astype(np.int64)
            fold_negative_ids = [str(value) for value in archive["negative_patch_id"]]
            fold_negative_grid = archive["negative_nonempty_grid"].astype(np.int64)
        if len(fold_ids) != len(set(fold_ids)) or set(fold_ids) != expected_positive:
            raise ValueError(f"positive held-out coverage mismatch in fold {fold}")
        if observed_positive_ids.intersection(fold_ids):
            raise ValueError("positive patch appears in more than one held-out fold")
        observed_positive_ids.update(fold_ids)
        all_positive_ids.extend(fold_ids)
        expected_negative_ids = {
            row["patch_id"] for row in held_out if int(row["label"]) == 0
        }
        if (
            len(fold_negative_ids) != len(set(fold_negative_ids))
            or set(fold_negative_ids) != expected_negative_ids
        ):
            raise ValueError(f"negative held-out coverage mismatch in fold {fold}")
        if observed_negative_ids.intersection(fold_negative_ids):
            raise ValueError("negative patch appears in more than one held-out fold")
        observed_negative_ids.update(fold_negative_ids)
        all_negative_ids.extend(fold_negative_ids)
        if fold_dice.shape != (len(fold_ids), len(fold_thresholds)):
            raise ValueError(f"invalid Dice grid in fold {fold}")
        if fold_negative.shape != fold_thresholds.shape:
            raise ValueError(f"invalid negative grid in fold {fold}")
        if fold_negative_grid.shape != (len(fold_negative_ids), len(fold_thresholds)):
            raise ValueError(f"invalid per-patch negative grid in fold {fold}")
        if not np.array_equal(fold_negative_grid.sum(axis=0), fold_negative):
            raise ValueError(f"negative activation total mismatch in fold {fold}")
        if thresholds is None:
            thresholds = fold_thresholds
            negative_nonempty = np.zeros_like(fold_negative)
        elif not np.array_equal(thresholds, fold_thresholds):
            raise ValueError("Focus threshold grids differ across folds")
        if int(checkpoint["metrics"]["negative_patches"]) != expected_negative:
            raise ValueError(f"negative held-out coverage mismatch in fold {fold}")
        all_dice.append(fold_dice)
        all_negative_grids.append(fold_negative_grid)
        negative_nonempty += fold_negative
        negative_total += expected_negative
        fold_rows.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint_path.resolve()),
                "prediction": str(prediction_path.resolve()),
                "positive_patches": len(fold_ids),
                "negative_patches": expected_negative,
                "checkpoint_epoch": int(checkpoint["epoch"]),
            }
        )
    assert thresholds is not None and negative_nonempty is not None
    dice_grid = np.concatenate(all_dice, axis=0)
    negative_grid = np.concatenate(all_negative_grids, axis=0)
    aggregate = aggregate_positive_dice(
        all_positive_ids,
        dice_grid,
        thresholds,
        rows,
        negative_nonempty,
        negative_total,
    )
    patch_mean = aggregate["patch_mean_grid"]
    case_mean = aggregate["case_balanced_grid"]
    cohort_mean = aggregate["cohort_balanced_grid"]
    negative_rate = aggregate["negative_nonempty_rate_grid"]
    selection = aggregate["selection_score_grid"]
    best = aggregate["best_index"]
    metadata = {str(row["patch_id"]): row for row in rows}
    source_names = [str(row.get("source_name", "")).strip() for row in rows]
    if any(source_names) and not all(source_names):
        raise ValueError("source_name must cover every patch or no patches")
    source_segmentation: dict[str, dict[str, object]] = {}
    source_segmentation_groups: dict[str, dict[str, object]] = {}
    if source_names:
        positive_indices_by_group: dict[str, list[int]] = {}
        negative_indices_by_group: dict[str, list[int]] = {}
        source_names_by_group: dict[str, set[str]] = {}
        for position, patch_id in enumerate(all_positive_ids):
            source_name = str(metadata[patch_id]["source_name"])
            group = semantic_source_group(source_name)
            positive_indices_by_group.setdefault(group, []).append(position)
            source_names_by_group.setdefault(group, set()).add(source_name)
        for position, patch_id in enumerate(all_negative_ids):
            source_name = str(metadata[patch_id]["source_name"])
            group = semantic_source_group(source_name)
            negative_indices_by_group.setdefault(group, []).append(position)
        for group, indices in sorted(positive_indices_by_group.items()):
            group_sources = source_names_by_group[group]
            group_rows = [
                row
                for row in rows
                if semantic_source_group(str(row["source_name"])) == group
            ]
            group_ids = [all_positive_ids[index] for index in indices]
            group_aggregate = aggregate_positive_dice(
                group_ids,
                dice_grid[np.asarray(indices, dtype=np.int64)],
                thresholds,
                group_rows,
            )
            group_negative_indices = negative_indices_by_group.get(group, [])
            if not group_negative_indices:
                raise ValueError(f"source group lacks negative patches: {group}")
            group_threshold_index = int(
                np.argmin(np.abs(thresholds - float(group_aggregate["threshold"])))
            )
            group_negative_rate = float(
                negative_grid[
                    np.asarray(group_negative_indices, dtype=np.int64),
                    group_threshold_index,
                ].mean()
            )
            report = {
                "source_group": group,
                "threshold": float(group_aggregate["threshold"]),
                "case_balanced_dice": float(
                    group_aggregate["case_balanced_dice"]
                ),
                "cohort_balanced_dice": float(
                    group_aggregate["cohort_balanced_dice"]
                ),
                "positive_patches": int(group_aggregate["positive_patches"]),
                "positive_cases": int(group_aggregate["positive_cases"]),
                "positive_cohorts": int(group_aggregate["positive_cohorts"]),
                "negative_patches": len(group_negative_indices),
                "negative_nonempty_rate": group_negative_rate,
            }
            for source_name in sorted(group_sources):
                source_segmentation[source_name] = dict(report)
        positive_indices_by_route: dict[str, list[int]] = {}
        negative_indices_by_route: dict[str, list[int]] = {}
        sources_by_route: dict[str, set[str]] = {}
        for position, patch_id in enumerate(all_positive_ids):
            source_name = str(metadata[patch_id]["source_name"])
            route = contour_route_group(source_name)
            positive_indices_by_route.setdefault(route, []).append(position)
            sources_by_route.setdefault(route, set()).add(source_name)
        for position, patch_id in enumerate(all_negative_ids):
            source_name = str(metadata[patch_id]["source_name"])
            route = contour_route_group(source_name)
            negative_indices_by_route.setdefault(route, []).append(position)
        for route, indices in sorted(positive_indices_by_route.items()):
            route_sources = sources_by_route[route]
            route_rows = [
                row
                for row in rows
                if contour_route_group(str(row["source_name"])) == route
            ]
            route_ids = [all_positive_ids[index] for index in indices]
            route_aggregate = aggregate_positive_dice(
                route_ids,
                dice_grid[np.asarray(indices, dtype=np.int64)],
                thresholds,
                route_rows,
            )
            route_negative_indices = negative_indices_by_route.get(route, [])
            if not route_negative_indices:
                raise ValueError(f"contour route lacks negative patches: {route}")
            route_threshold_index = int(
                np.argmin(np.abs(thresholds - float(route_aggregate["threshold"])))
            )
            route_negative_rate = float(
                negative_grid[
                    np.asarray(route_negative_indices, dtype=np.int64),
                    route_threshold_index,
                ].mean()
            )
            source_segmentation_groups[route] = {
                "sources": sorted(route_sources),
                "threshold": float(route_aggregate["threshold"]),
                "dice": float(route_aggregate["case_balanced_dice"]),
                "case_balanced_dice": float(
                    route_aggregate["case_balanced_dice"]
                ),
                "cohort_balanced_dice": float(
                    route_aggregate["cohort_balanced_dice"]
                ),
                "positive_patches": int(route_aggregate["positive_patches"]),
                "positive_cases": int(route_aggregate["positive_cases"]),
                "positive_cohorts": int(route_aggregate["positive_cohorts"]),
                "negative_patches": len(route_negative_indices),
                "negative_nonempty_rate": route_negative_rate,
            }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "grouped case-held-out complete-label patch OOF only",
        "index": str(index.resolve()),
        "index_sha256": index_sha,
        "folds": folds,
        "seed": seed,
        "loss_recipe": loss_recipe,
        "batch_size": batch_size,
        "input_mode": input_mode,
        "training_recipe": training_recipe,
        "positive_patches": int(len(dice_grid)),
        "positive_cases": int(aggregate["positive_cases"]),
        "positive_cohorts": int(aggregate["positive_cohorts"]),
        "negative_patches": int(negative_total),
        "segmentation_threshold": float(thresholds[best]),
        "segmentation_dice": float(case_mean[best]),
        "case_balanced_segmentation_dice": float(case_mean[best]),
        "patch_mean_segmentation_dice": float(patch_mean[best]),
        "cohort_balanced_segmentation_dice": float(cohort_mean[best]),
        "negative_nonempty_rate": float(negative_rate[best]),
        "selection_score": float(selection[best]),
        "source_segmentation": dict(sorted(source_segmentation.items())),
        "source_segmentation_groups": dict(
            sorted(source_segmentation_groups.items())
        ),
        "grid": [
            {
                "threshold": float(threshold),
                "segmentation_dice": float(case_mean[index]),
                "case_balanced_segmentation_dice": float(case_mean[index]),
                "patch_mean_segmentation_dice": float(patch_mean[index]),
                "cohort_balanced_segmentation_dice": float(cohort_mean[index]),
                "negative_nonempty_rate": float(negative_rate[index]),
                "selection_score": float(selection[index]),
            }
            for index, threshold in enumerate(thresholds)
        ],
        "fold_records": fold_rows,
        "public_read": False,
        "hidden_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        args.index, args.checkpoint, args.prediction, args.folds, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
