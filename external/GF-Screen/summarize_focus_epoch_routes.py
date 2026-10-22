#!/usr/bin/env python3
"""Select proposal-family Focus epochs from grouped held-out patch predictions.

This is a local-only post-training audit.  It never reads public or hidden
data.  Mature GF proposals, weak-domain proposals, and pooled lung cells each
select one checkpoint epoch per held-out fold.  The selected fold predictions
are then pooled to freeze one contour threshold per family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from candidate_refiner_sampling import semantic_source_group
from focus_validation_metrics import aggregate_positive_dice
from train_candidate_refiner import grouped_fold_rows


ROUTE_GROUPS = ("mature", "weak_sources", "lung_cells")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_group(source_name: str) -> str:
    source = semantic_source_group(source_name)
    if source in {"epoch105", "currentgf"}:
        return "mature"
    if source in {"autopet_weak", "lndb_weak"}:
        return "weak_sources"
    if source == "lung_cells":
        return "lung_cells"
    raise ValueError(f"unsupported Focus route source: {source_name}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def group_metrics(
    arrays: dict[str, np.ndarray],
    held_out_rows: list[dict[str, str]],
    group: str,
) -> dict[str, object]:
    rows = [row for row in held_out_rows if route_group(row["source_name"]) == group]
    positive_ids = arrays["positive_patch_id"].astype(str)
    negative_ids = arrays["negative_patch_id"].astype(str)
    patch_groups = route_group_by_patch(held_out_rows)
    positive_positions = [
        index
        for index, patch_id in enumerate(positive_ids)
        if patch_groups[patch_id] == group
    ]
    negative_positions = [
        index
        for index, patch_id in enumerate(negative_ids)
        if patch_groups[patch_id] == group
    ]
    selected_positive_ids = [positive_ids[index] for index in positive_positions]
    selected_negative = arrays["negative_nonempty_grid"][negative_positions]
    return aggregate_positive_dice(
        selected_positive_ids,
        arrays["segmentation_dice_grid"][positive_positions],
        arrays["segmentation_thresholds"],
        rows,
        selected_negative.sum(axis=0, dtype=np.int64),
        len(negative_positions),
    )


def route_group_by_patch(rows: list[dict[str, str]]) -> dict[str, str]:
    result = {
        str(row["patch_id"]): route_group(str(row["source_name"])) for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("patch index IDs are not unique")
    return result


def validate_prediction(
    arrays: dict[str, np.ndarray], held_out_rows: list[dict[str, str]]
) -> None:
    required = {
        "positive_patch_id",
        "segmentation_thresholds",
        "segmentation_dice_grid",
        "negative_patch_id",
        "negative_nonempty_grid",
        "negative_nonempty_by_threshold",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"Focus prediction archive lacks: {sorted(missing)}")
    positive_ids = arrays["positive_patch_id"].astype(str)
    negative_ids = arrays["negative_patch_id"].astype(str)
    expected_positive = {
        str(row["patch_id"]) for row in held_out_rows if int(row["label"]) == 1
    }
    expected_negative = {
        str(row["patch_id"]) for row in held_out_rows if int(row["label"]) == 0
    }
    if len(positive_ids) != len(set(positive_ids)) or set(positive_ids) != expected_positive:
        raise ValueError("positive held-out Focus coverage mismatch")
    if len(negative_ids) != len(set(negative_ids)) or set(negative_ids) != expected_negative:
        raise ValueError("negative held-out Focus coverage mismatch")
    thresholds = arrays["segmentation_thresholds"]
    if (
        thresholds.ndim != 1
        or len(thresholds) < 2
        or arrays["segmentation_dice_grid"].shape
        != (len(positive_ids), len(thresholds))
        or arrays["negative_nonempty_grid"].shape
        != (len(negative_ids), len(thresholds))
    ):
        raise ValueError("invalid Focus threshold-grid shape")
    if not np.array_equal(
        arrays["negative_nonempty_grid"].sum(axis=0),
        arrays["negative_nonempty_by_threshold"],
    ):
        raise ValueError("negative Focus activation totals mismatch")


def discover_epoch_pairs(run: Path, expected_epochs: int) -> list[tuple[int, Path, Path]]:
    checkpoints = {
        int(path.stem.rsplit("_", 1)[1]): path
        for path in run.glob("checkpoint_epoch_*.pt")
    }
    predictions = {
        int(path.name.removeprefix("validation_predictions_epoch_").removesuffix(".npz")): path
        for path in run.glob("validation_predictions_epoch_*.npz")
    }
    expected = set(range(expected_epochs))
    if set(checkpoints) != expected or set(predictions) != expected:
        raise ValueError(
            f"Focus epoch artifact coverage mismatch in {run}: "
            f"checkpoints={sorted(checkpoints)} predictions={sorted(predictions)}"
        )
    return [(epoch, checkpoints[epoch], predictions[epoch]) for epoch in sorted(expected)]


def summarize(
    index: Path,
    runs: list[Path],
    folds: int,
    seed: int,
    expected_epochs: int,
) -> dict[str, object]:
    if len(runs) != folds or expected_epochs < 2:
        raise ValueError("one run per fold and at least two epochs are required")
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not row.get("source_name", "").strip() for row in rows):
        raise ValueError("Focus route audit requires source_name for every patch")
    index_digest = sha256(index)
    observed_groups = {route_group(row["source_name"]) for row in rows}
    if observed_groups != set(ROUTE_GROUPS):
        raise ValueError(f"Focus route group mismatch: {sorted(observed_groups)}")

    selected_by_group: dict[str, list[dict[str, object]]] = {
        group: [] for group in ROUTE_GROUPS
    }
    fold_records = []
    common_thresholds: np.ndarray | None = None
    config_identity: dict[str, object] | None = None
    for fold, run in enumerate(runs):
        _, held_out_rows = grouped_fold_rows(rows, folds, fold, seed)
        held_out_groups = {route_group(row["source_name"]) for row in held_out_rows}
        if held_out_groups != set(ROUTE_GROUPS):
            raise ValueError(f"fold {fold} lacks a Focus route group")
        candidates: dict[str, list[dict[str, object]]] = {
            group: [] for group in ROUTE_GROUPS
        }
        epoch_records = []
        for epoch, checkpoint_path, prediction_path in discover_epoch_pairs(
            run, expected_epochs
        ):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            config = checkpoint.get("config", {})
            if (
                int(checkpoint.get("epoch", -1)) != epoch
                or int(checkpoint.get("fold", -1)) != fold
                or int(checkpoint.get("folds", -1)) != folds
                or int(config.get("seed", -1)) != seed
                or checkpoint.get("index_sha256") != index_digest
            ):
                raise ValueError(f"Focus checkpoint identity mismatch: {checkpoint_path}")
            identity = {
                key: config.get(key)
                for key in (
                    "epochs",
                    "batch_size",
                    "lr",
                    "weight_decay",
                    "loss_recipe",
                    "sampling_temperature",
                    "samples_per_epoch",
                    "freeze_encoder",
                    "encoder_lr_scale",
                    "input_mode",
                    "segmentation_threshold_max",
                )
            }
            if config_identity is None:
                config_identity = identity
            elif identity != config_identity:
                raise ValueError("Focus epoch-route runs use different training recipes")
            arrays = load_npz(prediction_path)
            validate_prediction(arrays, held_out_rows)
            thresholds = arrays["segmentation_thresholds"].astype(np.float64)
            if common_thresholds is None:
                common_thresholds = thresholds
            elif not np.array_equal(common_thresholds, thresholds):
                raise ValueError("Focus epoch threshold grids differ")
            epoch_group_metrics = {}
            for group in ROUTE_GROUPS:
                metrics = group_metrics(arrays, held_out_rows, group)
                epoch_group_metrics[group] = {
                    "selection_score": float(metrics["selection_score"]),
                    "case_balanced_dice": float(metrics["case_balanced_dice"]),
                    "cohort_balanced_dice": float(metrics["cohort_balanced_dice"]),
                    "negative_nonempty_rate": float(
                        metrics["negative_nonempty_rate"]
                    ),
                    "threshold": float(metrics["threshold"]),
                }
                candidates[group].append(
                    {
                        "epoch": epoch,
                        "checkpoint": checkpoint_path,
                        "prediction": prediction_path,
                        "arrays": arrays,
                        "metrics": metrics,
                    }
                )
            epoch_records.append({"epoch": epoch, "groups": epoch_group_metrics})

        selected_fold = {}
        for group in ROUTE_GROUPS:
            selected = max(
                candidates[group],
                key=lambda item: (
                    float(item["metrics"]["selection_score"]),
                    float(item["metrics"]["case_balanced_dice"]),
                    float(item["metrics"]["cohort_balanced_dice"]),
                    -int(item["epoch"]),
                ),
            )
            selected_by_group[group].append(
                {**selected, "fold": fold, "held_out_rows": held_out_rows}
            )
            selected_fold[group] = {
                "epoch": int(selected["epoch"]),
                "checkpoint": str(Path(selected["checkpoint"]).resolve()),
                "checkpoint_sha256": sha256(Path(selected["checkpoint"])),
                "prediction": str(Path(selected["prediction"]).resolve()),
                "selection_score": float(selected["metrics"]["selection_score"]),
                "case_balanced_dice": float(
                    selected["metrics"]["case_balanced_dice"]
                ),
                "negative_nonempty_rate": float(
                    selected["metrics"]["negative_nonempty_rate"]
                ),
            }
        fold_records.append(
            {
                "fold": fold,
                "run": str(run.resolve()),
                "selected": selected_fold,
                "trajectory": epoch_records,
            }
        )

    assert common_thresholds is not None and config_identity is not None
    group_reports = {}
    selected_positive_dice: dict[str, float] = {}
    selected_negative_active: dict[str, int] = {}
    for group in ROUTE_GROUPS:
        positive_ids: list[str] = []
        positive_grids = []
        negative_ids: list[str] = []
        negative_grids = []
        group_rows = [row for row in rows if route_group(row["source_name"]) == group]
        for selected in selected_by_group[group]:
            arrays = selected["arrays"]
            held_out_rows = selected["held_out_rows"]
            mapping = route_group_by_patch(held_out_rows)
            for kind, ids_key, grid_key, ids_out, grids_out in (
                ("positive", "positive_patch_id", "segmentation_dice_grid", positive_ids, positive_grids),
                ("negative", "negative_patch_id", "negative_nonempty_grid", negative_ids, negative_grids),
            ):
                ids = arrays[ids_key].astype(str)
                positions = [i for i, patch_id in enumerate(ids) if mapping[patch_id] == group]
                ids_out.extend(ids[positions].tolist())
                grids_out.append(arrays[grid_key][positions])
        positive_grid = np.concatenate(positive_grids, axis=0)
        negative_grid = np.concatenate(negative_grids, axis=0)
        aggregate = aggregate_positive_dice(
            positive_ids,
            positive_grid,
            common_thresholds,
            group_rows,
            negative_grid.sum(axis=0, dtype=np.int64),
            len(negative_ids),
        )
        best = int(aggregate["best_index"])
        for patch_id, value in zip(positive_ids, positive_grid[:, best]):
            if patch_id in selected_positive_dice:
                raise ValueError(f"duplicate routed positive patch: {patch_id}")
            selected_positive_dice[patch_id] = float(value)
        for patch_id, value in zip(negative_ids, negative_grid[:, best]):
            if patch_id in selected_negative_active:
                raise ValueError(f"duplicate routed negative patch: {patch_id}")
            selected_negative_active[patch_id] = int(value)
        group_reports[group] = {
            "threshold": float(aggregate["threshold"]),
            "selection_score": float(aggregate["selection_score"]),
            "dice": float(aggregate["case_balanced_dice"]),
            "cohort_balanced_dice": float(aggregate["cohort_balanced_dice"]),
            "negative_nonempty_rate": float(aggregate["negative_nonempty_rate"]),
            "positive_patches": int(aggregate["positive_patches"]),
            "positive_cases": int(aggregate["positive_cases"]),
            "positive_cohorts": int(aggregate["positive_cohorts"]),
        }

    all_positive = [str(row["patch_id"]) for row in rows if int(row["label"]) == 1]
    all_negative = [str(row["patch_id"]) for row in rows if int(row["label"]) == 0]
    if set(selected_positive_dice) != set(all_positive) or set(selected_negative_active) != set(all_negative):
        raise ValueError("routed Focus OOF coverage mismatch")
    routed = aggregate_positive_dice(
        all_positive,
        np.asarray([[selected_positive_dice[patch_id]] for patch_id in all_positive]),
        np.asarray([0.5]),
        rows,
        np.asarray([sum(selected_negative_active.values())]),
        len(all_negative),
    )
    source_to_group = {
        source: route_group(source)
        for source in sorted({str(row["source_name"]) for row in rows})
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": (
            "per-fold grouped held-out patch epoch selection followed by "
            "pooled grouped-OOF source threshold calibration"
        ),
        "index": str(index.resolve()),
        "index_sha256": index_digest,
        "folds": folds,
        "seed": seed,
        "expected_epochs": expected_epochs,
        "training_recipe": config_identity,
        "route_groups": list(ROUTE_GROUPS),
        "source_to_route": source_to_group,
        "source_segmentation_groups": group_reports,
        "segmentation": {
            "dice": float(routed["case_balanced_dice"]),
            "cohort_balanced_dice": float(routed["cohort_balanced_dice"]),
            "negative_nonempty_rate": float(routed["negative_nonempty_rate"]),
            "selection_score": float(routed["selection_score"]),
            "positive_patches": int(routed["positive_patches"]),
            "positive_cases": int(routed["positive_cases"]),
            "positive_cohorts": int(routed["positive_cohorts"]),
            "negative_patches": len(all_negative),
        },
        "fold_records": fold_records,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--expected-epochs", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        args.index, args.run, args.folds, args.seed, args.expected_epochs
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
