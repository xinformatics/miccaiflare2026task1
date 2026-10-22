#!/usr/bin/env python3
"""Audit conservative source-and-size checkpoint routing on one held-out fold.

This local-only diagnostic uses proposal seed volume, which is available at
inference.  A size-specific checkpoint is accepted only when the stratum has
enough positive cases and improves within-stratum case-balanced Dice by a
declared margin over the source-only checkpoint.  It never reads public or
hidden data and does not materialize predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from candidate_refiner_sampling import semantic_source_group
from focus_validation_metrics import aggregate_positive_dice
from train_candidate_refiner import grouped_fold_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def size_bin(row: dict[str, str]) -> str:
    volume = float(row["seed_mm3"])
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"invalid seed_mm3 for {row.get('patch_id')}: {volume}")
    if volume < 1000:
        return "small_lt1000"
    if volume < 10000:
        return "medium_1000to10000"
    return "large_ge10000"


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "positive_patch_id",
            "segmentation_thresholds",
            "segmentation_dice_grid",
        }
        if not required <= set(archive.files):
            raise ValueError(f"prediction archive lacks {sorted(required-set(archive.files))}")
        return {name: archive[name] for name in required}


def select(
    rows: list[dict[str, str]],
    predictions: dict[int, dict[str, np.ndarray]],
    minimum_cases: int,
    minimum_gain: float,
) -> dict[str, object]:
    positive_rows = [row for row in rows if int(row["label"]) == 1]
    if not positive_rows:
        raise ValueError("held-out fold has no positive patches")
    positive_ids = {row["patch_id"] for row in positive_rows}
    for epoch, arrays in predictions.items():
        ids = arrays["positive_patch_id"].astype(str)
        if len(ids) != len(set(ids)) or set(ids) != positive_ids:
            raise ValueError(f"epoch {epoch} positive coverage mismatch")

    def best_for(subset: list[dict[str, str]]) -> tuple[int, dict[str, object]]:
        wanted = {row["patch_id"] for row in subset if int(row["label"]) == 1}
        best = None
        for epoch, arrays in sorted(predictions.items()):
            ids = arrays["positive_patch_id"].astype(str)
            keep = np.asarray([patch_id in wanted for patch_id in ids], dtype=bool)
            aggregate = aggregate_positive_dice(
                ids[keep],
                arrays["segmentation_dice_grid"][keep],
                arrays["segmentation_thresholds"],
                subset,
            )
            key = (
                float(aggregate["case_balanced_dice"]),
                float(aggregate["cohort_balanced_dice"]),
                -epoch,
            )
            if best is None or key > best[0]:
                best = (key, epoch, aggregate)
        assert best is not None
        return best[1], best[2]

    sources = sorted({semantic_source_group(row["source_name"]) for row in rows})
    source_selected = {}
    for source in sources:
        subset = [
            row
            for row in rows
            if semantic_source_group(row["source_name"]) == source
        ]
        epoch, aggregate = best_for(subset)
        source_selected[source] = (epoch, aggregate)

    assignments: dict[str, float] = {}
    strata = []
    stratum_keys = sorted(
        {
            (semantic_source_group(row["source_name"]), size_bin(row))
            for row in positive_rows
        }
    )
    for source, bin_name in stratum_keys:
        subset = [
            row
            for row in rows
            if semantic_source_group(row["source_name"]) == source
            and size_bin(row) == bin_name
        ]
        candidate_epoch, candidate = best_for(subset)
        source_epoch, source_aggregate = source_selected[source]
        gain = float(candidate["case_balanced_dice"]) - float(
            source_aggregate["case_balanced_dice"]
        )
        accepted = (
            int(candidate["positive_cases"]) >= minimum_cases
            and gain >= minimum_gain
        )
        selected_epoch = candidate_epoch if accepted else source_epoch
        selected_aggregate = candidate if accepted else source_aggregate
        arrays = predictions[selected_epoch]
        ids = arrays["positive_patch_id"].astype(str)
        position = {patch_id: index for index, patch_id in enumerate(ids)}
        threshold_index = int(selected_aggregate["best_index"])
        wanted = {row["patch_id"] for row in subset if int(row["label"]) == 1}
        for patch_id in wanted:
            assignments[patch_id] = float(
                arrays["segmentation_dice_grid"][position[patch_id], threshold_index]
            )
        strata.append(
            {
                "source": source,
                "size_bin": bin_name,
                "positive_patches": int(candidate["positive_patches"]),
                "positive_cases": int(candidate["positive_cases"]),
                "source_epoch": source_epoch,
                "candidate_epoch": candidate_epoch,
                "selected_epoch": selected_epoch,
                "source_case_dice": float(source_aggregate["case_balanced_dice"]),
                "candidate_case_dice": float(candidate["case_balanced_dice"]),
                "candidate_gain": gain,
                "selected_threshold": float(selected_aggregate["threshold"]),
                "accepted": accepted,
            }
        )

    all_ids = [row["patch_id"] for row in positive_rows]
    if set(assignments) != set(all_ids):
        raise ValueError("size-route assignment coverage mismatch")
    routed = aggregate_positive_dice(
        all_ids,
        np.asarray([[assignments[patch_id]] for patch_id in all_ids]),
        np.asarray([0.5]),
        rows,
    )

    source_assignments: dict[str, float] = {}
    for source, (epoch, aggregate) in source_selected.items():
        arrays = predictions[epoch]
        ids = arrays["positive_patch_id"].astype(str)
        position = {patch_id: index for index, patch_id in enumerate(ids)}
        threshold_index = int(aggregate["best_index"])
        for row in positive_rows:
            if semantic_source_group(row["source_name"]) == source:
                source_assignments[row["patch_id"]] = float(
                    arrays["segmentation_dice_grid"][position[row["patch_id"]], threshold_index]
                )
    baseline = aggregate_positive_dice(
        all_ids,
        np.asarray([[source_assignments[patch_id]] for patch_id in all_ids]),
        np.asarray([0.5]),
        rows,
    )
    return {
        "source_only": {
            "case_dice": float(baseline["case_balanced_dice"]),
            "cohort_dice": float(baseline["cohort_balanced_dice"]),
            "selected": {
                source: {
                    "epoch": epoch,
                    "threshold": float(aggregate["threshold"]),
                    "case_dice": float(aggregate["case_balanced_dice"]),
                }
                for source, (epoch, aggregate) in source_selected.items()
            },
        },
        "conservative_source_size": {
            "case_dice": float(routed["case_balanced_dice"]),
            "cohort_dice": float(routed["cohort_balanced_dice"]),
            "case_dice_gain": float(routed["case_balanced_dice"])
            - float(baseline["case_balanced_dice"]),
            "cohort_dice_gain": float(routed["cohort_balanced_dice"])
            - float(baseline["cohort_balanced_dice"]),
            "accepted_strata": sum(int(row["accepted"]) for row in strata),
            "strata": strata,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--epochs", default="1,4,7,10,13,16,19,22,25,28,30")
    parser.add_argument("--minimum-positive-cases", type=int, default=50)
    parser.add_argument("--minimum-within-stratum-gain", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_positive_cases < 1 or args.minimum_within_stratum_gain < 0:
        parser.error("invalid conservative routing gate")
    epochs = [int(value) for value in args.epochs.split(",")]
    if len(epochs) != len(set(epochs)) or any(epoch < 1 for epoch in epochs):
        parser.error("invalid epoch inventory")
    with args.index.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    _, held_out = grouped_fold_rows(all_rows, args.folds, args.fold, args.seed)
    paths = {
        epoch: args.run / f"validation_predictions_epoch_{epoch:04d}.npz"
        for epoch in epochs
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("one or more declared validation archives are missing")
    report = select(
        held_out,
        {epoch: load_prediction(path) for epoch, path in paths.items()},
        args.minimum_positive_cases,
        args.minimum_within_stratum_gain,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "fold": args.fold,
        "folds": args.folds,
        "seed": args.seed,
        "index": str(args.index.resolve()),
        "index_sha256": sha256(args.index),
        "prediction_sha256": {
            str(epoch): sha256(path) for epoch, path in paths.items()
        },
        "size_bins_mm3": [0, 1000, 10000, None],
        "minimum_positive_cases": args.minimum_positive_cases,
        "minimum_within_stratum_gain": args.minimum_within_stratum_gain,
        **report,
        "decision": (
            "TEST_SOURCE_SIZE_ROUTING_FIVEFOLD"
            if report["conservative_source_size"]["case_dice_gain"] >= 0.002
            and report["conservative_source_size"]["cohort_dice_gain"] >= 0.002
            else "REJECT_SOURCE_SIZE_ROUTING"
        ),
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
