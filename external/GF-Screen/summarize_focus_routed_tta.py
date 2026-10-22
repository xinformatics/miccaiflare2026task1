#!/usr/bin/env python3
"""Pool selected x+y TTA Focus predictions across case-held-out folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from focus_validation_metrics import aggregate_positive_dice
from summarize_focus_epoch_routes import (
    load_npz,
    route_group,
    route_group_by_patch,
    validate_prediction,
)
from train_candidate_refiner import grouped_fold_rows


MIN_CASE_GAIN = 0.002
MIN_COHORT_GAIN = 0.002
MIN_SELECTION_GAIN = 0.001
MAX_NEGATIVE_INCREASE = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def qualification(gains: dict[str, float]) -> dict[str, bool]:
    return {
        "case_gain": gains["case_balanced_dice"] >= MIN_CASE_GAIN,
        "cohort_gain": gains["cohort_balanced_dice"] >= MIN_COHORT_GAIN,
        "selection_gain": gains["selection_score"] >= MIN_SELECTION_GAIN,
        "negative_increase": gains["negative_nonempty_rate"]
        <= MAX_NEGATIVE_INCREASE,
    }


def summarize(
    route_path: Path,
    pilot_path: Path,
    tta_summaries: list[Path],
) -> dict[str, object]:
    route = json.loads(route_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if (
        route.get("public_read")
        or route.get("hidden_read")
        or route.get("submission_created")
        or pilot.get("public_read")
        or pilot.get("hidden_read")
        or pilot.get("submission_created")
    ):
        raise ValueError("routed TTA requires local-only inputs")
    eligible = tuple(pilot.get("eligible_groups", []))
    if pilot.get("decision") != "PROMOTE_TTA_TO_FIVEFOLD" or not eligible:
        raise ValueError("fold-4 TTA pilot did not promote any source group")
    if not set(eligible) <= {"weak_sources", "lung_cells"}:
        raise ValueError("invalid TTA pilot source groups")
    folds = int(route["folds"])
    seed = int(route["seed"])
    index = Path(route["index"])
    if sha256(index) != route["index_sha256"]:
        raise ValueError("routed Focus index hash mismatch")
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    summaries: dict[tuple[int, int], tuple[Path, dict[str, object]]] = {}
    for path in tta_summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (int(payload["fold"]), int(payload["epoch"]))
        if key in summaries:
            raise ValueError("duplicate fold/epoch TTA summary")
        prediction = Path(str(payload["prediction"]))
        if (
            payload.get("tta_axes") != "xy"
            or int(payload.get("tta_views", -1)) != 3
            or payload.get("index_sha256") != route["index_sha256"]
            or payload.get("public_read")
            or payload.get("hidden_read")
            or payload.get("submission_created")
            or sha256(prediction) != payload.get("prediction_sha256")
        ):
            raise ValueError("invalid x+y TTA summary identity")
        summaries[key] = (path, payload)

    fold_records = route.get("fold_records", [])
    if sorted(int(record["fold"]) for record in fold_records) != list(range(folds)):
        raise ValueError("routed Focus fold coverage mismatch")
    reports = {}
    qualified_groups = []
    common_thresholds: np.ndarray | None = None
    provenance = []
    for group in eligible:
        positive_ids: list[str] = []
        positive_grids = []
        negative_ids: list[str] = []
        negative_grids = []
        group_rows = [row for row in rows if route_group(row["source_name"]) == group]
        for record in fold_records:
            fold = int(record["fold"])
            selected = record["selected"][group]
            epoch = int(selected["epoch"])
            key = (fold, epoch)
            if key not in summaries:
                raise ValueError(f"missing TTA summary for fold {fold} epoch {epoch}")
            summary_path, summary = summaries[key]
            if summary["checkpoint_sha256"] != selected["checkpoint_sha256"]:
                raise ValueError("selected route/TTA checkpoint hash mismatch")
            _, held_out_rows = grouped_fold_rows(rows, folds, fold, seed)
            arrays = load_npz(Path(str(summary["prediction"])))
            validate_prediction(arrays, held_out_rows)
            thresholds = arrays["segmentation_thresholds"].astype(np.float64)
            if common_thresholds is None:
                common_thresholds = thresholds
            elif not np.array_equal(common_thresholds, thresholds):
                raise ValueError("TTA prediction threshold grids differ")
            mapping = route_group_by_patch(held_out_rows)
            for ids_key, grid_key, selected_ids, selected_grids in (
                ("positive_patch_id", "segmentation_dice_grid", positive_ids, positive_grids),
                ("negative_patch_id", "negative_nonempty_grid", negative_ids, negative_grids),
            ):
                identifiers = arrays[ids_key].astype(str)
                positions = [
                    index
                    for index, patch_id in enumerate(identifiers)
                    if mapping[patch_id] == group
                ]
                selected_ids.extend(identifiers[positions].tolist())
                selected_grids.append(arrays[grid_key][positions])
            provenance.append(
                {
                    "group": group,
                    "fold": fold,
                    "epoch": epoch,
                    "summary": str(summary_path.resolve()),
                    "summary_sha256": sha256(summary_path),
                    "prediction": summary["prediction"],
                    "prediction_sha256": summary["prediction_sha256"],
                    "checkpoint": summary["checkpoint"],
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                }
            )
        assert common_thresholds is not None
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
        tta_metrics = {
            "threshold": float(aggregate["threshold"]),
            "selection_score": float(aggregate["selection_score"]),
            "case_balanced_dice": float(aggregate["case_balanced_dice"]),
            "cohort_balanced_dice": float(aggregate["cohort_balanced_dice"]),
            "negative_nonempty_rate": float(aggregate["negative_nonempty_rate"]),
            "positive_patches": int(aggregate["positive_patches"]),
            "positive_cases": int(aggregate["positive_cases"]),
            "positive_cohorts": int(aggregate["positive_cohorts"]),
        }
        baseline_source = route["source_segmentation_groups"][group]
        baseline = {
            "threshold": float(baseline_source["threshold"]),
            "selection_score": float(baseline_source["selection_score"]),
            "case_balanced_dice": float(baseline_source["dice"]),
            "cohort_balanced_dice": float(
                baseline_source["cohort_balanced_dice"]
            ),
            "negative_nonempty_rate": float(
                baseline_source["negative_nonempty_rate"]
            ),
        }
        gains = {
            name: tta_metrics[name] - baseline[name]
            for name in (
                "selection_score",
                "case_balanced_dice",
                "cohort_balanced_dice",
                "negative_nonempty_rate",
            )
        }
        checks = qualification(gains)
        if all(checks.values()):
            qualified_groups.append(group)
        reports[group] = {
            "notta": baseline,
            "tta_xy": tta_metrics,
            "gains": gains,
            "checks": checks,
            "qualified_for_full_volume": all(checks.values()),
        }

    source_to_tta = {
        source: route_group(source) in qualified_groups
        for source in sorted(route["source_to_route"])
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": (
            "exact case-held-out five-fold selected-epoch x+y TTA OOF; "
            "source gates frozen before evaluation"
        ),
        "decision": (
            "QUALIFY_ROUTED_TTA_FULL_VOLUME" if qualified_groups else "KEEP_NOTTA"
        ),
        "pilot_eligible_groups": list(eligible),
        "qualified_groups": qualified_groups,
        "source_to_tta": source_to_tta,
        "thresholds": {
            "minimum_case_gain": MIN_CASE_GAIN,
            "minimum_cohort_gain": MIN_COHORT_GAIN,
            "minimum_selection_gain": MIN_SELECTION_GAIN,
            "maximum_negative_nonempty_increase": MAX_NEGATIVE_INCREASE,
        },
        "groups": reports,
        "route_summary": str(route_path.resolve()),
        "route_summary_sha256": sha256(route_path),
        "pilot": str(pilot_path.resolve()),
        "pilot_sha256": sha256(pilot_path),
        "provenance": provenance,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-summary", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--tta-summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.route_summary, args.pilot, args.tta_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
