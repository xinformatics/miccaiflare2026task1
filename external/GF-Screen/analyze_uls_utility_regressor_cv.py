#!/usr/bin/env python3
"""Nested grouped-CV ULS+ selector trained on marginal Dice utility.

Public labels define the continuous training target and inner-fold deployment
choice. Every reported public case is held out from model fitting, threshold
selection, and top-k selection. The script has no hidden-data or upload path.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_uls_feature_classifier_cv import (
    FEATURES,
    SUPPORTS,
    Candidate,
    CaseStats,
    extract_case,
    load_component_rows,
    materialize,
    parse_named_directory,
)
from analyze_uls_ranked_selector_cv import macro_dice_selected, ranked_selection


TOP_K_VALUES = (1, 2, 3, 5, 0)  # zero means no per-case cap
FIXED_THRESHOLDS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)


def utility_target(candidate: Candidate, stats: dict[str, CaseStats]) -> float:
    """Exact Dice change from adding this candidate alone to its case base."""
    case = str(candidate.row["case"])
    case_stats = stats[case]
    base_dice = 2.0 * case_stats.base_overlap / max(
        case_stats.base_voxels + case_stats.target_voxels, 1
    )
    added_voxels = int(candidate.addition_indices.size)
    added_overlap = int(candidate.row["addition_target_overlap"])
    candidate_dice = 2.0 * (case_stats.base_overlap + added_overlap) / max(
        case_stats.base_voxels + added_voxels + case_stats.target_voxels, 1
    )
    return candidate_dice - base_dice


def model_grid(seed: int):
    for alpha in (0.1, 1.0, 10.0):
        yield (
            f"ridge_a{alpha:g}",
            make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        )
    for depth, leaf in ((4, 10), (8, 5)):
        yield (
            f"forest_d{depth}_l{leaf}",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=depth,
                min_samples_leaf=leaf,
                max_features="sqrt",
                n_jobs=2,
                random_state=seed,
            ),
        )


def predict_utility(model, x_train, y_train, x_test) -> np.ndarray:
    fitted = clone(model).fit(x_train, y_train)
    return np.asarray(fitted.predict(x_test), dtype=np.float64)


def threshold_grid(predictions: np.ndarray) -> tuple[float, ...]:
    values = set(FIXED_THRESHOLDS)
    for quantile in (0.50, 0.75, 0.90, 0.95, 0.975, 0.99):
        values.add(max(0.0, float(np.quantile(predictions, quantile))))
    return tuple(sorted(values))


def nested_predictions(
    candidates: list[Candidate], stats: dict[str, CaseStats], seed: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    x = np.asarray(
        [[float(candidate.row[name]) for name in FEATURES] for candidate in candidates],
        dtype=np.float64,
    )
    y = np.asarray([utility_target(candidate, stats) for candidate in candidates])
    groups = np.asarray([str(candidate.row["case"]) for candidate in candidates])
    out_of_fold = np.full(len(candidates), np.nan, dtype=np.float64)
    folds: list[dict[str, object]] = []
    outer = GroupKFold(n_splits=5)

    for fold, (train_index, test_index) in enumerate(outer.split(x, y, groups), 1):
        x_train = x[train_index]
        y_train = y[train_index]
        train_groups = groups[train_index]
        train_candidates = [candidates[value] for value in train_index]
        train_cases = set(train_groups)
        inner = GroupKFold(n_splits=4)
        best: dict[str, object] | None = None

        for model_name, model in model_grid(seed + fold):
            inner_prediction = np.full(len(train_index), np.nan, dtype=np.float64)
            for inner_train, inner_val in inner.split(x_train, y_train, train_groups):
                inner_prediction[inner_val] = predict_utility(
                    model,
                    x_train[inner_train],
                    y_train[inner_train],
                    x_train[inner_val],
                )
            if not np.isfinite(inner_prediction).all():
                raise RuntimeError("incomplete inner grouped utility predictions")

            for threshold in threshold_grid(inner_prediction):
                for top_k in TOP_K_VALUES:
                    selected = ranked_selection(
                        train_candidates,
                        inner_prediction,
                        threshold,
                        top_k,
                        train_cases,
                    )
                    score = macro_dice_selected(
                        train_candidates, selected, train_cases, stats
                    )
                    finite_cap_preference = -(
                        top_k if top_k > 0 else len(train_candidates) + 1
                    )
                    key = (
                        score,
                        -int(selected.sum()),
                        threshold,
                        finite_cap_preference,
                        model_name,
                    )
                    if best is None or key > best["key"]:
                        best = {
                            "key": key,
                            "score": score,
                            "threshold": threshold,
                            "top_k": top_k,
                            "model_name": model_name,
                            "model": model,
                            "inner_selected": int(selected.sum()),
                        }

        if best is None:
            raise RuntimeError("no inner utility model was selected")
        out_of_fold[test_index] = predict_utility(
            best["model"], x_train, y_train, x[test_index]
        )
        test_candidates = [candidates[value] for value in test_index]
        test_cases = set(groups[test_index])
        test_selected = ranked_selection(
            test_candidates,
            out_of_fold[test_index],
            float(best["threshold"]),
            int(best["top_k"]),
            test_cases,
        )
        test_dice = macro_dice_selected(
            test_candidates, test_selected, test_cases, stats
        )
        for local_index, value in enumerate(test_index):
            row = candidates[value].row
            row["utility_target"] = float(y[value])
            row["outer_fold"] = fold
            row["oof_predicted_utility"] = float(out_of_fold[value])
            row["oof_utility_threshold"] = float(best["threshold"])
            row["oof_top_k"] = int(best["top_k"])
            row["oof_selected"] = int(test_selected[local_index])
        folds.append(
            {
                "fold": fold,
                "model": best["model_name"],
                "threshold": best["threshold"],
                "top_k": best["top_k"],
                "inner_selected_refinements": best["inner_selected"],
                "inner_oof_DSC": best["score"],
                "outer_selected_refinements": int(test_selected.sum()),
                "outer_DSC": test_dice,
            }
        )

    if not np.isfinite(out_of_fold).all():
        raise RuntimeError("incomplete outer grouped utility predictions")
    return out_of_fold, folds


def top1_oracle(candidates: list[Candidate], stats: dict[str, CaseStats]) -> tuple[float, int]:
    utilities = np.asarray([utility_target(candidate, stats) for candidate in candidates])
    selected = ranked_selection(candidates, utilities, 0.0, 1, set(stats))
    # Exclude exact-zero choices; they do not change Dice and are not useful.
    selected &= utilities > 0
    return (
        macro_dice_selected(candidates, selected, set(stats), stats),
        int(selected.sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--component-csv", type=Path, required=True)
    parser.add_argument(
        "--support", type=parse_named_directory, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    supports = dict(args.support)
    if set(supports) != set(SUPPORTS):
        raise ValueError(f"supports must be exactly {sorted(SUPPORTS)}")

    records = json.loads((args.work / "manifest.json").read_text())
    by_case: dict[str, list[dict]] = {}
    for record in records:
        by_case.setdefault(record["case"], []).append(record)
    if len(by_case) != 50:
        raise ValueError(f"expected 50 manifest cases, found {len(by_case)}")

    component_rows = load_component_rows(args.component_csv)
    candidates: list[Candidate] = []
    stats: dict[str, CaseStats] = {}
    for index, (case, case_records) in enumerate(sorted(by_case.items()), 1):
        case_candidates, case_stats = extract_case(
            case,
            case_records,
            component_rows,
            args.images,
            args.labels,
            args.base,
            supports,
            args.work / "predicted_crops",
        )
        candidates.extend(case_candidates)
        stats[case] = case_stats
        print(
            f"[{index:02d}/50] {case} ULS_candidates={len(case_candidates)}",
            flush=True,
        )

    _, folds = nested_predictions(candidates, stats, args.seed)
    selected = np.asarray(
        [bool(int(candidate.row["oof_selected"])) for candidate in candidates]
    )
    oof_dice = macro_dice_selected(candidates, selected, set(stats), stats)
    anchor = fmean(
        2.0 * value.base_overlap / max(value.base_voxels + value.target_voxels, 1)
        for value in stats.values()
    )
    oracle_dice, oracle_selected = top1_oracle(candidates, stats)
    report = {
        "warning": "Public-only nested case-grouped utility regressor; no hidden model exists.",
        "cases": len(stats),
        "candidates": len(candidates),
        "selected_refinements": int(selected.sum()),
        "positive_utility_candidates": sum(
            utility_target(candidate, stats) > 0 for candidate in candidates
        ),
        "anchor_macro_DSC": anchor,
        "oof_macro_DSC": oof_dice,
        "oof_macro_DSC_gain": oof_dice - anchor,
        "top1_oracle_DSC": oracle_dice,
        "top1_oracle_gain": oracle_dice - anchor,
        "top1_oracle_selected": oracle_selected,
        "features": FEATURES,
        "folds": folds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [candidate.row for candidate in candidates]
    with (args.output_dir / "uls_utility_candidates_oof.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    materialize(candidates, stats, args.base, args.output_dir / "predictions_oof")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
