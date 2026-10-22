#!/usr/bin/env python3
"""Nested grouped ULS+ selector with a per-case false-positive budget.

This is a public-only diagnostic. Public labels define refinement targets and
inner-fold Dice, but every reported prediction is held out by case from model,
threshold, and top-k selection. No hidden-data or submission path exists here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np
from sklearn.model_selection import GroupKFold

from analyze_component_classifier_cv import model_grid, probability
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


THRESHOLDS = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))
TOP_K_VALUES = (1, 2, 3, 5, 8, 0)  # zero means no per-case cap


def ranked_selection(
    candidates: list[Candidate],
    probabilities: np.ndarray,
    threshold: float,
    top_k: int,
    cases: set[str],
) -> np.ndarray:
    """Select thresholded candidates with a deterministic cap in each case."""
    if len(candidates) != len(probabilities):
        raise ValueError("candidate and probability lengths differ")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    selected = np.zeros(len(candidates), dtype=bool)
    by_case: dict[str, list[int]] = {case: [] for case in cases}
    for index, (candidate, value) in enumerate(zip(candidates, probabilities)):
        case = str(candidate.row["case"])
        if case in by_case and float(value) >= threshold:
            by_case[case].append(index)
    for indices in by_case.values():
        indices.sort(
            key=lambda index: (
                -float(probabilities[index]),
                int(candidates[index].row["component_id"]),
                str(candidates[index].row["crop_id"]),
            )
        )
        keep = indices if top_k == 0 else indices[:top_k]
        selected[keep] = True
    return selected


def macro_dice_selected(
    candidates: list[Candidate],
    selected: np.ndarray,
    cases: set[str],
    stats: dict[str, CaseStats],
) -> float:
    """Compute exact case-macro Dice after unioning selected sparse additions."""
    if len(candidates) != len(selected):
        raise ValueError("candidate and selection lengths differ")
    chosen: dict[str, list[np.ndarray]] = {case: [] for case in cases}
    for candidate, keep in zip(candidates, selected):
        case = str(candidate.row["case"])
        if keep and case in chosen and candidate.addition_indices.size:
            chosen[case].append(candidate.addition_indices)
    scores = []
    for case in sorted(cases):
        additions = chosen[case]
        if additions:
            added = np.unique(np.concatenate(additions))
            added_overlap = np.intersect1d(
                added, stats[case].target_indices, assume_unique=True
            ).size
        else:
            added = np.empty(0, dtype=np.int64)
            added_overlap = 0
        predicted = stats[case].base_voxels + int(added.size)
        overlap = stats[case].base_overlap + int(added_overlap)
        scores.append(
            2.0 * overlap / max(predicted + stats[case].target_voxels, 1)
        )
    return fmean(scores)


def nested_ranked_predictions(
    candidates: list[Candidate], stats: dict[str, CaseStats], seed: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    x = np.asarray(
        [[float(candidate.row[name]) for name in FEATURES] for candidate in candidates],
        dtype=np.float64,
    )
    y = np.asarray(
        [int(candidate.row["is_true_refinement"]) for candidate in candidates],
        dtype=np.int64,
    )
    groups = np.asarray([str(candidate.row["case"]) for candidate in candidates])
    if len(np.unique(groups)) < 10:
        raise ValueError("at least ten candidate-containing cases are required")

    out_of_fold = np.full(len(candidates), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    outer = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(outer.split(x, y, groups), 1):
        x_train = x[train_index]
        y_train = y[train_index]
        train_groups = groups[train_index]
        train_candidates = [candidates[value] for value in train_index]
        train_cases = set(train_groups)
        inner = GroupKFold(n_splits=min(4, len(train_cases)))
        best: dict[str, object] | None = None

        for model_name, model in model_grid(seed + fold):
            inner_probability = np.full(len(train_index), np.nan, dtype=np.float64)
            for inner_train, inner_val in inner.split(x_train, y_train, train_groups):
                inner_probability[inner_val] = probability(
                    model,
                    x_train[inner_train],
                    y_train[inner_train],
                    x_train[inner_val],
                )
            if not np.isfinite(inner_probability).all():
                raise RuntimeError("incomplete inner grouped predictions")

            for threshold in THRESHOLDS:
                for top_k in TOP_K_VALUES:
                    selected = ranked_selection(
                        train_candidates,
                        inner_probability,
                        threshold,
                        top_k,
                        train_cases,
                    )
                    score = macro_dice_selected(
                        train_candidates, selected, train_cases, stats
                    )
                    # Exact ties prefer fewer additions, a higher threshold,
                    # and then a smaller finite cap. This is conservative and
                    # deterministic without looking at the outer fold.
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
            raise RuntimeError("no inner model was selected")
        out_of_fold[test_index] = probability(
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
            row["outer_fold"] = fold
            row["oof_probability"] = float(out_of_fold[value])
            row["oof_threshold"] = float(best["threshold"])
            row["oof_top_k"] = int(best["top_k"])
            row["oof_selected"] = int(test_selected[local_index])
        fold_rows.append(
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
        raise RuntimeError("incomplete outer grouped predictions")
    return out_of_fold, fold_rows


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

    _, folds = nested_ranked_predictions(candidates, stats, args.seed)
    cases = set(stats)
    selected_mask = np.asarray(
        [bool(int(candidate.row["oof_selected"])) for candidate in candidates]
    )
    oof_dice = macro_dice_selected(candidates, selected_mask, cases, stats)
    anchor = fmean(
        2.0 * value.base_overlap
        / max(value.base_voxels + value.target_voxels, 1)
        for value in stats.values()
    )
    selected = [candidate for candidate in candidates if int(candidate.row["oof_selected"])]
    report = {
        "warning": "Public-only nested case-grouped ranked ULS+ selector; no hidden model exists.",
        "cases": len(stats),
        "candidates": len(candidates),
        "true_refinements": sum(
            int(candidate.row["is_true_refinement"]) for candidate in candidates
        ),
        "selected_refinements": len(selected),
        "true_selected_refinements": sum(
            int(candidate.row["is_true_refinement"]) for candidate in selected
        ),
        "anchor_macro_DSC": anchor,
        "oof_macro_DSC": oof_dice,
        "oof_macro_DSC_gain": oof_dice - anchor,
        "features": FEATURES,
        "threshold_grid": THRESHOLDS,
        "top_k_grid": TOP_K_VALUES,
        "folds": folds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [candidate.row for candidate in candidates]
    with (args.output_dir / "uls_ranked_candidates_oof.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    materialize(candidates, stats, args.base, args.output_dir / "predictions_oof")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
