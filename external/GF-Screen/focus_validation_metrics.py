"""Case-balanced aggregation for candidate-centred contour validation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def aggregate_positive_dice(
    patch_ids: list[str] | np.ndarray,
    dice_grid: np.ndarray,
    thresholds: np.ndarray,
    rows: list[dict[str, str]],
    negative_nonempty_by_threshold: np.ndarray | None = None,
    negative_patches: int = 0,
    negative_weight: float = 0.02,
) -> dict:
    """Aggregate positive-patch Dice without overweighting tiled cases."""

    ids = [str(value) for value in patch_ids]
    grid = np.asarray(dice_grid, dtype=np.float64)
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    if len(ids) != len(set(ids)):
        raise ValueError("positive patch IDs are not unique")
    if grid.shape != (len(ids), len(threshold_values)):
        raise ValueError("positive Dice grid shape mismatch")
    metadata = {str(row["patch_id"]): row for row in rows}
    if len(metadata) != len(rows):
        raise ValueError("patch index IDs are not unique")
    missing = set(ids) - set(metadata)
    if missing:
        raise ValueError(f"positive patch IDs missing from index: {len(missing)}")
    expected_positive = {
        str(row["patch_id"]) for row in rows if int(row["label"]) == 1
    }
    if set(ids) != expected_positive:
        raise ValueError(
            "positive validation coverage mismatch: "
            f"expected={len(expected_positive)} observed={len(ids)}"
        )
    by_case: dict[str, list[int]] = defaultdict(list)
    case_cohort: dict[str, str] = {}
    for index, patch_id in enumerate(ids):
        row = metadata[patch_id]
        if int(row["label"]) != 1:
            raise ValueError(f"nonpositive patch in Dice grid: {patch_id}")
        case = str(row["case"])
        cohort = str(row.get("cohort", ""))
        if case in case_cohort and case_cohort[case] != cohort:
            raise ValueError(f"case belongs to multiple cohorts: {case}")
        case_cohort[case] = cohort
        by_case[case].append(index)
    if not by_case:
        raise ValueError("no positive cases in Dice grid")
    case_names = sorted(by_case)
    case_grid = np.stack(
        [grid[by_case[case]].mean(axis=0) for case in case_names], axis=0
    )
    by_cohort: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(case_names):
        by_cohort[case_cohort[case]].append(index)
    cohort_names = sorted(by_cohort)
    cohort_grid = np.stack(
        [case_grid[by_cohort[cohort]].mean(axis=0) for cohort in cohort_names],
        axis=0,
    )
    patch_mean = grid.mean(axis=0)
    case_mean = case_grid.mean(axis=0)
    cohort_mean = cohort_grid.mean(axis=0)
    if negative_nonempty_by_threshold is None:
        negative_rate = np.zeros(len(threshold_values), dtype=np.float64)
    else:
        negative_counts = np.asarray(
            negative_nonempty_by_threshold, dtype=np.float64
        )
        if negative_counts.shape != threshold_values.shape or negative_patches < 1:
            raise ValueError("negative activation grid shape/count mismatch")
        negative_rate = negative_counts / float(negative_patches)
    selection = case_mean - float(negative_weight) * negative_rate
    best = max(
        range(len(threshold_values)),
        key=lambda index: (
            float(selection[index]),
            float(case_mean[index]),
            float(cohort_mean[index]),
            float(patch_mean[index]),
            -abs(float(threshold_values[index]) - 0.5),
        ),
    )
    threshold_0p5 = int(np.argmin(np.abs(threshold_values - 0.5)))
    return {
        "best_index": int(best),
        "threshold": float(threshold_values[best]),
        "case_balanced_dice": float(case_mean[best]),
        "case_balanced_dice_at_0p5": float(case_mean[threshold_0p5]),
        "patch_mean_dice": float(patch_mean[best]),
        "cohort_balanced_dice": float(cohort_mean[best]),
        "negative_nonempty_rate": float(negative_rate[best]),
        "selection_score": float(selection[best]),
        "positive_patches": len(ids),
        "positive_cases": len(case_names),
        "positive_cohorts": len(cohort_names),
        "patch_mean_grid": patch_mean,
        "case_balanced_grid": case_mean,
        "cohort_balanced_grid": cohort_mean,
        "negative_nonempty_rate_grid": negative_rate,
        "selection_score_grid": selection,
    }


def apply_case_balanced_metrics(
    metrics: dict[str, float],
    predictions: dict[str, np.ndarray],
    validation_rows: list[dict[str, str]],
) -> dict[str, float]:
    """Replace the primary contour selector with case-balanced Dice."""

    summary = aggregate_positive_dice(
        predictions["positive_patch_id"],
        predictions["segmentation_dice_grid"],
        predictions["segmentation_thresholds"],
        validation_rows,
        predictions.get("negative_nonempty_by_threshold"),
        int(metrics.get("negative_patches", 0)),
    )
    updated = dict(metrics)
    updated.update(
        {
            "segmentation_threshold": summary["threshold"],
            "segmentation_dice": summary["case_balanced_dice"],
            "segmentation_dice_at_0p5": summary["case_balanced_dice_at_0p5"],
            "case_balanced_segmentation_dice": summary["case_balanced_dice"],
            "patch_mean_segmentation_dice": summary["patch_mean_dice"],
            "cohort_balanced_segmentation_dice": summary["cohort_balanced_dice"],
            "positive_cases": summary["positive_cases"],
            "positive_cohorts": summary["positive_cohorts"],
            "negative_nonempty_rate": summary["negative_nonempty_rate"],
        }
    )
    updated["selection_score"] = summary["selection_score"]
    return updated
