#!/usr/bin/env python3
"""Nested case-held-out evaluation of a conservative GF-base deletion gate.

The input table is mined only from complete-label training cases.  Labels and
per-component overlap are supervision/evaluation fields; model inputs are the
fixed inference-visible features in ``FEATURES``.  Model family and deletion
threshold are selected inside each outer training split.  The final report uses
the official DSC/NSD implementation on outer-fold predictions and never reads
challenge public or hidden validation data.  This OOF result can qualify only
components traceable to the frozen GF base.  It cannot by itself promote a
composition with the champion's separately guided disconnected additions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

from analyze_component_classifier_cv import model_grid, probability
from subtractive_gate_features import FEATURES


@dataclass(frozen=True)
class CaseStats:
    # These values may be expressed either in voxels or in mm3. Dice is
    # scale-invariant, and the frozen component table stores exact physical
    # volumes. Using those values avoids relabeling every very large CT merely
    # to reproduce statistics that the audited builder already computed.
    pred_voxels: float
    target_voxels: float
    overlap_voxels: float
    component_voxels: dict[int, float]
    component_overlap: dict[int, float]


def stable_digest(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def case_fold_map(
    items: list[dict[str, object]], folds: int, seed: int
) -> dict[str, int]:
    """Assign whole cases to cohort-balanced deterministic folds."""
    if folds < 2:
        raise ValueError("at least two folds are required")
    by_cohort: dict[str, set[str]] = {}
    for item in items:
        case = str(item["name"])
        cohort = str(item["cohort"])
        by_cohort.setdefault(cohort, set()).add(case)
    assignment: dict[str, int] = {}
    for cohort, cases in sorted(by_cohort.items()):
        ordered = sorted(cases, key=lambda case: stable_digest(f"{cohort}:{case}", seed))
        for index, case in enumerate(ordered):
            assignment[case] = index % folds
    if len(assignment) != len({str(item["name"]) for item in items}):
        raise AssertionError("case fold assignment is incomplete")
    return assignment


def dice(stats: CaseStats, deleted: set[int]) -> float:
    removed_voxels = sum(stats.component_voxels[value] for value in deleted)
    removed_overlap = sum(stats.component_overlap[value] for value in deleted)
    prediction = stats.pred_voxels - removed_voxels
    overlap = stats.overlap_voxels - removed_overlap
    denominator = prediction + stats.target_voxels
    return 1.0 if denominator == 0 else 2.0 * overlap / denominator


def selected_deletions(
    rows: list[dict[str, object]],
    probabilities: np.ndarray,
    threshold: float,
    minimum_deletion_mm3_exclusive: float = 0.0,
) -> dict[str, set[int]]:
    if len(rows) != len(probabilities):
        raise ValueError("row/probability length mismatch")
    result: dict[str, set[int]] = {}
    for row, value in zip(rows, probabilities):
        if (
            float(value) >= threshold
            and float(row.get("mm3", math.inf)) > minimum_deletion_mm3_exclusive
        ):
            result.setdefault(str(row["case"]), set()).add(int(row["comp"]))
    return result


def macro_dice_after_deletions(
    rows: list[dict[str, object]],
    probabilities: np.ndarray,
    threshold: float,
    cases: set[str],
    stats: dict[str, CaseStats],
    minimum_deletion_mm3_exclusive: float = 0.0,
) -> float:
    deleted = selected_deletions(
        rows, probabilities, threshold, minimum_deletion_mm3_exclusive
    )
    return fmean(dice(stats[case], deleted.get(case, set())) for case in sorted(cases))


def deletion_dsc_summary(
    rows: list[dict[str, object]],
    probabilities: np.ndarray,
    threshold: float,
    cases: set[str],
    stats: dict[str, CaseStats],
    minimum_deletion_mm3_exclusive: float = 0.0,
) -> tuple[float, float]:
    """Return candidate macro DSC and rate of case losses larger than 0.05."""
    deleted = selected_deletions(
        rows, probabilities, threshold, minimum_deletion_mm3_exclusive
    )
    pairs = [
        (
            dice(stats[case], set()),
            dice(stats[case], deleted.get(case, set())),
        )
        for case in sorted(cases)
    ]
    return (
        fmean(candidate for _, candidate in pairs),
        fmean(candidate < anchor - 0.05 for anchor, candidate in pairs),
    )


def feature_matrix(rows: list[dict[str, object]]) -> np.ndarray:
    matrix = np.asarray(
        [[float(row[name]) for name in FEATURES] for row in rows], dtype=np.float64
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES):
        raise ValueError("invalid feature matrix")
    # Shell statistics can be missing at a volume boundary.  Impute with a
    # fixed inference-safe sentinel rather than deriving a global median from
    # held-out data.
    return np.nan_to_num(matrix, nan=-2048.0, posinf=4096.0, neginf=-4096.0)


def threshold_grid() -> np.ndarray:
    # 1.01 is the unchanged-anchor option and makes every inner selection
    # explicitly prove that a deletion rule is better than deleting nothing.
    return np.concatenate((np.linspace(0.50, 0.99, 50), np.asarray([1.01])))


def nested_group_predictions(
    rows: list[dict[str, object]],
    items: list[dict[str, object]],
    stats: dict[str, CaseStats],
    seed: int,
    inner_catastrophic_cap: float | None = None,
    minimum_deletion_mm3_exclusive: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if inner_catastrophic_cap is not None and not 0.0 <= inner_catastrophic_cap <= 1.0:
        raise ValueError("inner catastrophic cap must be between zero and one")
    if minimum_deletion_mm3_exclusive < 0.0:
        raise ValueError("minimum deletion volume must be nonnegative")
    x = feature_matrix(rows)
    y = np.asarray([1 - int(row["is_tp"]) for row in rows], dtype=np.int64)
    cases = np.asarray([str(row["case"]) for row in rows])
    outer_assignment = case_fold_map(items, 5, seed)
    out = np.full(len(rows), np.nan, dtype=np.float64)
    records: list[dict[str, object]] = []
    all_cases = {str(item["name"]) for item in items}

    for fold in range(5):
        test_cases = {case for case in all_cases if outer_assignment[case] == fold}
        train_cases = all_cases - test_cases
        train_index = np.flatnonzero(np.asarray([case in train_cases for case in cases]))
        test_index = np.flatnonzero(np.asarray([case in test_cases for case in cases]))
        if not len(train_index) or not len(test_index):
            raise ValueError(f"empty component rows in outer fold {fold}")

        train_items = [item for item in items if str(item["name"]) in train_cases]
        inner_assignment = case_fold_map(train_items, 4, seed + 1000 + fold)
        inner_probability_by_model: dict[str, np.ndarray] = {}
        models = list(model_grid(seed + fold))
        for model_name, model in models:
            inner_probability = np.full(len(train_index), np.nan, dtype=np.float64)
            for inner_fold in range(4):
                inner_val_cases = {
                    case for case in train_cases if inner_assignment[case] == inner_fold
                }
                inner_val_local = np.flatnonzero(
                    np.asarray([cases[index] in inner_val_cases for index in train_index])
                )
                inner_train_local = np.flatnonzero(
                    np.asarray([cases[index] not in inner_val_cases for index in train_index])
                )
                if not len(inner_val_local) or not len(inner_train_local):
                    raise ValueError(f"empty inner split {fold}/{inner_fold}")
                inner_probability[inner_val_local] = probability(
                    model,
                    x[train_index[inner_train_local]],
                    y[train_index[inner_train_local]],
                    x[train_index[inner_val_local]],
                )
            if not np.isfinite(inner_probability).all():
                raise RuntimeError("incomplete inner OOF probabilities")
            inner_probability_by_model[model_name] = inner_probability

        train_rows = [rows[index] for index in train_index]
        best: tuple[float, float, str, object, float] | None = None
        for model_name, model in models:
            inner_probability = inner_probability_by_model[model_name]
            for threshold in threshold_grid():
                score, inner_catastrophic = deletion_dsc_summary(
                    train_rows,
                    inner_probability,
                    float(threshold),
                    train_cases,
                    stats,
                    minimum_deletion_mm3_exclusive,
                )
                if (
                    inner_catastrophic_cap is not None
                    and inner_catastrophic > inner_catastrophic_cap
                ):
                    continue
                candidate = (
                    score,
                    float(threshold),
                    model_name,
                    model,
                    inner_catastrophic,
                )
                # On a metric tie, prefer the higher (more conservative)
                # deletion threshold, then a stable model name.
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        assert best is not None
        inner_dice, threshold, model_name, model, inner_catastrophic = best
        out[test_index] = probability(model, x[train_index], y[train_index], x[test_index])
        test_rows = [rows[index] for index in test_index]
        outer_dice, outer_catastrophic = deletion_dsc_summary(
            test_rows,
            out[test_index],
            threshold,
            test_cases,
            stats,
            minimum_deletion_mm3_exclusive,
        )
        for index in test_index:
            rows[index]["outer_fold"] = fold
            rows[index]["oof_fp_probability"] = float(out[index])
            rows[index]["oof_threshold"] = threshold
            rows[index]["oof_delete"] = int(
                out[index] >= threshold
                and float(rows[index].get("mm3", math.inf))
                > minimum_deletion_mm3_exclusive
            )
        records.append(
            {
                "fold": fold,
                "train_cases": len(train_cases),
                "test_cases": len(test_cases),
                "model": model_name,
                "threshold": threshold,
                "inner_oof_DSC": inner_dice,
                "inner_oof_catastrophic_regression_rate": inner_catastrophic,
                "outer_DSC": outer_dice,
                "outer_catastrophic_regression_rate": outer_catastrophic,
            }
        )
    if not np.isfinite(out).all():
        raise RuntimeError("incomplete outer OOF probabilities")
    return out, records


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows: list[dict[str, object]] = list(csv.DictReader(stream))
    required = {"case", "comp", "is_tp", *FEATURES}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"component table missing fields: {sorted(required-set(rows[0]) if rows else required)}")
    keys = [(str(row["case"]), int(row["comp"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate case/component rows")
    for row in rows:
        if int(row["is_tp"]) not in (0, 1):
            raise ValueError("is_tp must be binary")
    return rows


def load_manifest(path: Path, expected_cases: int) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items: list[dict[str, object]] = payload["items"]
    if len(items) != expected_cases or len({str(item["name"]) for item in items}) != expected_cases:
        raise ValueError("manifest case count/uniqueness failure")
    if any(item.get("complete_label") is not True for item in items):
        raise ValueError("subtractive gate requires complete training labels")
    return items


def load_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asarray(image.dataobj) > 0, image


def build_case_stats(
    items: list[dict[str, object]], predictions: Path, min_mm3: float
) -> tuple[dict[str, CaseStats], dict[str, tuple[np.ndarray, nib.Nifti1Image]]]:
    stats: dict[str, CaseStats] = {}
    # Arrays are intentionally not retained; the second return value exists to
    # make the no-cache policy explicit and testable through an empty mapping.
    retained: dict[str, tuple[np.ndarray, nib.Nifti1Image]] = {}
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for index, item in enumerate(items, 1):
        case = str(item["name"])
        pred_path = predictions / str(item["prediction_name"])
        pred, pred_image = load_mask(pred_path)
        target, target_image = load_mask(Path(str(item["label"])))
        if pred.shape != target.shape or not np.allclose(
            pred_image.affine, target_image.affine, atol=1e-3
        ):
            raise ValueError(f"geometry mismatch: {case}")
        component_map, count = ndimage.label(pred, structure=structure)
        sizes = np.asarray(
            ndimage.sum(pred, component_map, index=np.arange(1, count + 1)),
            dtype=np.int64,
        )
        voxel_mm3 = float(abs(np.linalg.det(pred_image.affine[:3, :3])))
        keep = [value for value in range(1, count + 1) if sizes[value - 1] * voxel_mm3 >= min_mm3]
        deployed = np.isin(component_map, keep)
        component_voxels = {value: int(sizes[value - 1]) for value in keep}
        component_overlap = {
            value: int(np.count_nonzero((component_map == value) & target)) for value in keep
        }
        stats[case] = CaseStats(
            pred_voxels=int(np.count_nonzero(deployed)),
            target_voxels=int(np.count_nonzero(target)),
            overlap_voxels=int(np.count_nonzero(deployed & target)),
            component_voxels=component_voxels,
            component_overlap=component_overlap,
        )
        if index % 25 == 0:
            print(f"[{index}/{len(items)}] reconstructed component statistics", flush=True)
    return stats, retained


def build_case_stats_from_rows(
    items: list[dict[str, object]], rows: list[dict[str, object]]
) -> dict[str, CaseStats]:
    """Reconstruct exact Dice sufficient statistics from the frozen table.

    ``mm3`` and ``case_pred_mm3`` share a case-specific scale, while
    ``overlap_frac`` and ``base_dice`` were computed from the integer masks by
    the audited table builder. Therefore the target volume follows from
    ``base_dice = 2 * overlap / (prediction + target)``. Cases with no retained
    component, and zero-overlap cases, contribute the same constant to every
    deletion rule; a unit non-empty target is sufficient for model selection.
    Full masks are still used later for the official DSC/NSD pass, but only if
    the cheap held-out DSC checks have not already rejected the gate.
    """
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case"]), []).append(row)
    expected = {str(item["name"]) for item in items}
    extra = set(grouped) - expected
    if extra:
        raise ValueError(f"component table has cases outside manifest: {sorted(extra)[:5]}")

    stats: dict[str, CaseStats] = {}
    for case in sorted(expected):
        case_rows = grouped.get(case, [])
        if not case_rows:
            stats[case] = CaseStats(0.0, 1.0, 0.0, {}, {})
            continue
        prediction_values = {float(row["case_pred_mm3"]) for row in case_rows}
        base_values = {float(row["base_dice"]) for row in case_rows}
        if len(prediction_values) != 1 or len(base_values) != 1:
            raise ValueError(f"inconsistent case totals in component table: {case}")
        prediction = prediction_values.pop()
        base = base_values.pop()
        component = {int(row["comp"]): float(row["mm3"]) for row in case_rows}
        component_overlap = {
            int(row["comp"]): float(row["overlap_frac"]) * float(row["mm3"])
            for row in case_rows
        }
        overlap = float(sum(component_overlap.values()))
        if base > 0.0:
            target = 2.0 * overlap / base - prediction
            if target < -1e-5:
                raise ValueError(f"negative reconstructed target volume: {case}")
            target = max(0.0, target)
        else:
            if abs(overlap) > 1e-6:
                raise ValueError(f"zero base Dice with nonzero overlap: {case}")
            target = 1.0
        value = CaseStats(
            prediction, target, overlap, component, component_overlap
        )
        if base > 0.0 and not math.isclose(dice(value, set()), base, abs_tol=1e-9):
            raise ValueError(f"failed to reproduce frozen base Dice: {case}")
        stats[case] = value
    return stats


def table_dsc_metrics(
    items: list[dict[str, object]],
    rows: list[dict[str, object]],
    stats: dict[str, CaseStats],
) -> list[dict[str, object]]:
    deleted: dict[str, set[int]] = {}
    for row in rows:
        if int(row["oof_delete"]):
            deleted.setdefault(str(row["case"]), set()).add(int(row["comp"]))
    return [
        {
            "case": str(item["name"]),
            "cohort": str(item["cohort"]),
            "anchor_DSC": dice(stats[str(item["name"])], set()),
            "candidate_DSC": dice(
                stats[str(item["name"])], deleted.get(str(item["name"]), set())
            ),
            "deleted_components": len(deleted.get(str(item["name"]), set())),
        }
        for item in items
    ]


def stratified_bootstrap_one(
    metrics: list[dict[str, object]], metric: str, seed: int, iterations: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    by_cohort = [
        np.asarray(
            [index for index, row in enumerate(metrics) if str(row["cohort"]) == cohort],
            dtype=np.int64,
        )
        for cohort in sorted({str(row["cohort"]) for row in metrics})
    ]
    delta = np.asarray(
        [
            float(row[f"candidate_{metric}"]) - float(row[f"anchor_{metric}"])
            for row in metrics
        ],
        dtype=np.float64,
    )
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in by_cohort]
        )
        draws[iteration] = float(np.mean(delta[sampled]))
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def validate_row_identity(
    rows: list[dict[str, object]], stats: dict[str, CaseStats]
) -> None:
    expected = {
        (case, component)
        for case, value in stats.items()
        for component in value.component_voxels
    }
    observed = {(str(row["case"]), int(row["comp"])) for row in rows}
    if observed != expected:
        raise ValueError(
            "component table does not reproduce deployed masks: "
            f"missing={sorted(expected-observed)[:5]} extra={sorted(observed-expected)[:5]}"
        )


def score_official_oof(
    items: list[dict[str, object]],
    predictions: Path,
    rows: list[dict[str, object]],
    min_mm3: float,
    workers: int = 1,
    anchor_cache: dict[str, tuple[float, float, str]] | None = None,
) -> list[dict[str, object]]:
    deleted: dict[str, set[int]] = {}
    for row in rows:
        if int(row["oof_delete"]):
            deleted.setdefault(str(row["case"]), set()).add(int(row["comp"]))
    tasks = [
        (
            item,
            str(predictions),
            sorted(deleted.get(str(item["name"]), set())),
            min_mm3,
            None if anchor_cache is None else anchor_cache[str(item["name"])],
        )
        for item in items
    ]
    result: list[dict[str, object]] = []
    if workers == 1:
        iterator = map(score_official_case, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(score_official_case, tasks, chunksize=1)
    try:
        for index, value in enumerate(iterator, 1):
            result.append(value)
            if index % 25 == 0:
                print(f"[{index}/{len(items)}] official OOF metrics", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return result


def crop_to_union(
    target: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Crop masks to their exact joint foreground box.

    The official SurfaceDice implementation performs this identical crop
    internally before its distance transforms. Doing it before connected
    components prevents empty 1024x1024 planes from dominating runtime and
    memory while preserving DSC and NSD exactly.
    """
    if target.shape != prediction.shape:
        raise ValueError("target/prediction shape mismatch")
    support = target | prediction
    if not np.any(support):
        empty = np.zeros((1,) * target.ndim, dtype=bool)
        return empty, empty.copy()
    slices = []
    for axis in range(support.ndim):
        other = tuple(value for value in range(support.ndim) if value != axis)
        occupied = np.flatnonzero(np.any(support, axis=other))
        slices.append(slice(int(occupied[0]), int(occupied[-1]) + 1))
    region = tuple(slices)
    return target[region], prediction[region]


def score_official_case(
    task: tuple[
        dict[str, object],
        str,
        list[int],
        float,
        tuple[float, float, str] | None,
    ]
) -> dict[str, object]:
    item, predictions_text, deleted_values, min_mm3, cached_anchor = task
    root = Path(__file__).resolve().parents[2]
    metric_root = root / "handoff_bundle" / "code"
    if str(metric_root) not in sys.path:
        sys.path.insert(0, str(metric_root))
    from evaluate import score_case

    case = str(item["name"])
    if cached_anchor is not None:
        cached_dsc, cached_nsd, cached_cohort = cached_anchor
        if cached_cohort != str(item["cohort"]):
            raise ValueError(f"anchor cache cohort mismatch: {case}")
        if not deleted_values:
            return {
                "case": case,
                "cohort": cached_cohort,
                "anchor_DSC": cached_dsc,
                "anchor_NSD": cached_nsd,
                "candidate_DSC": cached_dsc,
                "candidate_NSD": cached_nsd,
                "deleted_components": 0,
            }

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    predictions = Path(predictions_text)
    pred, pred_image = load_mask(predictions / str(item["prediction_name"]))
    target, target_image = load_mask(Path(str(item["label"])))
    if pred.shape != target.shape or not np.allclose(
        pred_image.affine, target_image.affine, atol=1e-3
    ):
        raise ValueError(f"geometry mismatch: {case}")
    target, pred = crop_to_union(target, pred)
    component_map, count = ndimage.label(pred, structure=structure)
    sizes = np.asarray(
        ndimage.sum(pred, component_map, index=np.arange(1, count + 1)), dtype=np.int64
    )
    voxel_mm3 = float(abs(np.linalg.det(pred_image.affine[:3, :3])))
    keep = [
        value
        for value in range(1, count + 1)
        if sizes[value - 1] * voxel_mm3 >= min_mm3
    ]
    anchor = np.isin(component_map, keep)
    candidate = anchor & ~np.isin(component_map, deleted_values)
    spacing = target_image.header.get_zooms()[:3]
    if cached_anchor is None:
        anchor_dsc, anchor_nsd = score_case(
            target.astype(np.uint8), anchor.astype(np.uint8), spacing
        )
    else:
        anchor_dsc, anchor_nsd, _ = cached_anchor
    if deleted_values:
        candidate_dsc, candidate_nsd = score_case(
            target.astype(np.uint8), candidate.astype(np.uint8), spacing
        )
    else:
        candidate_dsc, candidate_nsd = anchor_dsc, anchor_nsd
    return {
        "case": case,
        "cohort": str(item["cohort"]),
        "anchor_DSC": float(anchor_dsc),
        "anchor_NSD": float(anchor_nsd),
        "candidate_DSC": float(candidate_dsc),
        "candidate_NSD": float(candidate_nsd),
        "deleted_components": len(deleted_values),
    }


def load_anchor_metrics_cache(
    path: Path,
    items: list[dict[str, object]],
    table_metrics: list[dict[str, object]],
) -> dict[str, tuple[float, float, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"case", "cohort", "anchor_DSC", "anchor_NSD"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("anchor metrics cache is missing required fields")
    observed = {str(row["case"]): row for row in rows}
    expected = {str(item["name"]): item for item in items}
    if len(rows) != len(observed) or set(observed) != set(expected):
        raise ValueError("anchor metrics cache case coverage mismatch")
    table = {str(row["case"]): float(row["anchor_DSC"]) for row in table_metrics}
    result: dict[str, tuple[float, float, str]] = {}
    for case, item in expected.items():
        row = observed[case]
        dsc = float(row["anchor_DSC"])
        nsd = float(row["anchor_NSD"])
        cohort = str(row["cohort"])
        if cohort != str(item["cohort"]):
            raise ValueError(f"anchor cache cohort mismatch: {case}")
        if not math.isclose(dsc, table[case], rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"anchor cache/table DSC mismatch: {case}")
        if not math.isfinite(nsd):
            raise ValueError(f"nonfinite anchor NSD cache value: {case}")
        result[case] = (dsc, nsd, cohort)
    return result


def stratified_bootstrap(
    metrics: list[dict[str, object]], seed: int, iterations: int
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    by_cohort: dict[str, np.ndarray] = {}
    for cohort in sorted({str(row["cohort"]) for row in metrics}):
        indices = [index for index, row in enumerate(metrics) if str(row["cohort"]) == cohort]
        by_cohort[cohort] = np.asarray(indices, dtype=np.int64)
    deltas = {
        metric: np.asarray(
            [float(row[f"candidate_{metric}"]) - float(row[f"anchor_{metric}"]) for row in metrics],
            dtype=np.float64,
        )
        for metric in ("DSC", "NSD")
    }
    draws = {metric: np.empty(iterations, dtype=np.float64) for metric in deltas}
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in by_cohort.values()]
        )
        for metric in deltas:
            draws[metric][iteration] = float(np.mean(deltas[metric][sampled]))
    return {
        metric: [float(value) for value in np.percentile(draws[metric], [2.5, 97.5])]
        for metric in draws
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=707)
    parser.add_argument("--min-mm3", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--anchor-metrics-cache",
        type=Path,
        default=None,
        help="Optional audited TSV of anchor metrics from the identical frozen base.",
    )
    parser.add_argument(
        "--minimum-deletion-mm3-exclusive",
        type=float,
        default=0.0,
        help="Never delete components at or below this physical volume.",
    )
    parser.add_argument(
        "--inner-catastrophic-cap",
        type=float,
        default=None,
        help=(
            "Optional maximum inner-OOF fraction of cases losing >0.05 DSC; "
            "the unchanged threshold remains available as a safe anchor."
        ),
    )
    args = parser.parse_args()
    if (
        args.expected_cases < 10
        or args.min_mm3 <= 0
        or args.bootstrap < 1000
        or args.workers < 1
        or args.minimum_deletion_mm3_exclusive < 0.0
    ):
        parser.error("invalid evaluation configuration")
    if args.inner_catastrophic_cap is not None and not (
        0.0 <= args.inner_catastrophic_cap <= 1.0
    ):
        parser.error("--inner-catastrophic-cap must be between zero and one")

    os.nice(19)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.dataset)
    items = load_manifest(args.manifest, args.expected_cases)
    prediction_marker = args.predictions / "PROPOSALS_COMPLETE.json"
    if not prediction_marker.is_file():
        raise ValueError(f"missing frozen prediction marker: {prediction_marker}")
    stats = build_case_stats_from_rows(items, rows)
    validate_row_identity(rows, stats)
    probabilities, folds = nested_group_predictions(
        rows,
        items,
        stats,
        args.seed,
        inner_catastrophic_cap=args.inner_catastrophic_cap,
        minimum_deletion_mm3_exclusive=args.minimum_deletion_mm3_exclusive,
    )

    with (args.output / "components_oof.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    # DSC is exactly recoverable from the audited component table. Apply all
    # decisive DSC-only rejection checks before the expensive full-volume NSD
    # pass. A failed gate can never become qualified through NSD, so stopping
    # here is both statistically equivalent and substantially faster.
    table_metrics = table_dsc_metrics(items, rows, stats)
    table_anchor_dsc = fmean(float(row["anchor_DSC"]) for row in table_metrics)
    table_candidate_dsc = fmean(float(row["candidate_DSC"]) for row in table_metrics)
    table_dsc_gain = table_candidate_dsc - table_anchor_dsc
    table_dsc_ci = stratified_bootstrap_one(
        table_metrics, "DSC", args.seed, args.bootstrap
    )
    table_catastrophic = fmean(
        float(row["candidate_DSC"]) < float(row["anchor_DSC"]) - 0.05
        for row in table_metrics
    )
    screening_checks = {
        "DSC_gain_at_least_0p005": table_dsc_gain >= 0.005,
        "DSC_95ci_positive": table_dsc_ci[0] > 0.0,
        "catastrophic_regression_rate_at_most_0p02": table_catastrophic <= 0.02,
    }
    with (args.output / "table_dsc_metrics.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_metrics[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(table_metrics)

    if not all(screening_checks.values()):
        checks = {
            **screening_checks,
            "NSD_nonregression": False,
            "NSD_95ci_above_minus_0p002": False,
        }
        payload: dict[str, object] = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "decision": "KEEP_UNFILTERED_GF_BASE",
            "passed": False,
            "cases": len(items),
            "components": len(rows),
            "features": FEATURES,
            "target": "pure false-positive component (zero overlap with complete training label)",
            "scope": (
                "GF-base components only; separately guided champion additions are "
                "out of domain and must remain untouched"
            ),
            "rejection_stage": "exact_table_DSC_screen",
            "official_full_volume_metrics_computed": False,
            "official_full_volume_metrics_skip_reason": (
                "At least one necessary DSC/catastrophic-regression qualification "
                "check failed; NSD cannot reverse this rejection."
            ),
            "outer_folds": folds,
            "inner_catastrophic_cap": args.inner_catastrophic_cap,
            "minimum_deletion_mm3_exclusive": args.minimum_deletion_mm3_exclusive,
            "table_means": {
                "anchor_DSC": table_anchor_dsc,
                "candidate_DSC": table_candidate_dsc,
            },
            "paired_gains": {"DSC": table_dsc_gain, "NSD": None},
            "stratified_bootstrap_95ci": {"DSC": table_dsc_ci, "NSD": None},
            "catastrophic_regression_rate": table_catastrophic,
            "checks": checks,
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "predictions": str(args.predictions.resolve()),
            "prediction_marker": str(prediction_marker.resolve()),
            "prediction_marker_sha256": hashlib.sha256(
                prediction_marker.read_bytes()
            ).hexdigest(),
            "min_mm3": args.min_mm3,
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap,
            "workers": args.workers,
            "labels_read": True,
            "labels_role": "complete training supervision and held-out OOF evaluation only",
            "public_or_hidden_data_read": False,
            "submission_created": False,
        }
        atomic_json(args.output / "report.json", payload)
        (args.output / "COMPLETE.txt").write_text(
            datetime.now().astimezone().isoformat() + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2), flush=True)
        return

    anchor_cache = None
    if args.anchor_metrics_cache is not None:
        if not args.anchor_metrics_cache.is_file():
            raise ValueError(f"missing anchor metrics cache: {args.anchor_metrics_cache}")
        anchor_cache = load_anchor_metrics_cache(
            args.anchor_metrics_cache, items, table_metrics
        )
    official = score_official_oof(
        items,
        args.predictions,
        rows,
        args.min_mm3,
        workers=args.workers,
        anchor_cache=anchor_cache,
    )

    with (args.output / "official_metrics.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(official[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(official)

    means = {
        f"{family}_{metric}": fmean(float(row[f"{family}_{metric}"]) for row in official)
        for family in ("anchor", "candidate")
        for metric in ("DSC", "NSD")
    }
    dsc_parity = {
        "anchor": math.isclose(
            means["anchor_DSC"], table_anchor_dsc, rel_tol=0.0, abs_tol=1e-8
        ),
        "candidate": math.isclose(
            means["candidate_DSC"], table_candidate_dsc, rel_tol=0.0, abs_tol=1e-8
        ),
    }
    if not all(dsc_parity.values()):
        raise RuntimeError(
            "cropped full-volume DSC does not reproduce audited table DSC: "
            f"{dsc_parity}"
        )
    gains = {
        metric: means[f"candidate_{metric}"] - means[f"anchor_{metric}"]
        for metric in ("DSC", "NSD")
    }
    intervals = stratified_bootstrap(official, args.seed, args.bootstrap)
    catastrophic = fmean(
        float(row["candidate_DSC"]) < float(row["anchor_DSC"]) - 0.05
        for row in official
    )
    checks = {
        "DSC_gain_at_least_0p005": gains["DSC"] >= 0.005,
        "NSD_nonregression": gains["NSD"] >= 0.0,
        "DSC_95ci_positive": intervals["DSC"][0] > 0.0,
        "NSD_95ci_above_minus_0p002": intervals["NSD"][0] >= -0.002,
        "catastrophic_regression_rate_at_most_0p02": catastrophic <= 0.02,
    }
    decision = (
        "QUALIFY_GF_BASE_GATE_FOR_COMPOSITION"
        if all(checks.values())
        else "KEEP_UNFILTERED_GF_BASE"
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "passed": all(checks.values()),
        "cases": len(items),
        "components": len(rows),
        "features": FEATURES,
        "target": "pure false-positive component (zero overlap with complete training label)",
        "scope": (
            "GF-base components only; separately guided champion additions are "
            "out of domain and must remain untouched"
        ),
        "next_gate": (
            "if qualified, compose only onto GF-base-traceable components and "
            "require paired public DSC/NSD robustness before ZIP creation"
        ),
        "outer_folds": folds,
        "inner_catastrophic_cap": args.inner_catastrophic_cap,
        "minimum_deletion_mm3_exclusive": args.minimum_deletion_mm3_exclusive,
        "exact_table_DSC_screen": {
            "means": {
                "anchor_DSC": table_anchor_dsc,
                "candidate_DSC": table_candidate_dsc,
            },
            "paired_gain": table_dsc_gain,
            "stratified_bootstrap_95ci": table_dsc_ci,
            "catastrophic_regression_rate": table_catastrophic,
            "checks": screening_checks,
        },
        "official_full_volume_metrics_computed": True,
        "full_volume_table_DSC_parity": dsc_parity,
        "official_means": means,
        "paired_gains": gains,
        "stratified_bootstrap_95ci": intervals,
        "catastrophic_regression_rate": catastrophic,
        "checks": checks,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "predictions": str(args.predictions.resolve()),
        "prediction_marker": str(prediction_marker.resolve()),
        "prediction_marker_sha256": hashlib.sha256(
            prediction_marker.read_bytes()
        ).hexdigest(),
        "min_mm3": args.min_mm3,
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap,
        "workers": args.workers,
        "anchor_metrics_cache": (
            None
            if args.anchor_metrics_cache is None
            else str(args.anchor_metrics_cache.resolve())
        ),
        "anchor_metrics_cache_sha256": (
            None
            if args.anchor_metrics_cache is None
            else hashlib.sha256(args.anchor_metrics_cache.read_bytes()).hexdigest()
        ),
        "labels_read": True,
        "labels_role": "complete training supervision and held-out OOF evaluation only",
        "public_or_hidden_data_read": False,
        "submission_created": False,
    }
    atomic_json(args.output / "report.json", payload)
    (args.output / "COMPLETE.txt").write_text(datetime.now().astimezone().isoformat() + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
