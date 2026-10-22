#!/usr/bin/env python3
"""Nested grouped-CV classifier for disconnected lesion candidates.

The deployed champion remains immutable. Components from a sensitive GF mask
that do not touch the champion are described using inference-visible shape,
location, CT appearance, and independent-model overlap features. Model family,
hyperparameters, and probability threshold are selected only inside each
outer training split. The held-out public predictions are therefore genuinely
out-of-fold at the case level.

This script is public-only. It cannot read hidden images, train a hidden model,
package a ZIP, or upload anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "log_volume_mm3",
    "bbox_x_mm",
    "bbox_y_mm",
    "bbox_z_mm",
    "bbox_fill_fraction",
    "centroid_x_fraction",
    "centroid_y_fraction",
    "centroid_z_fraction",
    "border_fraction",
    "hu_mean",
    "hu_std",
    "hu_min",
    "hu_max",
    "candidate_case_burden_fraction",
    "candidate_component_count_log",
    "seed40_fraction",
    "seed55_fraction",
    "richfull_fraction",
    "whole_prompt_fraction",
    "tumor_prompt_fraction",
    "nnunet_fraction",
)

# Coarse, deliberately low-dimensional anatomical families from the public
# TotalSegmentator ``total`` task (v2 class ordering).  Grouping the 117 labels
# avoids fitting a sparse organ identity lookup to only 50 public cases while
# still exposing the main false-positive modes: vessels, bone, bowel, and
# muscle.  These features are optional so the historical analysis remains
# exactly reproducible when --anatomy-dir is omitted.
ANATOMY_FAMILIES = {
    "solid_organ": (1, 2, 3, 4, 5, 7, 8, 9, 17, 23, 24),
    "lung": (10, 11, 12, 13, 14),
    "gastrointestinal": (6, 15, 18, 19, 20),
    "urinary_reproductive": (21, 22),
    "heart": (51, 61),
    "vessel": tuple(range(52, 69)),
    "bone": tuple(range(25, 51)) + tuple(range(69, 79)) + tuple(range(91, 118)),
    "muscle": tuple(range(80, 90)),
    "brain_spinal": (79, 90),
}
ANATOMY_FEATURES = (
    "anatomy_nonzero_fraction",
    "anatomy_background_fraction",
    "anatomy_near5_fraction",
    "anatomy_near15_fraction",
    "anatomy_distance_min_log1p_mm",
    "anatomy_distance_mean_log1p_mm",
    "anatomy_dominant_family_fraction",
) + tuple(f"anatomy_{name}_fraction" for name in ANATOMY_FAMILIES)


@dataclass
class CaseStats:
    base_voxels: int
    target_voxels: int
    base_overlap: int
    component_voxels: dict[int, int]
    component_overlap: dict[int, int]


def parse_named_directory(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("support must be NAME=DIRECTORY")
    name, directory = value.split("=", 1)
    if not name or not directory:
        raise argparse.ArgumentTypeError("support must be NAME=DIRECTORY")
    return name, Path(directory).resolve()


def load_image(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    array = np.asanyarray(image.dataobj)
    return array, image


def validate_geometry(reference: nib.Nifti1Image, other: nib.Nifti1Image, name: str) -> None:
    if reference.shape != other.shape or not np.allclose(
        reference.affine, other.affine, atol=1e-3
    ):
        raise ValueError(f"geometry mismatch: {name}")


def overlap_fraction(labels: np.ndarray, mask: np.ndarray, counts: np.ndarray) -> np.ndarray:
    overlap = np.bincount(labels[mask], minlength=len(counts))
    return np.divide(
        overlap,
        counts,
        out=np.zeros(len(counts), dtype=np.float64),
        where=counts > 0,
    )


def extract_anatomy_features(
    component_map: np.ndarray,
    counts: np.ndarray,
    anatomy: np.ndarray,
    spacing: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return inference-visible coarse anatomy features for every component."""
    if component_map.shape != anatomy.shape:
        raise ValueError("component and anatomy arrays must have identical shapes")
    anatomy = anatomy.astype(np.int16, copy=False)
    nonzero = anatomy > 0
    nonzero_fraction = overlap_fraction(component_map, nonzero, counts)

    # A single physical-distance transform is considerably cheaper than one
    # transform per family and answers the useful question: is this component
    # inside or close to any recognized structure?
    if np.any(nonzero):
        distance = ndimage.distance_transform_edt(~nonzero, sampling=spacing)
        component_ids = np.arange(len(counts), dtype=np.int64)
        distance_min = np.asarray(
            ndimage.minimum(distance, labels=component_map, index=component_ids),
            dtype=np.float64,
        )
        distance_mean = np.asarray(
            ndimage.mean(distance, labels=component_map, index=component_ids),
            dtype=np.float64,
        )
    else:
        fallback = float(np.linalg.norm(np.asarray(anatomy.shape) * spacing))
        distance = np.full(anatomy.shape, fallback, dtype=np.float32)
        distance_min = np.full(len(counts), fallback, dtype=np.float64)
        distance_mean = np.full(len(counts), fallback, dtype=np.float64)

    result: dict[str, np.ndarray] = {
        "anatomy_nonzero_fraction": nonzero_fraction,
        "anatomy_background_fraction": 1.0 - nonzero_fraction,
        "anatomy_near5_fraction": overlap_fraction(component_map, distance <= 5.0, counts),
        "anatomy_near15_fraction": overlap_fraction(component_map, distance <= 15.0, counts),
        "anatomy_distance_min_log1p_mm": np.log1p(distance_min),
        "anatomy_distance_mean_log1p_mm": np.log1p(distance_mean),
    }
    family_fractions = []
    for family, values in ANATOMY_FAMILIES.items():
        fraction = overlap_fraction(component_map, np.isin(anatomy, values), counts)
        result[f"anatomy_{family}_fraction"] = fraction
        family_fractions.append(fraction)
    result["anatomy_dominant_family_fraction"] = np.max(
        np.stack(family_fractions, axis=0), axis=0
    )
    return result


def extract_case(
    name: str,
    images: Path,
    labels_dir: Path,
    base_dir: Path,
    candidate_dir: Path,
    supports: dict[str, Path],
    anatomy_dir: Path | None = None,
) -> tuple[list[dict[str, object]], CaseStats]:
    candidate, reference = load_image(candidate_dir / name)
    candidate = candidate.astype(bool)
    base, base_image = load_image(base_dir / name)
    target, target_image = load_image(labels_dir / name)
    base = base.astype(bool)
    target = target.astype(bool)
    validate_geometry(reference, base_image, f"base/{name}")
    validate_geometry(reference, target_image, f"target/{name}")

    case_id = name.removesuffix(".nii.gz")
    image_path = images / f"{case_id}_0000.nii.gz"
    ct, ct_image = load_image(image_path)
    validate_geometry(reference, ct_image, f"image/{name}")
    ct = np.nan_to_num(ct.astype(np.float32), nan=-1024.0, posinf=3071.0, neginf=-1024.0)

    support_masks: dict[str, np.ndarray] = {}
    for support_name, directory in supports.items():
        mask, support_image = load_image(directory / name)
        validate_geometry(reference, support_image, f"{support_name}/{name}")
        support_masks[support_name] = mask.astype(bool)

    component_map = cc3d.connected_components(candidate.astype(np.uint8), connectivity=26)
    statistics = cc3d.statistics(component_map)
    counts = np.asarray(statistics["voxel_counts"], dtype=np.int64)
    component_count = max(0, len(counts) - 1)
    base_overlap = np.bincount(component_map[base], minlength=len(counts))
    target_overlap = np.bincount(component_map[target], minlength=len(counts))
    support_fraction = {
        support_name: overlap_fraction(component_map, mask, counts)
        for support_name, mask in support_masks.items()
    }
    component_ids = np.arange(1, len(counts), dtype=np.int64)
    hu_mean = ndimage.mean(ct, labels=component_map, index=component_ids)
    hu_std = ndimage.standard_deviation(ct, labels=component_map, index=component_ids)
    hu_min = ndimage.minimum(ct, labels=component_map, index=component_ids)
    hu_max = ndimage.maximum(ct, labels=component_map, index=component_ids)
    spacing = np.asarray(reference.header.get_zooms()[:3], dtype=np.float64)
    anatomy_feature_values: dict[str, np.ndarray] = {}
    if anatomy_dir is not None:
        anatomy, anatomy_image = load_image(anatomy_dir / name)
        validate_geometry(reference, anatomy_image, f"anatomy/{name}")
        anatomy_feature_values = extract_anatomy_features(
            component_map, counts, anatomy, spacing
        )
    voxel_mm3 = float(np.prod(spacing))
    shape = np.asarray(candidate.shape, dtype=np.float64)
    total_candidate = max(1, int(np.count_nonzero(candidate)))

    rows: list[dict[str, object]] = []
    component_voxels: dict[int, int] = {}
    component_target: dict[int, int] = {}
    for component_id in component_ids:
        # The deployed champion is immutable; overlapping components cannot be
        # additions and are excluded from both learning and materialization.
        if base_overlap[component_id] > 0:
            continue
        box = statistics["bounding_boxes"][component_id]
        extents = np.asarray([axis.stop - axis.start for axis in box], dtype=np.float64)
        bbox_voxels = max(1.0, float(np.prod(extents)))
        centroid = np.asarray(statistics["centroids"][component_id], dtype=np.float64)
        centroid_fraction = centroid / np.maximum(shape - 1.0, 1.0)
        border_fraction = float(
            np.min(np.concatenate((centroid_fraction, 1.0 - centroid_fraction)))
        )
        size = int(counts[component_id])
        overlap = int(target_overlap[component_id])
        row: dict[str, object] = {
            "case": case_id,
            "component_id": int(component_id),
            "component_voxels": size,
            "target_overlap_voxels": overlap,
            "is_true_component": int(overlap > 0),
            "target_precision": overlap / size,
            "log_volume_mm3": math.log1p(size * voxel_mm3),
            "bbox_x_mm": extents[0] * spacing[0],
            "bbox_y_mm": extents[1] * spacing[1],
            "bbox_z_mm": extents[2] * spacing[2],
            "bbox_fill_fraction": size / bbox_voxels,
            "centroid_x_fraction": centroid_fraction[0],
            "centroid_y_fraction": centroid_fraction[1],
            "centroid_z_fraction": centroid_fraction[2],
            "border_fraction": border_fraction,
            "hu_mean": float(hu_mean[component_id - 1]),
            "hu_std": float(hu_std[component_id - 1]),
            "hu_min": float(hu_min[component_id - 1]),
            "hu_max": float(hu_max[component_id - 1]),
            "candidate_case_burden_fraction": size / total_candidate,
            "candidate_component_count_log": math.log1p(component_count),
        }
        for support_name in supports:
            row[f"{support_name}_fraction"] = float(
                support_fraction[support_name][component_id]
            )
        if anatomy_dir is not None:
            for feature_name in ANATOMY_FEATURES:
                row[feature_name] = float(
                    anatomy_feature_values[feature_name][component_id]
                )
        rows.append(row)
        component_voxels[int(component_id)] = size
        component_target[int(component_id)] = overlap

    return rows, CaseStats(
        base_voxels=int(np.count_nonzero(base)),
        target_voxels=int(np.count_nonzero(target)),
        base_overlap=int(np.count_nonzero(base & target)),
        component_voxels=component_voxels,
        component_overlap=component_target,
    )


def model_grid(seed: int, deep_features: bool = False):
    if deep_features:
        # A 512-D frozen representation with only 50 independent cases needs
        # strong linear regularization. Tree models are both statistically
        # inappropriate here and prohibitively expensive inside nested CV.
        for c in (0.001, 0.005, 0.02, 0.10):
            yield (
                f"deep_logistic_c{c:g}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            )
        return
    for c in (0.05, 0.20, 1.0):
        yield (
            f"logistic_c{c:g}",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c, class_weight="balanced", max_iter=3000, random_state=seed
                ),
            ),
        )
    for depth, leaf in ((4, 10), (8, 5)):
        yield (
            f"forest_d{depth}_l{leaf}",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=depth,
                min_samples_leaf=leaf,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=2,
                random_state=seed,
            ),
        )


def probability(model, x_train, y_train, x_test) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_test), float(y_train[0]), dtype=np.float64)
    fitted = clone(model).fit(x_train, y_train)
    return fitted.predict_proba(x_test)[:, 1]


def macro_dice(
    rows: list[dict[str, object]],
    probabilities: np.ndarray,
    threshold: float,
    cases: set[str],
    stats: dict[str, CaseStats],
) -> float:
    chosen: dict[str, list[int]] = {case: [] for case in cases}
    for row, value in zip(rows, probabilities):
        case = str(row["case"])
        if case in chosen and value >= threshold:
            chosen[case].append(int(row["component_id"]))
    scores = []
    for case in sorted(cases):
        case_stats = stats[case]
        component_ids = chosen[case]
        pred = case_stats.base_voxels + sum(
            case_stats.component_voxels[value] for value in component_ids
        )
        overlap = case_stats.base_overlap + sum(
            case_stats.component_overlap[value] for value in component_ids
        )
        denominator = pred + case_stats.target_voxels
        scores.append(1.0 if denominator == 0 else 2.0 * overlap / denominator)
    return fmean(scores)


def nested_group_predictions(
    rows: list[dict[str, object]],
    stats: dict[str, CaseStats],
    seed: int,
    feature_names: tuple[str, ...] = FEATURES,
    extra_features: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    x = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64
    )
    if extra_features is not None:
        extra_features = np.asarray(extra_features, dtype=np.float64)
        if extra_features.ndim != 2 or extra_features.shape[0] != len(rows):
            raise ValueError(
                "extra feature rows must match component rows: "
                f"{extra_features.shape} versus {len(rows)}"
            )
        if not np.isfinite(extra_features).all():
            raise ValueError("extra features contain non-finite values")
        x = np.concatenate((x, extra_features), axis=1)
    y = np.asarray([int(row["is_true_component"]) for row in rows], dtype=np.int64)
    groups = np.asarray([str(row["case"]) for row in rows])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 10:
        raise ValueError("at least ten component-containing cases are required")
    outer = GroupKFold(n_splits=5)
    out_of_fold = np.full(len(rows), np.nan, dtype=np.float64)
    fold_records: list[dict[str, object]] = []
    thresholds = np.linspace(0.05, 0.95, 19)

    for fold, (train_index, test_index) in enumerate(outer.split(x, y, groups), 1):
        train_groups = groups[train_index]
        test_cases = set(groups[test_index])
        inner_splits = min(4, len(np.unique(train_groups)))
        inner = GroupKFold(n_splits=inner_splits)
        best = None
        for model_name, model in model_grid(
            seed + fold, deep_features=extra_features is not None
        ):
            inner_probability = np.full(len(train_index), np.nan, dtype=np.float64)
            x_train_all = x[train_index]
            y_train_all = y[train_index]
            for inner_train, inner_val in inner.split(
                x_train_all, y_train_all, train_groups
            ):
                inner_probability[inner_val] = probability(
                    model,
                    x_train_all[inner_train],
                    y_train_all[inner_train],
                    x_train_all[inner_val],
                )
            if not np.isfinite(inner_probability).all():
                raise RuntimeError("inner grouped predictions are incomplete")
            train_rows = [rows[index] for index in train_index]
            train_cases = set(train_groups)
            for threshold in thresholds:
                value = macro_dice(
                    train_rows, inner_probability, float(threshold), train_cases, stats
                )
                candidate = (value, float(threshold), model_name, model)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        assert best is not None
        inner_dice, threshold, model_name, model = best
        test_probability = probability(model, x[train_index], y[train_index], x[test_index])
        out_of_fold[test_index] = test_probability
        test_rows = [rows[index] for index in test_index]
        test_dice = macro_dice(test_rows, test_probability, threshold, test_cases, stats)
        for index in test_index:
            rows[index]["outer_fold"] = fold
            rows[index]["oof_probability"] = float(out_of_fold[index])
            rows[index]["oof_threshold"] = threshold
            rows[index]["oof_selected"] = int(out_of_fold[index] >= threshold)
        fold_records.append(
            {
                "fold": fold,
                "train_cases": len(set(train_groups)),
                "test_cases": len(test_cases),
                "model": model_name,
                "threshold": threshold,
                "inner_oof_DSC": inner_dice,
                "outer_DSC": test_dice,
            }
        )
    if not np.isfinite(out_of_fold).all():
        raise RuntimeError("outer grouped predictions are incomplete")
    return out_of_fold, fold_records


def load_component_embeddings(
    path: Path, rows: list[dict[str, object]]
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a label-free embedding archive and enforce exact row identity."""
    with np.load(path, allow_pickle=False) as payload:
        required = {"cases", "component_ids", "embeddings", "logits"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"embedding archive is missing {sorted(missing)}")
        cases = payload["cases"].astype(str)
        component_ids = payload["component_ids"].astype(np.int64)
        embeddings = payload["embeddings"].astype(np.float32)
        logits = payload["logits"].astype(np.float32)
    expected = [(str(row["case"]), int(row["component_id"])) for row in rows]
    observed = list(zip(cases.tolist(), component_ids.tolist()))
    if observed != expected:
        mismatch = next(
            (
                index,
                expected[index] if index < len(expected) else None,
                observed[index] if index < len(observed) else None,
            )
            for index in range(max(len(expected), len(observed)))
            if index >= len(expected)
            or index >= len(observed)
            or expected[index] != observed[index]
        )
        raise ValueError(f"embedding/component row identity mismatch: {mismatch}")
    if embeddings.ndim != 2 or logits.shape != (len(rows), 2):
        raise ValueError(
            f"invalid embedding/logit shapes: {embeddings.shape}, {logits.shape}"
        )
    # Logit difference is a calibrated-free representation of the source
    # Glance decision and avoids the numerical redundancy of two logits.
    extra = np.concatenate(
        (embeddings, (logits[:, 1] - logits[:, 0])[:, None]), axis=1
    )
    return extra, {
        "path": str(path.resolve()),
        "embedding_dimensions": int(embeddings.shape[1]),
        "extra_feature_dimensions": int(extra.shape[1]),
    }


def load_frozen_component_table(
    path: Path,
    base: Path,
    labels_dir: Path,
    expected_cases: set[str],
    feature_names: tuple[str, ...],
) -> tuple[list[dict[str, object]], dict[str, CaseStats]]:
    """Load previously extracted component rows and rebuild exact case totals.

    This avoids rereading CT and six support volumes when testing a new
    label-free feature representation against an immutable component table.
    Case-level base/target totals are still recomputed from masks, so macro DSC
    is not trusted from stale summary files.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        rows: list[dict[str, object]] = list(csv.DictReader(stream))
    required = {
        "case",
        "component_id",
        "component_voxels",
        "target_overlap_voxels",
        "is_true_component",
        *feature_names,
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"frozen component table is missing {sorted(required-set(rows[0]) if rows else required)}"
        )
    keys = [(str(row["case"]), int(row["component_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("frozen component table has duplicate case/component keys")
    observed_cases = {str(row["case"]) for row in rows}
    if observed_cases != expected_cases:
        raise ValueError(
            "frozen component table case mismatch: "
            f"missing={sorted(expected_cases-observed_cases)[:3]} "
            f"extra={sorted(observed_cases-expected_cases)[:3]}"
        )

    grouped: dict[str, list[dict[str, object]]] = {
        case: [] for case in expected_cases
    }
    for row in rows:
        grouped[str(row["case"])].append(row)
    stats: dict[str, CaseStats] = {}
    for case in sorted(expected_cases):
        base_array, reference = load_image(base / f"{case}.nii.gz")
        target, target_image = load_image(labels_dir / f"{case}.nii.gz")
        validate_geometry(reference, target_image, f"target/{case}.nii.gz")
        base_mask = base_array.astype(bool)
        target_mask = target.astype(bool)
        stats[case] = CaseStats(
            base_voxels=int(np.count_nonzero(base_mask)),
            target_voxels=int(np.count_nonzero(target_mask)),
            base_overlap=int(np.count_nonzero(base_mask & target_mask)),
            component_voxels={
                int(row["component_id"]): int(row["component_voxels"])
                for row in grouped[case]
            },
            component_overlap={
                int(row["component_id"]): int(row["target_overlap_voxels"])
                for row in grouped[case]
            },
        )
    return rows, stats


def materialize_oof(
    rows: list[dict[str, object]], base: Path, candidate: Path, output: Path
) -> None:
    selected: dict[str, set[int]] = {}
    for row in rows:
        if int(row["oof_selected"]):
            selected.setdefault(str(row["case"]), set()).add(int(row["component_id"]))
    output.mkdir(parents=True, exist_ok=True)
    for base_path in sorted(base.glob("*.nii.gz")):
        name = base_path.name
        case = name.removesuffix(".nii.gz")
        base_array, reference = load_image(base_path)
        candidate_array, candidate_image = load_image(candidate / name)
        validate_geometry(reference, candidate_image, f"candidate/{name}")
        component_map = cc3d.connected_components(
            candidate_array.astype(bool).astype(np.uint8), connectivity=26
        )
        ids = selected.get(case, set())
        if ids:
            keep = np.zeros(int(component_map.max()) + 1, dtype=bool)
            keep[list(ids)] = True
            prediction = base_array.astype(bool) | keep[component_map]
        else:
            prediction = base_array.astype(bool)
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(
            nib.Nifti1Image(prediction.astype(np.uint8), reference.affine, header),
            output / name,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--support", type=parse_named_directory, action="append", required=True)
    parser.add_argument(
        "--anatomy-dir",
        type=Path,
        help="optional TotalSegmentator multilabel masks with base-mask filenames",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--component-embeddings",
        type=Path,
        help=(
            "optional label-free NPZ from extract_glance_component_embeddings.py; "
            "activates a strongly regularized deep-feature model grid"
        ),
    )
    parser.add_argument(
        "--component-table",
        type=Path,
        help=(
            "optional frozen CSV from a prior identical component extraction; "
            "recomputes case totals from masks but skips redundant CT/support IO"
        ),
    )
    args = parser.parse_args()

    supports = dict(args.support)
    expected_supports = {
        "seed40", "seed55", "richfull", "whole_prompt", "tumor_prompt", "nnunet"
    }
    if set(supports) != expected_supports:
        raise ValueError(f"support names must be exactly {sorted(expected_supports)}")
    names = sorted(path.name for path in args.base.glob("*.nii.gz"))
    if len(names) != 50:
        raise ValueError(f"expected 50 base masks, found {len(names)}")
    expected = set(names)
    for label, directory in {
        "labels": args.labels,
        "candidate": args.candidate,
        **supports,
        **({"anatomy": args.anatomy_dir} if args.anatomy_dir is not None else {}),
    }.items():
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected:
            raise ValueError(
                f"{label} case mismatch: missing={sorted(expected-observed)[:3]} "
                f"extra={sorted(observed-expected)[:3]}"
            )
    image_names = {path.name for path in args.images.glob("*_0000.nii.gz")}
    expected_images = {
        f"{name.removesuffix('.nii.gz')}_0000.nii.gz" for name in names
    }
    if image_names != expected_images:
        raise ValueError("public image set does not match the 50 mask cases")

    feature_names = FEATURES + (ANATOMY_FEATURES if args.anatomy_dir is not None else ())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.component_table is not None:
        rows, stats = load_frozen_component_table(
            args.component_table,
            args.base,
            args.labels,
            {name.removesuffix(".nii.gz") for name in names},
            feature_names,
        )
        print(
            f"loaded frozen component table: {len(rows)} rows, {len(stats)} cases",
            flush=True,
        )
    else:
        rows = []
        stats = {}
        for index, name in enumerate(names, 1):
            case_rows, case_stats = extract_case(
                name,
                args.images,
                args.labels,
                args.base,
                args.candidate,
                supports,
                args.anatomy_dir,
            )
            case = name.removesuffix(".nii.gz")
            rows.extend(case_rows)
            stats[case] = case_stats
            print(
                f"[{index:02d}/50] {case} disconnected_components={len(case_rows)}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("no disconnected candidate components were found")
    extra_features = None
    embedding_metadata = None
    if args.component_embeddings is not None:
        extra_features, embedding_metadata = load_component_embeddings(
            args.component_embeddings, rows
        )
    probabilities, fold_records = nested_group_predictions(
        rows, stats, args.seed, feature_names, extra_features=extra_features
    )
    all_cases = set(stats)
    # Each row carries its fold-specific threshold; compute the exact grouped
    # OOF prediction rather than applying one global post-hoc cutoff.
    selected_by_case: dict[str, list[int]] = {case: [] for case in all_cases}
    true_selected = 0
    selected_count = 0
    total_true = sum(int(row["is_true_component"]) for row in rows)
    for row, value in zip(rows, probabilities):
        if value >= float(row["oof_threshold"]):
            selected_by_case[str(row["case"])].append(int(row["component_id"]))
            selected_count += 1
            true_selected += int(row["is_true_component"])
    case_scores = {}
    anchor_scores = {}
    for case in sorted(all_cases):
        case_stats = stats[case]
        ids = selected_by_case[case]
        pred = case_stats.base_voxels + sum(case_stats.component_voxels[value] for value in ids)
        overlap = case_stats.base_overlap + sum(
            case_stats.component_overlap[value] for value in ids
        )
        denominator = pred + case_stats.target_voxels
        case_scores[case] = 1.0 if denominator == 0 else 2.0 * overlap / denominator
        anchor_denominator = case_stats.base_voxels + case_stats.target_voxels
        anchor_scores[case] = (
            1.0
            if anchor_denominator == 0
            else 2.0 * case_stats.base_overlap / anchor_denominator
        )
    report = {
        "warning": (
            "Public-only grouped cross-validation diagnostic. No case trains "
            "its own component decision; no hidden model or submission exists."
        ),
        "cases": len(stats),
        "components": len(rows),
        "true_components": total_true,
        "selected_components": selected_count,
        "true_selected_components": true_selected,
        "selected_component_precision": (
            true_selected / selected_count if selected_count else None
        ),
        "selected_component_recall": true_selected / total_true if total_true else None,
        "anchor_macro_DSC": fmean(anchor_scores.values()),
        "oof_macro_DSC": fmean(case_scores.values()),
        "oof_macro_DSC_gain": fmean(case_scores.values()) - fmean(anchor_scores.values()),
        "win_rate": fmean(case_scores[case] > anchor_scores[case] for case in all_cases),
        "catastrophic_regression_rate": fmean(
            case_scores[case] < anchor_scores[case] - 0.05 for case in all_cases
        ),
        "features": feature_names,
        "component_embeddings": embedding_metadata,
        "folds": fold_records,
    }
    with (args.output_dir / "components_oof.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "case_dice.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("case", "anchor_DSC", "oof_DSC", "gain"))
        for case in sorted(all_cases):
            writer.writerow(
                (case, anchor_scores[case], case_scores[case], case_scores[case] - anchor_scores[case])
            )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    materialize_oof(rows, args.base, args.candidate, args.output_dir / "predictions_oof")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
