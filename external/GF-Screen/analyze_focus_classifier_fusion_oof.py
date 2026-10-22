#!/usr/bin/env python3
"""Cross-fit B32 and Focus confidence for exact707 candidate classification."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from candidate_refiner_sampling import semantic_source_group
from focus_classifier_fusion import (
    DEPLOYMENT_FEATURE_NAMES,
    FOCUS_CONFIDENCE_FEATURE_NAMES,
)
from summarize_candidate_refiner_oof import case_balanced_threshold_metrics
from train_candidate_refiner import grouped_fold_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_weights(rows: list[dict[str, str]], indices: np.ndarray) -> np.ndarray:
    counts: dict[str, int] = {}
    for index in indices:
        case = str(rows[int(index)]["case"])
        counts[case] = counts.get(case, 0) + 1
    values = np.asarray(
        [1.0 / counts[str(rows[int(index)]["case"])] for index in indices],
        dtype=np.float64,
    )
    return values * (len(values) / max(values.sum(), 1e-12))


def fit_model(
    features: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str]],
    indices: np.ndarray,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(features[indices])
    classifier = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
    classifier.fit(
        scaler.transform(features[indices]),
        labels[indices],
        sample_weight=case_weights(rows, indices),
    )
    return scaler, classifier


def metrics(
    rows: list[dict[str, str]], labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, object]:
    expected = {str(row["patch_id"]): row for row in rows}
    patch_ids = [str(row["patch_id"]) for row in rows]
    indices = np.arange(len(rows), dtype=np.int64)
    weights = case_weights(rows, indices)
    threshold = case_balanced_threshold_metrics(
        expected, patch_ids, labels, probabilities
    )
    return {
        **threshold,
        "average_precision": float(
            average_precision_score(labels, probabilities, sample_weight=weights)
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities, sample_weight=weights)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--b32-prediction", type=Path, action="append", required=True)
    parser.add_argument("--focus-prediction", type=Path, action="append", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--focus-input-mode",
        choices=("soft", "soft_seed", "soft_lung_seed"),
        default="soft_lung_seed",
    )
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.folds < 2
        or len(args.b32_prediction) != args.folds
        or len(args.focus_prediction) != args.folds
        or args.bootstrap < 0
    ):
        parser.error("one B32 and Focus archive are required per fold")

    with args.index.open(newline="", encoding="utf-8") as stream:
        index_rows = list(csv.DictReader(stream))
    by_id = {str(row["patch_id"]): row for row in index_rows}
    if len(by_id) != len(index_rows):
        raise ValueError("duplicate patch IDs in exact707 index")
    ordered_ids: list[str] = []
    ordered_labels: list[int] = []
    ordered_b32: list[float] = []
    ordered_focus: list[np.ndarray] = []
    ordered_folds: list[int] = []
    provenance = []
    expected_focus_names = list(FOCUS_CONFIDENCE_FEATURE_NAMES)
    for fold, (b32_path, focus_path) in enumerate(
        zip(args.b32_prediction, args.focus_prediction)
    ):
        _, held_out = grouped_fold_rows(index_rows, args.folds, fold, args.seed)
        expected_ids = {str(row["patch_id"]) for row in held_out}
        with np.load(b32_path, allow_pickle=False) as archive:
            b32_ids = [str(value) for value in archive["patch_id"]]
            b32_labels = archive["label"].astype(np.uint8)
            b32_probabilities = archive["classification_probability"].astype(
                np.float64
            )
        with np.load(focus_path, allow_pickle=False) as archive:
            focus_ids = [str(value) for value in archive["all_patch_id"]]
            focus_labels = archive["all_label"].astype(np.uint8)
            focus_names = [str(value) for value in archive["confidence_feature_names"]]
            focus_features = archive["confidence_features"].astype(np.float64)
        if (
            len(b32_ids) != len(set(b32_ids))
            or set(b32_ids) != expected_ids
            or len(focus_ids) != len(set(focus_ids))
            or set(focus_ids) != expected_ids
        ):
            raise ValueError(f"held-out patch coverage mismatch in fold {fold}")
        if focus_names != expected_focus_names or focus_features.shape != (
            len(focus_ids),
            len(expected_focus_names),
        ):
            raise ValueError(f"Focus confidence feature mismatch in fold {fold}")
        b32_position = {patch_id: index for index, patch_id in enumerate(b32_ids)}
        focus_position = {
            patch_id: index for index, patch_id in enumerate(focus_ids)
        }
        for patch_id in sorted(expected_ids):
            b32_index = b32_position[patch_id]
            focus_index = focus_position[patch_id]
            expected_label = int(by_id[patch_id]["label"])
            if (
                int(b32_labels[b32_index]) != expected_label
                or int(focus_labels[focus_index]) != expected_label
            ):
                raise ValueError(f"label mismatch: {patch_id}")
            ordered_ids.append(patch_id)
            ordered_labels.append(expected_label)
            ordered_b32.append(float(b32_probabilities[b32_index]))
            ordered_focus.append(focus_features[focus_index])
            ordered_folds.append(fold)
        provenance.append(
            {
                "fold": fold,
                "b32_prediction": str(b32_path.resolve()),
                "b32_prediction_sha256": sha256(b32_path),
                "focus_prediction": str(focus_path.resolve()),
                "focus_prediction_sha256": sha256(focus_path),
                "patches": len(expected_ids),
            }
        )
    if len(ordered_ids) != len(index_rows) or set(ordered_ids) != set(by_id):
        raise ValueError("cross-fitted fusion lacks exact index coverage")

    rows = [by_id[patch_id] for patch_id in ordered_ids]
    labels = np.asarray(ordered_labels, dtype=np.uint8)
    b32 = np.asarray(ordered_b32, dtype=np.float64)
    focus = np.stack(ordered_focus).astype(np.float64)
    fold_ids = np.asarray(ordered_folds, dtype=np.int64)
    b32_logit = np.log(np.clip(b32, 1e-6, 1 - 1e-6) / np.clip(1 - b32, 1e-6, 1))
    geometry = np.asarray(
        [
            [
                np.log1p(float(row.get("seed_mm3", 0.0) or 0.0)),
                float(row.get("max_source_support", 0.0) or 0.0),
                float(row.get("mean_source_support", 0.0) or 0.0),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    def source_group(row: dict[str, str]) -> str:
        source = semantic_source_group(str(row["source_name"]))
        if source in {"epoch105", "currentgf"}:
            return "mature"
        if source in {"autopet_weak", "lndb_weak"}:
            return "weak_sources"
        return source

    source_groups = [source_group(row) for row in rows]
    group_names = ["mature", "weak_sources", "lung_cells"]
    if set(source_groups) - set(group_names):
        raise ValueError(f"unexpected source groups: {sorted(set(source_groups))}")
    one_hot = np.asarray(
        [[float(group == name) for name in group_names] for group in source_groups],
        dtype=np.float64,
    )
    features = np.column_stack((b32_logit, focus, geometry, one_hot))
    feature_names = (
        ["b32_logit"]
        + [f"focus_{name}" for name in expected_focus_names]
        + ["log1p_seed_mm3", "max_source_support", "mean_source_support"]
        + [f"source_{name}" for name in group_names]
    )
    if tuple(feature_names) != DEPLOYMENT_FEATURE_NAMES:
        raise RuntimeError("classifier-fusion deployment feature identity drift")
    if not np.isfinite(features).all():
        raise ValueError("non-finite stacked feature")

    fused = np.empty(len(rows), dtype=np.float64)
    fold_models = []
    for fold in range(args.folds):
        train = np.flatnonzero(fold_ids != fold)
        held_out = np.flatnonzero(fold_ids == fold)
        scaler, classifier = fit_model(features, labels, rows, train)
        fused[held_out] = classifier.predict_proba(
            scaler.transform(features[held_out])
        )[:, 1]
        fold_models.append(
            {
                "fold": fold,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "coefficient": classifier.coef_[0].tolist(),
                "intercept": float(classifier.intercept_[0]),
            }
        )
    full_scaler, full_classifier = fit_model(
        features, labels, rows, np.arange(len(rows), dtype=np.int64)
    )
    baseline_metrics = metrics(rows, labels, b32)
    candidate_metrics = metrics(rows, labels, fused)
    by_source = {}
    source_checks = []
    for group in group_names:
        selected = np.flatnonzero(np.asarray(source_groups) == group)
        source_rows = [rows[int(index)] for index in selected]
        baseline = metrics(source_rows, labels[selected], b32[selected])
        candidate = metrics(source_rows, labels[selected], fused[selected])
        gain = float(candidate["average_precision"]) - float(
            baseline["average_precision"]
        )
        by_source[group] = {
            "patches": len(selected),
            "baseline": baseline,
            "candidate": candidate,
            "average_precision_gain": gain,
        }
        source_checks.append(gain >= -0.01)

    rng = np.random.default_rng(args.seed)
    cases = sorted({str(row["case"]) for row in rows})
    case_to_indices = {
        case: np.asarray(
            [index for index, row in enumerate(rows) if str(row["case"]) == case],
            dtype=np.int64,
        )
        for case in cases
    }
    bootstrap_gains = []
    for _ in range(args.bootstrap):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        sampled_indices = np.concatenate([case_to_indices[case] for case in sampled])
        sampled_weights = np.concatenate(
            [
                np.full(len(case_to_indices[case]), 1.0 / len(case_to_indices[case]))
                for case in sampled
            ]
        )
        sampled_labels = labels[sampled_indices]
        if np.unique(sampled_labels).size < 2:
            continue
        bootstrap_gains.append(
            float(
                average_precision_score(
                    sampled_labels,
                    fused[sampled_indices],
                    sample_weight=sampled_weights,
                )
                - average_precision_score(
                    sampled_labels,
                    b32[sampled_indices],
                    sample_weight=sampled_weights,
                )
            )
        )
    ap_gain = float(candidate_metrics["average_precision"]) - float(
        baseline_metrics["average_precision"]
    )
    f_gain = float(candidate_metrics["f0p5"]) - float(baseline_metrics["f0p5"])
    checks = {
        "average_precision_gain": ap_gain >= 0.01,
        "f0p5_gain": f_gain >= 0.005,
        "source_nonregression": all(source_checks),
    }
    decision = "QUALIFY_FOCUS_CLASSIFIER_FUSION" if all(checks.values()) else "KEEP_B32_CLASSIFIER"
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = args.predictions_output.with_name(
        f".{args.predictions_output.name}.{os.getpid()}.tmp"
    )
    with temporary_npz.open("wb") as stream:
        np.savez_compressed(
            stream,
            patch_id=np.asarray(ordered_ids),
            label=labels,
            classification_probability=fused.astype(np.float32),
            fold=fold_ids,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_npz, args.predictions_output)
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "selection_basis": "cross-fitted exact707 grouped held-out patch classification only",
        "index": str(args.index.resolve()),
        "index_sha256": sha256(args.index),
        "feature_names": feature_names,
        "focus_input_mode": args.focus_input_mode,
        "model": {"type": "standardized logistic regression", "C": 0.1},
        "fold_models": fold_models,
        "deployment_model": {
            "scaler_mean": full_scaler.mean_.tolist(),
            "scaler_scale": full_scaler.scale_.tolist(),
            "coefficient": full_classifier.coef_[0].tolist(),
            "intercept": float(full_classifier.intercept_[0]),
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "average_precision_gain": ap_gain,
        "f0p5_gain": f_gain,
        "average_precision_gain_95ci": (
            [float(value) for value in np.quantile(bootstrap_gains, (0.025, 0.975))]
            if bootstrap_gains
            else None
        ),
        "by_source": by_source,
        "checks": checks,
        "decision": decision,
        "predictions": str(args.predictions_output.resolve()),
        "predictions_sha256": sha256(args.predictions_output),
        "provenance": provenance,
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
