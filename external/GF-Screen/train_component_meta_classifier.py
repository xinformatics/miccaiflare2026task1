#!/usr/bin/env python3
"""Cross-fit a nonlinear component acceptor on exact-707 OOF predictions."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from candidate_refiner_sampling import semantic_source_group
from component_meta_classifier import FEATURE_NAMES, SOURCE_NAMES, component_features, sha256


MODEL_PARAMETERS = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 160,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def case_weights(cases: np.ndarray) -> np.ndarray:
    counts = Counter(str(value) for value in cases)
    values = np.asarray([1.0 / counts[str(case)] for case in cases], dtype=np.float64)
    return values * (len(values) / values.sum())


def best_f0p5(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(
        labels, probabilities, sample_weight=weights
    )
    scores = 1.25 * precision[:-1] * recall[:-1] / np.maximum(
        0.25 * precision[:-1] + recall[:-1], 1e-12
    )
    index = int(np.argmax(scores))
    return {
        "threshold": float(thresholds[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f0p5": float(scores[index]),
    }


def metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    return {
        **best_f0p5(labels, probabilities, weights),
        "average_precision": float(
            average_precision_score(labels, probabilities, sample_weight=weights)
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
    }


def source_group(source: str) -> str:
    group = semantic_source_group(source)
    if group in {"epoch105", "currentgf"}:
        return "mature"
    if group in {"autopet_weak", "lndb_weak"}:
        return "weak_sources"
    if group != "lung_cells":
        raise ValueError(f"unexpected semantic source group: {source!r}")
    return group


def fit(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, seed: int):
    model = HistGradientBoostingClassifier(random_state=seed, **MODEL_PARAMETERS)
    model.fit(features, labels, sample_weight=weights)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--patch-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.folds < 2 or args.bootstrap < 0:
        parser.error("folds must be >=2 and bootstrap must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)

    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {str(row["patch_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate patch IDs in component index")
    with np.load(args.patch_predictions, allow_pickle=False) as archive:
        required = {"patch_id", "classification_probability", "fold"}
        if not required.issubset(archive.files):
            raise ValueError(f"patch prediction archive lacks {sorted(required)}")
        patch_ids = [str(value) for value in archive["patch_id"]]
        probabilities = archive["classification_probability"].astype(np.float64)
        fold_ids = archive["fold"].astype(np.int64)
    if (
        len(patch_ids) != len(set(patch_ids))
        or set(patch_ids) != set(by_id)
        or probabilities.shape != (len(rows),)
        or fold_ids.shape != (len(rows),)
        or set(fold_ids.tolist()) != set(range(args.folds))
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("invalid or incomplete patch-level OOF predictions")
    prediction_by_id = {
        patch_id: (float(probability), int(fold))
        for patch_id, probability, fold in zip(patch_ids, probabilities, fold_ids)
    }

    grouped: dict[tuple[str, str, str], list[tuple[dict[str, str], float, int]]] = defaultdict(list)
    for patch_id, row in by_id.items():
        if row.get("component_policy") != "source_separate":
            raise ValueError("component meta-classification requires source-separated proposals")
        probability, fold = prediction_by_id[patch_id]
        key = (str(row["case"]), str(row["source_name"]), str(row["source_component_id"]))
        grouped[key].append((row, probability, fold))

    features = []
    labels = []
    components = []
    component_folds = []
    cases = []
    sources = []
    baseline_probabilities = []
    for (case, source, component_id), records_with_values in sorted(grouped.items()):
        records = [record for record, _, _ in records_with_values]
        values = [value for _, value, _ in records_with_values]
        folds = {fold for _, _, fold in records_with_values}
        component_labels = {int(record["candidate_positive"]) for record in records}
        if len(folds) != 1 or len(component_labels) != 1:
            raise ValueError(f"inconsistent component target/fold: {case}/{source}/{component_id}")
        features.append(component_features(records, values))
        labels.append(next(iter(component_labels)))
        component_folds.append(next(iter(folds)))
        cases.append(case)
        sources.append(source)
        components.append(f"{case}|{source}|{component_id}")
        baseline_probabilities.append(max(values))
    features_array = np.stack(features)
    labels_array = np.asarray(labels, dtype=np.uint8)
    folds_array = np.asarray(component_folds, dtype=np.int64)
    cases_array = np.asarray(cases)
    sources_array = np.asarray(sources)
    baseline_array = np.asarray(baseline_probabilities, dtype=np.float64)
    weights = case_weights(cases_array)

    cross_fitted = np.empty(len(labels_array), dtype=np.float64)
    fold_models = {}
    for fold in range(args.folds):
        train = folds_array != fold
        held_out = ~train
        model = fit(
            features_array[train], labels_array[train], weights[train], args.seed + fold
        )
        cross_fitted[held_out] = model.predict_proba(features_array[held_out])[:, 1]
        fold_models[fold] = model
    deployment_model = fit(features_array, labels_array, weights, args.seed + args.folds)

    baseline = metrics(labels_array, baseline_array, weights)
    candidate = metrics(labels_array, cross_fitted, weights)
    group_reports = {}
    group_checks = []
    semantic_groups = np.asarray([source_group(source) for source in sources_array])
    for group in ("mature", "weak_sources", "lung_cells"):
        selected = semantic_groups == group
        group_weights = case_weights(cases_array[selected])
        base_report = metrics(labels_array[selected], baseline_array[selected], group_weights)
        candidate_report = metrics(
            labels_array[selected], cross_fitted[selected], group_weights
        )
        gain = candidate_report["average_precision"] - base_report["average_precision"]
        group_reports[group] = {
            "components": int(selected.sum()),
            "baseline": base_report,
            "candidate": candidate_report,
            "average_precision_gain": gain,
        }
        group_checks.append(gain >= -0.01)

    rng = np.random.default_rng(args.seed)
    unique_cases = sorted(set(cases))
    case_to_indices = {
        case: np.flatnonzero(cases_array == case) for case in unique_cases
    }
    bootstrap_gains = []
    for _ in range(args.bootstrap):
        sampled = rng.choice(unique_cases, len(unique_cases), replace=True)
        indices = np.concatenate([case_to_indices[str(case)] for case in sampled])
        sampled_cases = np.concatenate(
            [np.full(len(case_to_indices[str(case)]), index) for index, case in enumerate(sampled)]
        )
        sampled_weights = case_weights(sampled_cases)
        sampled_labels = labels_array[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        bootstrap_gains.append(
            float(
                average_precision_score(
                    sampled_labels, cross_fitted[indices], sample_weight=sampled_weights
                )
                - average_precision_score(
                    sampled_labels, baseline_array[indices], sample_weight=sampled_weights
                )
            )
        )
    gain_interval = (
        [float(value) for value in np.quantile(bootstrap_gains, [0.025, 0.975])]
        if bootstrap_gains
        else [float("nan"), float("nan")]
    )

    bundle_path = args.output / "component_meta_classifier.joblib"
    temporary_bundle = bundle_path.with_name(f".{bundle_path.name}.{os.getpid()}.tmp")
    joblib.dump(
        {
            "schema_version": 1,
            "feature_names": FEATURE_NAMES,
            "source_names": SOURCE_NAMES,
            "model_parameters": MODEL_PARAMETERS,
            "fold_models": fold_models,
            "deployment_model": deployment_model,
        },
        temporary_bundle,
        compress=3,
    )
    os.replace(temporary_bundle, bundle_path)
    predictions_path = args.output / "component_oof_predictions.npz"
    temporary_predictions = predictions_path.with_name(
        f".{predictions_path.name}.{os.getpid()}.tmp.npz"
    )
    np.savez_compressed(
        temporary_predictions,
        component=np.asarray(components),
        case=cases_array,
        source=sources_array,
        fold=folds_array,
        label=labels_array,
        baseline_probability=baseline_array.astype(np.float32),
        classification_probability=cross_fitted.astype(np.float32),
    )
    os.replace(temporary_predictions, predictions_path)

    ap_gain = candidate["average_precision"] - baseline["average_precision"]
    f_gain = candidate["f0p5"] - baseline["f0p5"]
    checks = {
        "average_precision_gain": ap_gain >= 0.02,
        "average_precision_gain_ci": bool(gain_interval[0] > 0),
        "f0p5_gain": f_gain >= 0.01,
        "source_nonregression": all(group_checks),
    }
    center = candidate["threshold"]
    deployment_thresholds = sorted(
        {round(float(np.clip(center + delta, 0.05, 0.95)), 2) for delta in (-0.05, 0, 0.05)}
    )
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "selection_basis": "cross-fitted exact707 grouped held-out component classification only",
        "index": str(args.index.resolve()),
        "index_sha256": sha256(args.index),
        "patch_predictions": str(args.patch_predictions.resolve()),
        "patch_predictions_sha256": sha256(args.patch_predictions),
        "components": len(labels_array),
        "positive_components": int(labels_array.sum()),
        "cases": len(unique_cases),
        "folds": args.folds,
        "feature_names": FEATURE_NAMES,
        "forbidden_label_features_excluded": True,
        "model": {"type": "sklearn HistGradientBoostingClassifier", **MODEL_PARAMETERS},
        "baseline": baseline,
        "candidate": candidate,
        "average_precision_gain": ap_gain,
        "f0p5_gain": f_gain,
        "average_precision_gain_95ci": gain_interval,
        "by_source_group": group_reports,
        "deployment_thresholds": deployment_thresholds,
        "checks": checks,
        "decision": (
            "QUALIFY_COMPONENT_META_CLASSIFIER" if all(checks.values()) else "KEEP_PATCH_CLASSIFIER"
        ),
        "model_bundle": str(bundle_path.resolve()),
        "model_bundle_sha256": sha256(bundle_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256(predictions_path),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    atomic_json(args.output / "DECISION.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
