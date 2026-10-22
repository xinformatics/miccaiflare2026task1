#!/usr/bin/env python3
"""Aggregate leakage-free candidate-refiner predictions across grouped folds."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from focus_validation_metrics import aggregate_positive_dice
from train_candidate_refiner import threshold_metrics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_transfer_result(
    sidecar_path: Path,
    index_path: Path,
    index_digest: str,
    checkpoint_path: Path,
    checkpoint: dict,
    prediction_path: Path,
) -> dict:
    """Validate explicit provenance before aggregating cross-index predictions."""

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    config = checkpoint.get("config", {})
    expected_fold = int(checkpoint.get("fold", config.get("fold", -1)))
    expected_folds = int(checkpoint.get("folds", config.get("folds", -1)))
    expected_seed = int(config.get("seed", checkpoint.get("seed", -1)))
    checks = {
        "cross_index_transfer": payload.get("cross_index_transfer") is True,
        "checkpoint_path": payload.get("checkpoint") == str(checkpoint_path.resolve()),
        "checkpoint_sha256": payload.get("checkpoint_sha256")
        == file_sha256(checkpoint_path),
        "training_index_sha256": payload.get("training_index_sha256")
        == checkpoint.get("index_sha256"),
        "evaluation_index": payload.get("evaluation_index")
        == str(index_path.resolve()),
        "evaluation_index_sha256": payload.get("evaluation_index_sha256")
        == index_digest,
        "prediction_path": payload.get("predictions")
        == str(prediction_path.resolve()),
        "prediction_sha256": payload.get("predictions_sha256")
        == file_sha256(prediction_path),
        "fold": int(payload.get("fold", -1)) == expected_fold,
        "folds": int(payload.get("folds", -1)) == expected_folds,
        "seed": int(payload.get("seed", -1)) == expected_seed,
        "public_blind": payload.get("public_read") is False,
        "hidden_blind": payload.get("hidden_read") is False,
        "metrics": isinstance(payload.get("metrics"), dict),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"invalid transfer sidecar {sidecar_path}: failed={failed}"
        )
    return payload


def validate_composition_result(
    audit_path: Path,
    index_digest: str,
    classification_checkpoint_path: Path,
    classification_checkpoint: dict,
    contour_checkpoint_path: Path,
    contour_checkpoint: dict,
    prediction_path: Path,
) -> dict:
    """Bind a composed prediction archive to both matched-fold checkpoints."""

    payload = json.loads(audit_path.read_text(encoding="utf-8"))

    def identity(checkpoint_path: Path, checkpoint: dict) -> dict[str, object]:
        config = checkpoint.get("config", {})
        return {
            "path": str(checkpoint_path.resolve()),
            "sha256": file_sha256(checkpoint_path),
            "fold": int(checkpoint.get("fold", config.get("fold", -1))),
            "folds": int(checkpoint.get("folds", config.get("folds", -1))),
            "seed": int(config.get("seed", checkpoint.get("seed", -1))),
            "epoch": int(checkpoint.get("epoch", -1)),
            "index_sha256": str(checkpoint.get("index_sha256", "")),
        }

    classification_identity = identity(
        classification_checkpoint_path, classification_checkpoint
    )
    contour_identity = identity(contour_checkpoint_path, contour_checkpoint)
    recorded_classification = payload.get("classification_checkpoint", {})
    recorded_contour = payload.get("contour_checkpoint", {})
    classification_predictions = Path(str(payload.get("classification", "")))
    contour_predictions = Path(str(payload.get("contour", "")))
    checks = {
        "schema": payload.get("schema_version") == 1,
        "classification_checkpoint": recorded_classification
        == classification_identity,
        "contour_checkpoint": recorded_contour == contour_identity,
        "matched_fold_identity": all(
            classification_identity[key] == contour_identity[key]
            for key in ("fold", "folds", "seed", "index_sha256")
        ),
        "index": classification_identity["index_sha256"] == index_digest,
        "classification_predictions": classification_predictions.is_file()
        and payload.get("classification_sha256")
        == file_sha256(classification_predictions),
        "contour_predictions": contour_predictions.is_file()
        and payload.get("contour_sha256") == file_sha256(contour_predictions),
        "output": payload.get("output") == str(prediction_path.resolve())
        and payload.get("output_sha256") == file_sha256(prediction_path),
        "label_blind": payload.get("labels_read") is False,
        "public_blind": payload.get("public_read") is False,
        "hidden_blind": payload.get("hidden_read") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid composition audit {audit_path}: failed={failed}")
    return payload


def validate_source_composition_result(
    audit_path: Path,
    index_digest: str,
    classification_checkpoint_path: Path,
    classification_checkpoint: dict,
    prediction_path: Path,
) -> dict:
    """Bind a source-routed archive to every matched-fold checkpoint."""

    payload = json.loads(audit_path.read_text(encoding="utf-8"))

    def identity(checkpoint_path: Path, checkpoint: dict) -> dict[str, object]:
        config = checkpoint.get("config", {})
        return {
            "path": str(checkpoint_path.resolve()),
            "sha256": file_sha256(checkpoint_path),
            "fold": int(checkpoint.get("fold", config.get("fold", -1))),
            "folds": int(checkpoint.get("folds", config.get("folds", -1))),
            "seed": int(config.get("seed", checkpoint.get("seed", -1))),
            "epoch": int(checkpoint.get("epoch", -1)),
            "index_sha256": str(checkpoint.get("index_sha256", "")),
        }

    classification_identity = identity(
        classification_checkpoint_path, classification_checkpoint
    )
    classification_predictions = Path(str(payload.get("classification", "")))
    routes = payload.get("contour_routes", {})
    source_to_route = payload.get("source_to_route", {})
    route_checks: dict[str, bool] = {}
    route_identities: dict[str, dict[str, object]] = {}
    if isinstance(routes, dict):
        for name, route in routes.items():
            if not isinstance(route, dict):
                route_checks[str(name)] = False
                continue
            checkpoint_record = route.get("contour_checkpoint", {})
            checkpoint_path = Path(str(checkpoint_record.get("path", "")))
            contour_predictions = Path(str(route.get("contour", "")))
            if not checkpoint_path.is_file():
                route_checks[str(name)] = False
                continue
            route_checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            route_identity = identity(checkpoint_path, route_checkpoint)
            route_identities[str(name)] = route_identity
            route_checks[str(name)] = bool(
                checkpoint_record == route_identity
                and all(
                    classification_identity[key] == route_identity[key]
                    for key in ("fold", "folds", "seed", "index_sha256")
                )
                and contour_predictions.is_file()
                and route.get("contour_sha256")
                == file_sha256(contour_predictions)
                and isinstance(route.get("sources"), list)
                and route.get("sources")
            )
    route_mapping = routes if isinstance(routes, dict) else {}
    declared_sources = {
        str(source)
        for route in route_mapping.values()
        if isinstance(route, dict)
        for source in route.get("sources", [])
    }
    mapped_sources = set(source_to_route) if isinstance(source_to_route, dict) else set()
    mapped_routes = (
        set(source_to_route.values()) if isinstance(source_to_route, dict) else set()
    )
    checks = {
        "schema": payload.get("schema_version") == 2,
        "classification_checkpoint": payload.get("classification_checkpoint", {})
        == classification_identity,
        "index": payload.get("index_sha256") == index_digest
        == classification_identity["index_sha256"],
        "classification_predictions": classification_predictions.is_file()
        and payload.get("classification_sha256")
        == file_sha256(classification_predictions),
        "routes_present": bool(route_mapping),
        "routes_valid": bool(route_checks) and all(route_checks.values()),
        "source_mapping_complete": mapped_sources == declared_sources
        and mapped_routes == set(route_mapping),
        "output": payload.get("output") == str(prediction_path.resolve())
        and payload.get("output_sha256") == file_sha256(prediction_path),
        "labels_only_checked_for_identity": payload.get(
            "labels_read_for_identity_only"
        )
        is True,
        "label_blind_routing": payload.get("labels_used_for_routing") is False,
        "public_blind": payload.get("public_read") is False,
        "hidden_blind": payload.get("hidden_read") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"invalid source composition audit {audit_path}: failed={failed}"
        )
    payload["validated_route_checkpoint_identities"] = route_identities
    return payload


def interval(values: list[float]) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def deployment_classification_thresholds(center: float, count: int = 3) -> list[float]:
    """Return an exact set contained in the threshold grid evaluated in OOF."""

    if count < 1 or not np.isfinite(center) or not 0 <= center <= 1:
        raise ValueError("invalid threshold request")
    offsets = [0.0]
    if center >= 0.90 - 1e-8:
        # The patch evaluator only calibrates through 0.95.  Near that edge,
        # move downward rather than extrapolating to an unsupported 0.99.
        offsets.extend(-0.05 * step for step in range(1, 20))
    else:
        for step in range(1, 20):
            offsets.extend((0.05 * step, -0.05 * step))
    selected: list[float] = []
    for offset in offsets:
        value = round(min(0.95, max(0.05, center + offset)), 2)
        if value not in selected:
            selected.append(value)
        if len(selected) == count:
            return sorted(selected)
    raise ValueError("could not construct distinct deployment thresholds")


def case_balanced_threshold_metrics(
    expected: dict[str, dict[str, str]],
    patch_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int | str]:
    """Calibrate F0.5 after giving every held-out case equal total weight.

    Candidate counts vary substantially by scan.  Pooled patch counts would
    otherwise let scans with many proposals choose a deployment threshold for
    the entire cohort, while the challenge metric gives each scan equal weight.
    """

    if not (len(patch_ids) == len(labels) == len(probabilities)):
        raise ValueError("classification arrays do not align")
    if not patch_ids:
        raise ValueError("classification calibration is empty")
    case_counts: dict[str, int] = {}
    cases: list[str] = []
    for patch_id in patch_ids:
        if patch_id not in expected or not str(expected[patch_id].get("case", "")):
            raise ValueError(f"missing case identity: {patch_id}")
        case = str(expected[patch_id]["case"])
        cases.append(case)
        case_counts[case] = case_counts.get(case, 0) + 1
    weights = np.asarray([1.0 / case_counts[case] for case in cases], dtype=np.float64)
    best = None
    for threshold in np.linspace(0.05, 0.95, 19):
        selected = probabilities >= threshold
        true_positive = float(weights[selected & (labels == 1)].sum())
        false_positive = float(weights[selected & (labels == 0)].sum())
        false_negative = float(weights[~selected & (labels == 1)].sum())
        precision = true_positive / max(true_positive + false_positive, 1e-12)
        recall = true_positive / max(true_positive + false_negative, 1e-12)
        beta2 = 0.25
        fbeta = (1 + beta2) * precision * recall / max(
            beta2 * precision + recall, 1e-12
        )
        candidate = (fbeta, precision, recall, -threshold)
        if best is None or candidate > best[0]:
            best = (candidate, threshold, precision, recall, fbeta)
    assert best is not None
    return {
        "threshold": float(best[1]),
        "precision": float(best[2]),
        "recall": float(best[3]),
        "f0p5": float(best[4]),
        "aggregation": "patches weighted to equal total weight per case",
        "cases": len(case_counts),
    }


def deployment_segmentation_thresholds(center: float, count: int = 3) -> list[float]:
    """Return a bounded contour audit, extending upward at a grid ceiling."""

    if count < 1 or not np.isfinite(center) or not 0 < center < 1:
        raise ValueError("invalid segmentation threshold request")
    offsets = [0.0]
    if center >= 0.80 - 1e-8:
        offsets.extend(0.05 * step for step in range(1, 20))
        offsets.extend(-0.05 * step for step in range(1, 20))
    else:
        for step in range(1, 20):
            offsets.extend((0.05 * step, -0.05 * step))
    selected: list[float] = []
    for offset in offsets:
        value = round(min(0.95, max(0.05, center + offset)), 2)
        if value not in selected:
            selected.append(value)
        if len(selected) == count:
            return sorted(selected)
    raise ValueError("could not construct distinct segmentation thresholds")


def cohort_metrics(
    expected: dict[str, dict[str, str]],
    patch_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    classification_threshold: float,
    positive_ids: list[str],
    segmentation_grid: np.ndarray,
    segmentation_index: int,
) -> list[dict[str, object]]:
    """Report fixed-global-threshold performance without cohort retuning."""

    if not (len(patch_ids) == len(labels) == len(probabilities)):
        raise ValueError("classification arrays do not align")
    if len(positive_ids) != len(segmentation_grid):
        raise ValueError("segmentation arrays do not align")
    cohorts = sorted({str(row["cohort"]) for row in expected.values()})
    reports: list[dict[str, object]] = []
    for cohort in cohorts:
        indices = np.asarray(
            [
                index
                for index, patch_id in enumerate(patch_ids)
                if str(expected[patch_id]["cohort"]) == cohort
            ],
            dtype=np.int64,
        )
        positive_indices = np.asarray(
            [
                index
                for index, patch_id in enumerate(positive_ids)
                if str(expected[patch_id]["cohort"]) == cohort
            ],
            dtype=np.int64,
        )
        cohort_labels = labels[indices]
        cohort_probabilities = probabilities[indices]
        selected = cohort_probabilities >= classification_threshold
        true_positive = int(np.count_nonzero(selected & (cohort_labels == 1)))
        false_positive = int(np.count_nonzero(selected & (cohort_labels == 0)))
        false_negative = int(np.count_nonzero(~selected & (cohort_labels == 1)))
        has_both_labels = np.unique(cohort_labels).size == 2
        cases = {str(expected[patch_ids[index]]["case"]) for index in indices}
        positive_case_indices: dict[str, list[int]] = {}
        for index in positive_indices:
            case = str(expected[positive_ids[index]]["case"])
            positive_case_indices.setdefault(case, []).append(int(index))
        case_dice = [
            float(segmentation_grid[values, segmentation_index].mean())
            for values in positive_case_indices.values()
        ]
        reports.append(
            {
                "cohort": cohort,
                "cases": len(cases),
                "patches": int(len(indices)),
                "positive_patches": int(cohort_labels.sum()),
                "negative_patches": int(np.count_nonzero(cohort_labels == 0)),
                "classification_average_precision": (
                    float(
                        average_precision_score(
                            cohort_labels, cohort_probabilities
                        )
                    )
                    if has_both_labels
                    else None
                ),
                "classification_precision_at_global_threshold": float(
                    true_positive / max(true_positive + false_positive, 1)
                ),
                "classification_recall_at_global_threshold": float(
                    true_positive / max(true_positive + false_negative, 1)
                ),
                "segmentation_positive_patches": int(len(positive_indices)),
                "segmentation_positive_cases": len(positive_case_indices),
                "segmentation_dice_at_global_threshold": (
                    float(np.mean(case_dice))
                    if case_dice
                    else None
                ),
            }
        )
    if sum(int(row["patches"]) for row in reports) != len(patch_ids):
        raise AssertionError("cohort classification coverage mismatch")
    if sum(int(row["segmentation_positive_patches"]) for row in reports) != len(
        positive_ids
    ):
        raise AssertionError("cohort segmentation coverage mismatch")
    return reports


def source_classification_metrics(
    expected: dict[str, dict[str, str]],
    patch_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Calibrate fixed proposal-source gates from held-out patch predictions.

    A source-separated patch index must identify every patch with one nonempty
    ``source_name``.  Returning an empty mapping for a legacy index keeps the
    historical single/global-threshold path byte-for-byte compatible.
    """

    if not (len(patch_ids) == len(labels) == len(probabilities)):
        raise ValueError("source classification arrays do not align")
    has_source = [bool(str(expected[patch_id].get("source_name", "")).strip()) for patch_id in patch_ids]
    if not any(has_source):
        return {}
    if not all(has_source):
        raise ValueError("source_name must cover every patch or no patches")
    reports: dict[str, dict[str, float | int]] = {}
    sources = sorted(
        {str(expected[patch_id]["source_name"]).strip() for patch_id in patch_ids}
    )
    for source in sources:
        indices = np.asarray(
            [
                index
                for index, patch_id in enumerate(patch_ids)
                if str(expected[patch_id]["source_name"]).strip() == source
            ],
            dtype=np.int64,
        )
        source_patch_ids = [patch_ids[index] for index in indices]
        source_labels = labels[indices]
        source_probabilities = probabilities[indices]
        if np.unique(source_labels).size != 2:
            raise ValueError(
                f"source-specific calibration requires both labels: {source}"
            )
        calibrated = case_balanced_threshold_metrics(
            expected,
            source_patch_ids,
            source_labels,
            source_probabilities,
        )
        case_counts: dict[str, int] = {}
        source_cases = [str(expected[patch_id]["case"]) for patch_id in source_patch_ids]
        for case in source_cases:
            case_counts[case] = case_counts.get(case, 0) + 1
        weights = np.asarray(
            [1.0 / case_counts[case] for case in source_cases], dtype=np.float64
        )
        reports[source] = {
            **calibrated,
            "patches": int(len(indices)),
            "positive": int(source_labels.sum()),
            "negative": int(np.count_nonzero(source_labels == 0)),
            "average_precision": float(
                average_precision_score(
                    source_labels, source_probabilities, sample_weight=weights
                )
            ),
            "roc_auc": float(
                roc_auc_score(
                    source_labels, source_probabilities, sample_weight=weights
                )
            ),
            "patch_pooled_average_precision": float(
                average_precision_score(source_labels, source_probabilities)
            ),
            "patch_pooled_roc_auc": float(
                roc_auc_score(source_labels, source_probabilities)
            ),
        }
    if sum(int(report["patches"]) for report in reports.values()) != len(patch_ids):
        raise AssertionError("source classification coverage mismatch")
    return reports


def source_segmentation_metrics(
    expected: dict[str, dict[str, str]],
    positive_ids: list[str],
    segmentation_grid: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Calibrate contour thresholds per proposal source from held-out patches.

    Positive contours from different proposal families have radically different
    target scales. Each source threshold is therefore selected with the same
    case-balanced Dice rule as the global contour. Legacy indexes without a
    source identity retain the historical global-only path.
    """

    if len(positive_ids) != len(segmentation_grid):
        raise ValueError("source segmentation arrays do not align")
    has_source = [
        bool(str(expected[patch_id].get("source_name", "")).strip())
        for patch_id in positive_ids
    ]
    if not any(has_source):
        return {}
    if not all(has_source):
        raise ValueError("source_name must cover every positive patch or no patches")
    reports: dict[str, dict[str, object]] = {}
    sources = sorted(
        {str(expected[patch_id]["source_name"]).strip() for patch_id in positive_ids}
    )
    for source in sources:
        indices = np.asarray(
            [
                index
                for index, patch_id in enumerate(positive_ids)
                if str(expected[patch_id]["source_name"]).strip() == source
            ],
            dtype=np.int64,
        )
        source_ids = [positive_ids[index] for index in indices]
        source_rows = [expected[patch_id] for patch_id in source_ids]
        aggregate = aggregate_positive_dice(
            source_ids, segmentation_grid[indices], thresholds, source_rows
        )
        best = int(aggregate["best_index"])
        reports[source] = {
            "threshold": float(thresholds[best]),
            "dice": float(aggregate["case_balanced_dice"]),
            "patch_mean_dice": float(aggregate["patch_mean_dice"]),
            "cohort_balanced_dice": float(aggregate["cohort_balanced_dice"]),
            "positive_patches": int(aggregate["positive_patches"]),
            "positive_cases": int(aggregate["positive_cases"]),
            "positive_cohorts": int(aggregate["positive_cohorts"]),
            "thresholds": [float(value) for value in thresholds],
            "mean_dice_by_threshold": [
                float(value) for value in aggregate["case_balanced_grid"]
            ],
        }
    if sum(int(report["positive_patches"]) for report in reports.values()) != len(
        positive_ids
    ):
        raise AssertionError("source segmentation coverage mismatch")
    return reports


def pool_source_segmentation_groups(
    reports: dict[str, dict[str, object]],
    expected: dict[str, dict[str, str]],
    positive_ids: list[str],
    segmentation_grid: np.ndarray,
    thresholds: np.ndarray,
    specifications: list[str],
) -> dict[str, dict[str, object]]:
    """Pool contour calibration for exchangeable proposal-source identities."""

    grouped: dict[str, dict[str, object]] = {}
    claimed: set[str] = set()
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"invalid source group: {specification}")
        name, raw_members = specification.split("=", 1)
        name = name.strip()
        members = sorted(
            {value.strip() for value in raw_members.split(",") if value.strip()}
        )
        missing = set(members) - set(reports)
        overlap = set(members) & claimed
        if not name or len(members) < 2 or missing or overlap:
            raise ValueError(
                f"invalid source group {name}: missing={sorted(missing)} "
                f"overlap={sorted(overlap)}"
            )
        claimed.update(members)
        indices = np.asarray(
            [
                index
                for index, patch_id in enumerate(positive_ids)
                if str(expected[patch_id]["source_name"]).strip() in members
            ],
            dtype=np.int64,
        )
        group_ids = [positive_ids[index] for index in indices]
        group_rows = [expected[patch_id] for patch_id in group_ids]
        aggregate = aggregate_positive_dice(
            group_ids, segmentation_grid[indices], thresholds, group_rows
        )
        best = int(aggregate["best_index"])
        grouped[name] = {
            "sources": members,
            "threshold": float(thresholds[best]),
            "dice": float(aggregate["case_balanced_dice"]),
            "patch_mean_dice": float(aggregate["patch_mean_dice"]),
            "cohort_balanced_dice": float(aggregate["cohort_balanced_dice"]),
            "positive_patches": int(aggregate["positive_patches"]),
            "positive_cases": int(aggregate["positive_cases"]),
            "positive_cohorts": int(aggregate["positive_cohorts"]),
            "thresholds": [float(value) for value in thresholds],
            "mean_dice_by_threshold": [
                float(value) for value in aggregate["case_balanced_grid"]
            ],
        }
        for source in members:
            reports[source]["individual_threshold"] = reports[source]["threshold"]
            reports[source]["threshold"] = float(thresholds[best])
            reports[source]["threshold_group"] = name
    return grouped


def pool_source_classification_groups(
    reports: dict[str, dict[str, float | int]],
    expected: dict[str, dict[str, str]],
    patch_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    specifications: list[str],
) -> dict[str, dict[str, object]]:
    """Share one held-out threshold across exchangeable proposal sources.

    Parity-separated lung cells must remain separate connected-component
    sources, but their colour identity is arbitrary. Pooling only their OOF
    classification rows avoids fitting eight noisy colour-specific gates.
    """

    grouped: dict[str, dict[str, object]] = {}
    claimed: set[str] = set()
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"invalid source group: {specification}")
        name, raw_members = specification.split("=", 1)
        name = name.strip()
        members = sorted({value.strip() for value in raw_members.split(",") if value.strip()})
        if not name or len(members) < 2:
            raise ValueError(f"invalid source group: {specification}")
        missing = set(members) - set(reports)
        overlap = set(members) & claimed
        if missing or overlap:
            raise ValueError(
                f"invalid source group {name}: missing={sorted(missing)} "
                f"overlap={sorted(overlap)}"
            )
        claimed.update(members)
        indices = np.asarray(
            [
                index
                for index, patch_id in enumerate(patch_ids)
                if str(expected[patch_id]["source_name"]).strip() in members
            ],
            dtype=np.int64,
        )
        group_patch_ids = [patch_ids[index] for index in indices]
        group_labels = labels[indices]
        group_probabilities = probabilities[indices]
        if np.unique(group_labels).size != 2:
            raise ValueError(f"source group requires both labels: {name}")
        calibrated = case_balanced_threshold_metrics(
            expected,
            group_patch_ids,
            group_labels,
            group_probabilities,
        )
        case_counts: dict[str, int] = {}
        group_cases = [str(expected[patch_id]["case"]) for patch_id in group_patch_ids]
        for case in group_cases:
            case_counts[case] = case_counts.get(case, 0) + 1
        weights = np.asarray(
            [1.0 / case_counts[case] for case in group_cases], dtype=np.float64
        )
        grouped[name] = {
            **calibrated,
            "sources": members,
            "patches": int(len(indices)),
            "positive": int(group_labels.sum()),
            "negative": int(np.count_nonzero(group_labels == 0)),
            "average_precision": float(
                average_precision_score(
                    group_labels, group_probabilities, sample_weight=weights
                )
            ),
            "roc_auc": float(
                roc_auc_score(
                    group_labels, group_probabilities, sample_weight=weights
                )
            ),
            "patch_pooled_average_precision": float(
                average_precision_score(group_labels, group_probabilities)
            ),
            "patch_pooled_roc_auc": float(
                roc_auc_score(group_labels, group_probabilities)
            ),
        }
        for source in members:
            reports[source]["individual_threshold"] = reports[source]["threshold"]
            reports[source]["threshold"] = calibrated["threshold"]
            reports[source]["threshold_group"] = name
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument(
        "--contour-checkpoint",
        type=Path,
        action="append",
        help=(
            "one segmentation-selected candidate-refiner checkpoint per fold; "
            "requires a matching --composition-audit"
        ),
    )
    parser.add_argument(
        "--composition-audit",
        type=Path,
        action="append",
        help="one compose_candidate_refiner_validation_predictions audit per fold",
    )
    parser.add_argument(
        "--source-composition-audit",
        type=Path,
        action="append",
        help=(
            "one schema-2 source-routed composition audit per fold; mutually "
            "exclusive with --contour-checkpoint/--composition-audit"
        ),
    )
    parser.add_argument(
        "--transfer-result",
        type=Path,
        action="append",
        help="one explicit cross-index provenance sidecar per checkpoint/prediction",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--source-group",
        action="append",
        default=[],
        metavar="NAME=SOURCE1,SOURCE2",
        help="pool exchangeable proposal sources for one OOF classification threshold",
    )
    args = parser.parse_args()
    if len(args.checkpoint) != len(args.prediction) or len(args.checkpoint) < 2:
        parser.error("checkpoint and prediction lists must have equal length >=2")
    if args.transfer_result is not None and len(args.transfer_result) != len(args.checkpoint):
        parser.error("transfer-result count must match checkpoint/prediction count")
    if bool(args.contour_checkpoint) != bool(args.composition_audit):
        parser.error("contour-checkpoint and composition-audit must be supplied together")
    if args.contour_checkpoint is not None and (
        len(args.contour_checkpoint) != len(args.checkpoint)
        or len(args.composition_audit) != len(args.checkpoint)
    ):
        parser.error("decoupled checkpoint/audit counts must match predictions")
    if args.source_composition_audit is not None and (
        args.contour_checkpoint is not None or args.composition_audit is not None
    ):
        parser.error("source composition audits are mutually exclusive with legacy composition")
    if args.source_composition_audit is not None and len(
        args.source_composition_audit
    ) != len(args.checkpoint):
        parser.error("source composition audit count must match predictions")
    transfer_results = args.transfer_result or [None] * len(args.checkpoint)
    contour_checkpoints = args.contour_checkpoint or [None] * len(args.checkpoint)
    composition_audits = args.composition_audit or [None] * len(args.checkpoint)
    source_composition_audits = args.source_composition_audit or [None] * len(
        args.checkpoint
    )

    with args.index.open(newline="", encoding="utf-8") as stream:
        index_rows = list(csv.DictReader(stream))
    expected = {row["patch_id"]: row for row in index_rows}
    index_digest = hashlib.sha256(args.index.read_bytes()).hexdigest()
    folds = []
    patch_ids, labels, probabilities = [], [], []
    positive_ids, segmentation_grids = [], []
    thresholds = None
    for checkpoint_path, prediction_path, transfer_result, contour_path, composition_audit, source_composition_audit in zip(
        args.checkpoint,
        args.prediction,
        transfer_results,
        contour_checkpoints,
        composition_audits,
        source_composition_audits,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        is_transfer = checkpoint.get("index_sha256") != index_digest
        if is_transfer and transfer_result is None:
            raise ValueError(f"index digest mismatch: {checkpoint_path}")
        if not is_transfer and transfer_result is not None:
            raise ValueError(
                f"transfer sidecar supplied for native-index checkpoint: {checkpoint_path}"
            )
        transfer_payload = (
            validate_transfer_result(
                transfer_result,
                args.index,
                index_digest,
                checkpoint_path,
                checkpoint,
                prediction_path,
            )
            if transfer_result is not None
            else None
        )
        composition_payload = None
        source_composition_payload = None
        contour_checkpoint = None
        if contour_path is not None:
            if is_transfer:
                raise ValueError("decoupled OOF does not support cross-index transfer")
            assert composition_audit is not None
            contour_checkpoint = torch.load(
                contour_path, map_location="cpu", weights_only=False
            )
            composition_payload = validate_composition_result(
                composition_audit,
                index_digest,
                checkpoint_path,
                checkpoint,
                contour_path,
                contour_checkpoint,
                prediction_path,
            )
        if source_composition_audit is not None:
            if is_transfer:
                raise ValueError("source-routed OOF does not support cross-index transfer")
            source_composition_payload = validate_source_composition_result(
                source_composition_audit,
                index_digest,
                checkpoint_path,
                checkpoint,
                prediction_path,
            )
        with np.load(prediction_path, allow_pickle=False) as archive:
            fold_ids = archive["patch_id"].astype(str)
            fold_labels = archive["label"].astype(np.uint8)
            fold_probabilities = archive["classification_probability"].astype(np.float64)
            fold_positive_ids = archive["positive_patch_id"].astype(str)
            fold_thresholds = archive["segmentation_thresholds"].astype(np.float64)
            fold_grid = archive["segmentation_dice_grid"].astype(np.float64)
        if not (len(fold_ids) == len(fold_labels) == len(fold_probabilities)):
            raise ValueError(f"classification row mismatch: {prediction_path}")
        if fold_grid.shape != (len(fold_positive_ids), len(fold_thresholds)):
            raise ValueError(f"segmentation grid mismatch: {prediction_path}")
        if thresholds is None:
            thresholds = fold_thresholds
        elif not np.array_equal(thresholds, fold_thresholds):
            raise ValueError("fold segmentation threshold grids differ")
        folds.append(
            {
                "fold": int(checkpoint["fold"]),
                "epoch": int(checkpoint["epoch"]),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "prediction": str(prediction_path.resolve()),
                "prediction_sha256": file_sha256(prediction_path),
                "contour_epoch": (
                    int(contour_checkpoint["epoch"])
                    if contour_checkpoint is not None
                    else (
                        None
                        if source_composition_payload is not None
                        else int(checkpoint["epoch"])
                    )
                ),
                "contour_epochs_by_route": (
                    {
                        name: int(identity["epoch"])
                        for name, identity in source_composition_payload[
                            "validated_route_checkpoint_identities"
                        ].items()
                    }
                    if source_composition_payload is not None
                    else None
                ),
                "decoupled_classification_and_contour": (
                    composition_payload is not None
                    or source_composition_payload is not None
                ),
                "source_routed_contours": source_composition_payload is not None,
                "contour_checkpoint": (
                    str(contour_path.resolve()) if contour_path is not None else None
                ),
                "contour_checkpoint_sha256": (
                    file_sha256(contour_path) if contour_path is not None else None
                ),
                "cases": len({value.rsplit("__c", 1)[0] for value in fold_ids}),
                "patches": len(fold_ids),
                "cross_index_transfer": is_transfer,
                **(
                    transfer_payload["metrics"]
                    if transfer_payload is not None
                    else checkpoint["metrics"]
                ),
            }
        )
        patch_ids.extend(fold_ids.tolist())
        labels.extend(fold_labels.tolist())
        probabilities.extend(fold_probabilities.tolist())
        positive_ids.extend(fold_positive_ids.tolist())
        segmentation_grids.append(fold_grid)

    if len(patch_ids) != len(set(patch_ids)) or set(patch_ids) != set(expected):
        missing = sorted(set(expected) - set(patch_ids))[:10]
        extra = sorted(set(patch_ids) - set(expected))[:10]
        raise ValueError(f"OOF coverage mismatch missing={missing} extra={extra}")
    labels_array = np.asarray(labels, dtype=np.uint8)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    for patch_id, label in zip(patch_ids, labels_array):
        if int(expected[patch_id]["label"]) != int(label):
            raise ValueError(f"label mismatch: {patch_id}")
    if len(positive_ids) != len(set(positive_ids)) or set(positive_ids) != {
        patch_id for patch_id, label in zip(patch_ids, labels_array) if label
    }:
        raise ValueError("positive OOF coverage mismatch")

    assert thresholds is not None
    segmentation_grid = np.concatenate(segmentation_grids, axis=0)
    segmentation_aggregate = aggregate_positive_dice(
        positive_ids,
        segmentation_grid,
        thresholds,
        index_rows,
    )
    segmentation_index = int(segmentation_aggregate["best_index"])
    segmentation_case_means = segmentation_aggregate["case_balanced_grid"]
    segmentation_patch_means = segmentation_aggregate["patch_mean_grid"]
    segmentation_cohort_means = segmentation_aggregate["cohort_balanced_grid"]

    case_to_indices: dict[str, list[int]] = {}
    for index, patch_id in enumerate(patch_ids):
        case_to_indices.setdefault(str(expected[patch_id]["case"]), []).append(index)
    case_weights = np.empty(len(patch_ids), dtype=np.float64)
    for indices in case_to_indices.values():
        case_weights[indices] = 1.0 / len(indices)
    classification = case_balanced_threshold_metrics(
        expected,
        patch_ids,
        labels_array,
        probabilities_array,
    )
    patch_pooled_classification = threshold_metrics(
        labels_array, probabilities_array
    )
    prevalence = float(np.average(labels_array, weights=case_weights))
    patch_pooled_prevalence = float(labels_array.mean())
    average_precision = float(
        average_precision_score(
            labels_array, probabilities_array, sample_weight=case_weights
        )
    )
    patch_pooled_average_precision = float(
        average_precision_score(labels_array, probabilities_array)
    )
    roc_auc = float(
        roc_auc_score(labels_array, probabilities_array, sample_weight=case_weights)
    )
    patch_pooled_roc_auc = float(
        roc_auc_score(labels_array, probabilities_array)
    )
    positive_case_to_indices: dict[str, list[int]] = {}
    for index, patch_id in enumerate(positive_ids):
        positive_case_to_indices.setdefault(
            str(expected[patch_id]["case"]), []
        ).append(index)
    cases = sorted(case_to_indices)
    positive_cases = sorted(positive_case_to_indices)
    rng = np.random.default_rng(args.seed)
    bootstrap_ap, bootstrap_segmentation = [], []
    for _ in range(args.bootstrap):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        class_indices = np.asarray(
            [index for case in sampled for index in case_to_indices[case]], dtype=int
        )
        sampled_weights = np.asarray(
            [
                1.0 / len(case_to_indices[case])
                for case in sampled
                for _ in case_to_indices[case]
            ],
            dtype=np.float64,
        )
        sample_labels = labels_array[class_indices]
        if np.unique(sample_labels).size == 2:
            bootstrap_ap.append(
                float(
                    average_precision_score(
                        sample_labels,
                        probabilities_array[class_indices],
                        sample_weight=sampled_weights,
                    )
                )
            )
        sampled_positive = rng.choice(
            positive_cases, size=len(positive_cases), replace=True
        )
        sampled_case_dice = [
            float(
                segmentation_grid[
                    positive_case_to_indices[case], segmentation_index
                ].mean()
            )
            for case in sampled_positive
        ]
        if sampled_case_dice:
            bootstrap_segmentation.append(
                float(np.mean(sampled_case_dice))
            )

    segmentation_dice = float(segmentation_case_means[segmentation_index])
    by_cohort = cohort_metrics(
        expected,
        patch_ids,
        labels_array,
        probabilities_array,
        float(classification["threshold"]),
        positive_ids,
        segmentation_grid,
        segmentation_index,
    )
    by_source = source_classification_metrics(
        expected,
        patch_ids,
        labels_array,
        probabilities_array,
    )
    source_groups = pool_source_classification_groups(
        by_source,
        expected,
        patch_ids,
        labels_array,
        probabilities_array,
        args.source_group,
    )
    source_segmentation = source_segmentation_metrics(
        expected,
        positive_ids,
        segmentation_grid,
        thresholds,
    )
    source_segmentation_groups = pool_source_segmentation_groups(
        source_segmentation,
        expected,
        positive_ids,
        segmentation_grid,
        thresholds,
        args.source_group,
    )
    recommendation = (
        "PROMISING_FOR_707_SCALE"
        if average_precision >= prevalence + 0.15
        and classification["precision"] >= 0.50
        and classification["recall"] >= 0.20
        and segmentation_dice >= 0.50
        else "REVIEW_BEFORE_SCALE"
    )
    report = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "index": str(args.index.resolve()),
        "index_sha256": index_digest,
        "folds": sorted(folds, key=lambda row: row["fold"]),
        "coverage": {
            "cases": len(cases),
            "patches": len(patch_ids),
            "positive": int(labels_array.sum()),
            "negative": int((labels_array == 0).sum()),
            "every_patch_exactly_once": True,
        },
        "classification": {
            **classification,
            "deployment_thresholds": deployment_classification_thresholds(
                float(classification["threshold"])
            ),
            "prevalence": prevalence,
            "average_precision": average_precision,
            "average_precision_case_bootstrap_95ci": interval(bootstrap_ap),
            "roc_auc": roc_auc,
            "patch_pooled_threshold_metrics": patch_pooled_classification,
            "patch_pooled_prevalence": patch_pooled_prevalence,
            "patch_pooled_average_precision": patch_pooled_average_precision,
            "patch_pooled_roc_auc": patch_pooled_roc_auc,
        },
        "segmentation": {
            "threshold": float(thresholds[segmentation_index]),
            "dice": segmentation_dice,
            "aggregation": "patches averaged within case, then cases averaged",
            "positive_cases": int(segmentation_aggregate["positive_cases"]),
            "patch_mean_dice": float(
                segmentation_patch_means[segmentation_index]
            ),
            "cohort_balanced_dice": float(
                segmentation_cohort_means[segmentation_index]
            ),
            "dice_case_bootstrap_95ci": interval(bootstrap_segmentation),
            "thresholds": thresholds.tolist(),
            "mean_dice_by_threshold": segmentation_case_means.tolist(),
            "patch_mean_dice_by_threshold": segmentation_patch_means.tolist(),
            "cohort_balanced_dice_by_threshold": segmentation_cohort_means.tolist(),
        },
        "cohort_summary": by_cohort,
        "source_classification": by_source,
        "source_classification_groups": source_groups,
        "source_segmentation": source_segmentation,
        "source_segmentation_groups": source_segmentation_groups,
        "scale_recommendation": recommendation,
        "scale_rule": {
            "average_precision_lift_over_prevalence": 0.15,
            "precision": 0.50,
            "recall": 0.20,
            "segmentation_dice": 0.50,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
