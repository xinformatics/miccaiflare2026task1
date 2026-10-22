#!/usr/bin/env python3
"""Predeclared patch-OOF gate for STU seed-component confirmation.

STU already provides a full-volume contour.  This selector therefore judges
the refiner only as a component/tile classifier and freezes a high-precision
classification threshold before full-volume OOF.  It never reads public or
hidden data.
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
from sklearn.metrics import average_precision_score


EXPECTED_CASES = 707
THRESHOLDS = tuple(round(float(value), 2) for value in np.arange(0.50, 0.951, 0.05))
PRECISION_MINIMUM = 0.90
PRECISION_LOWER_CI_MINIMUM = 0.86
RECALL_MINIMUM = 0.15
AP_LIFT_MINIMUM = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_balanced_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    cases: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    counts = {case: int(np.count_nonzero(cases == case)) for case in np.unique(cases)}
    weights = np.asarray([1.0 / counts[str(case)] for case in cases], dtype=np.float64)
    selected = probabilities >= threshold
    positive = labels == 1
    tp = float(weights[selected & positive].sum())
    fp = float(weights[selected & ~positive].sum())
    fn = float(weights[~selected & positive].sum())
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "selected_weight": tp + fp,
    }


def choose_threshold(
    labels: np.ndarray, probabilities: np.ndarray, cases: np.ndarray
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = [
        case_balanced_metrics(labels, probabilities, cases, threshold)
        for threshold in THRESHOLDS
    ]
    precision_eligible = [row for row in rows if row["precision"] >= PRECISION_MINIMUM]
    pool = precision_eligible or rows
    # Preserve as much recall as possible subject to the hard precision floor.
    selected = max(
        pool,
        key=lambda row: (
            row["recall"] if precision_eligible else row["precision"],
            row["precision"],
            -row["threshold"],
        ),
    )
    return selected, rows


def bootstrap_precision_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    cases: np.ndarray,
    threshold: float,
    samples: int,
    seed: int,
) -> list[float]:
    unique_cases = np.unique(cases)
    case_indices = {case: np.flatnonzero(cases == case) for case in unique_cases}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        tp = fp = 0.0
        for case in sampled:
            indices = case_indices[case]
            weight = 1.0 / len(indices)
            selected = probabilities[indices] >= threshold
            positive = labels[indices] == 1
            tp += weight * float(np.count_nonzero(selected & positive))
            fp += weight * float(np.count_nonzero(selected & ~positive))
        values.append(tp / max(tp + fp, 1e-12))
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def load_oof(index_path: Path, summary_path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty patch index")
    expected = {row["patch_id"]: row for row in rows}
    cases_in_index = {row["case"] for row in rows}
    if len(cases_in_index) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} cases, found {len(cases_in_index)}")
    if {row["source_name"] for row in rows} != {"source_stunet"}:
        raise ValueError("patch index is not source-STU-only")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    coverage = summary.get("coverage", {})
    if (
        coverage.get("cases") != EXPECTED_CASES
        or coverage.get("patches") != len(rows)
        or coverage.get("every_patch_exactly_once") is not True
        or summary.get("index_sha256") != sha256(index_path)
    ):
        raise ValueError("OOF summary coverage/index contract failed")

    records: list[tuple[str, int, float]] = []
    seen: set[str] = set()
    for fold in summary.get("folds", []):
        prediction = Path(str(fold.get("prediction", "")))
        if not prediction.is_file() or fold.get("prediction_sha256") != sha256(prediction):
            raise ValueError("OOF prediction provenance mismatch")
        with np.load(prediction, allow_pickle=False) as archive:
            ids = archive["patch_id"].astype(str)
            labels = archive["label"].astype(np.uint8)
            probabilities = archive["classification_probability"].astype(np.float64)
        if not (len(ids) == len(labels) == len(probabilities)):
            raise ValueError("OOF prediction arrays do not align")
        for patch_id, label, probability in zip(ids, labels, probabilities):
            if patch_id in seen or patch_id not in expected:
                raise ValueError(f"duplicate or unexpected patch: {patch_id}")
            if int(expected[patch_id]["label"]) != int(label):
                raise ValueError(f"label mismatch: {patch_id}")
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"invalid probability: {patch_id}")
            records.append((patch_id, int(label), float(probability)))
            seen.add(patch_id)
    if seen != set(expected):
        raise ValueError("OOF predictions do not cover the patch index exactly once")
    records.sort()
    labels = np.asarray([record[1] for record in records], dtype=np.uint8)
    probabilities = np.asarray([record[2] for record in records], dtype=np.float64)
    cases = np.asarray([expected[record[0]]["case"] for record in records])
    return summary, labels, probabilities, cases


def select(
    index_path: Path,
    summary_path: Path,
    bootstrap: int = 5000,
    seed: int = 20260827,
) -> dict:
    summary, labels, probabilities, cases = load_oof(index_path, summary_path)
    selected, threshold_rows = choose_threshold(labels, probabilities, cases)
    counts = {case: int(np.count_nonzero(cases == case)) for case in np.unique(cases)}
    weights = np.asarray([1.0 / counts[str(case)] for case in cases], dtype=np.float64)
    prevalence = float(np.average(labels, weights=weights))
    average_precision = float(
        average_precision_score(labels, probabilities, sample_weight=weights)
    )
    precision_ci = bootstrap_precision_interval(
        labels,
        probabilities,
        cases,
        selected["threshold"],
        bootstrap,
        seed,
    )
    checks = {
        "average_precision_lift": average_precision - prevalence >= AP_LIFT_MINIMUM,
        "precision": selected["precision"] >= PRECISION_MINIMUM,
        "precision_bootstrap_lower": precision_ci[0] >= PRECISION_LOWER_CI_MINIMUM,
        "recall": selected["recall"] >= RECALL_MINIMUM,
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": (
            "QUALIFY_STUNET_SEED_FULL_VOLUME_OOF"
            if all(checks.values())
            else "REJECT_STUNET_SEED_PATCH_GATE"
        ),
        "selection_basis": (
            "exact-707 grouped five-fold OOF; maximize recall on a predeclared "
            "0.50:0.05:0.95 threshold grid subject to case-balanced precision >=0.90"
        ),
        "deployment_mode": "seed",
        "checkpoint_filename": "checkpoint_best_classification.pt",
        "classification_threshold": selected["threshold"],
        "classification_precision": selected["precision"],
        "classification_precision_case_bootstrap_95ci": precision_ci,
        "classification_recall": selected["recall"],
        "prevalence": prevalence,
        "average_precision": average_precision,
        "average_precision_lift": average_precision - prevalence,
        "threshold_grid": threshold_rows,
        "thresholds": {
            "precision_minimum": PRECISION_MINIMUM,
            "precision_bootstrap_lower_minimum": PRECISION_LOWER_CI_MINIMUM,
            "recall_minimum": RECALL_MINIMUM,
            "average_precision_lift_minimum": AP_LIFT_MINIMUM,
        },
        "checks": checks,
        "cases": EXPECTED_CASES,
        "patches": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(np.count_nonzero(labels == 0)),
        "index": {"path": str(index_path.resolve()), "sha256": sha256(index_path)},
        "oof_summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256(summary_path),
        },
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error("at least 1000 case bootstraps are required")
    payload = select(args.index, args.summary, args.bootstrap, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
