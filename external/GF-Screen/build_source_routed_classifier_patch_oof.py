#!/usr/bin/env python3
"""Build a deployable patch-OOF summary for source-routed classifiers.

Classification probabilities come from the excluded-fold, source-family epoch
selected by ``analyze_source_routed_classifier_oof.py``. Contours and coverage
are copied from the independently selected global-contour OOF summary, so this
artifact changes exactly one factor and remains public/hidden blind.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from analyze_focus_classifier_fusion_oof import case_weights, metrics
from focus_classifier_fusion import SOURCE_GROUP_NAMES, source_group
from summarize_candidate_refiner_oof import deployment_classification_thresholds


EXPECTED_SOURCES = {
    "epoch105", "currentgf", "autopet_weak", "lndb_weak",
    *(f"lung_cell{index}" for index in range(8)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pooled_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    best = None
    for threshold in np.linspace(0.05, 0.95, 19):
        selected = probabilities >= threshold
        tp = int(np.count_nonzero(selected & (labels == 1)))
        fp = int(np.count_nonzero(selected & (labels == 0)))
        fn = int(np.count_nonzero(~selected & (labels == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f0p5 = 1.25 * precision * recall / max(0.25 * precision + recall, 1e-12)
        candidate = (f0p5, precision, recall, -float(threshold))
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold), precision, recall, f0p5)
    assert best is not None
    return {"threshold": best[1], "precision": best[2], "recall": best[3], "f0p5": best[4]}


def report(rows: list[dict[str, str]], labels: np.ndarray, probabilities: np.ndarray) -> dict:
    result = dict(metrics(rows, labels, probabilities))
    weights = case_weights(rows, np.arange(len(rows), dtype=np.int64))
    result.update({
        "deployment_thresholds": deployment_classification_thresholds(float(result["threshold"])),
        "prevalence": float(np.average(labels, weights=weights)),
        "patch_pooled_threshold_metrics": pooled_threshold(labels, probabilities),
        "patch_pooled_prevalence": float(labels.mean()),
        "patch_pooled_average_precision": float(average_precision_score(labels, probabilities)),
        "patch_pooled_roc_auc": float(roc_auc_score(labels, probabilities)),
        "patches": len(rows),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
    })
    return result


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if decision.get("decision") != "QUALIFY_SOURCE_ROUTED_CLASSIFIER":
        raise ValueError("source-routed classifier patch gate did not qualify")
    if decision.get("index_sha256") != sha256(args.index):
        raise ValueError("classifier decision/index identity mismatch")
    prediction = Path(decision["predictions"])
    if sha256(prediction) != decision.get("predictions_sha256"):
        raise ValueError("classifier decision/prediction identity mismatch")
    if baseline.get("index_sha256") != sha256(args.index):
        raise ValueError("global contour summary/index identity mismatch")
    coverage = baseline.get("coverage", {})
    if coverage.get("cases") != 707 or coverage.get("every_patch_exactly_once") is not True:
        raise ValueError("global contour OOF does not have exact 707-case coverage")

    with args.index.open(newline="", encoding="utf-8") as stream:
        index_rows = list(csv.DictReader(stream))
    by_id = {str(row["patch_id"]): row for row in index_rows}
    if not index_rows or len(by_id) != len(index_rows):
        raise ValueError("invalid exact707 index")
    with np.load(prediction, allow_pickle=False) as archive:
        patch_ids = [str(value) for value in archive["patch_id"]]
        labels = archive["label"].astype(np.uint8)
        probabilities = archive["classification_probability"].astype(np.float64)
        folds = archive["fold"].astype(np.int64)
    if (
        len(patch_ids) != len(index_rows)
        or len(set(patch_ids)) != len(patch_ids)
        or set(patch_ids) != set(by_id)
        or not (len(labels) == len(probabilities) == len(folds) == len(patch_ids))
        or set(int(value) for value in folds) != set(range(5))
    ):
        raise ValueError("routed classifier prediction lacks exact five-fold index coverage")
    rows = [by_id[patch_id] for patch_id in patch_ids]
    expected_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if not np.array_equal(labels, expected_labels):
        raise ValueError("routed classifier labels disagree with exact707 index")

    sources = np.asarray([str(row["source_name"]) for row in rows])
    if set(sources) != EXPECTED_SOURCES:
        raise ValueError("source-routed classifier source inventory mismatch")
    global_report = report(rows, labels, probabilities)
    source_reports = {}
    for source in sorted(EXPECTED_SOURCES):
        selected = np.flatnonzero(sources == source)
        source_reports[source] = report(
            [rows[int(index)] for index in selected],
            labels[selected], probabilities[selected],
        )
    groups = np.asarray([source_group(source) for source in sources])
    group_reports = {}
    for group in SOURCE_GROUP_NAMES:
        selected = np.flatnonzero(groups == group)
        item = report(
            [rows[int(index)] for index in selected],
            labels[selected], probabilities[selected],
        )
        item["sources"] = sorted(set(sources[selected]))
        group_reports[group] = item

    output = copy.deepcopy(baseline)
    output["classification"] = global_report
    output["source_classification"] = source_reports
    output["source_classification_groups"] = group_reports
    output["checkpoint_selection"] = {
        "selection_basis": "exact707 grouped held-out source-family classifier epochs with global decoupled contours",
        "selected_strategy": "classifier_routed",
        "checkpoint_filename": "checkpoint_best_segmentation.pt",
        "prediction_filename": "validation_predictions_best_segmentation.npz",
        "classifier_route_decision": str(args.decision.resolve()),
        "classifier_route_decision_sha256": sha256(args.decision),
        "contour_summary": str(args.baseline.resolve()),
        "contour_summary_sha256": sha256(args.baseline),
        "classification_eligible": True,
        "public_read": False,
        "hidden_read": False,
    }
    output["source_routed_classifier"] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": str(args.decision.resolve()),
        "decision_sha256": sha256(args.decision),
        "predictions": str(prediction.resolve()),
        "predictions_sha256": sha256(prediction),
        "classification_only_change": True,
        "global_contour_unchanged": True,
        "average_precision_gain_95ci": decision.get("average_precision_gain_95ci"),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    atomic_json(args.output, output)
    print(json.dumps({
        "status": "PASS", "output": str(args.output.resolve()),
        "average_precision": global_report["average_precision"],
        "f0p5": global_report["f0p5"], "cases": coverage["cases"],
    }))


if __name__ == "__main__":
    main()
