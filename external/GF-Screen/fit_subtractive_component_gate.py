#!/usr/bin/env python3
"""Fit the final all-707 subtractive gate after nested DSC/NSD qualification."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from statistics import median

import joblib
import numpy as np
from sklearn.base import clone

from analyze_component_classifier_cv import model_grid
from subtractive_gate_features import FEATURES, feature_matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    report = json.loads(args.qualification.read_text(encoding="utf-8"))
    required_checks = {
        "qualified": report.get("decision") == "QUALIFY_GF_BASE_GATE_FOR_COMPOSITION"
        and report.get("passed") is True,
        "dataset": report.get("dataset") == str(args.dataset.resolve()),
        "dataset_sha": report.get("dataset_sha256") == sha256(args.dataset),
        "risk_cap": float(report.get("inner_catastrophic_cap", -1)) == 0.01,
        "trace_floor": float(report.get("minimum_deletion_mm3_exclusive", -1))
        == 2000.0,
        "official": report.get("official_full_volume_metrics_computed") is True,
        "dsc_parity": all(report.get("full_volume_table_DSC_parity", {}).values()),
        "blind": report.get("public_or_hidden_data_read") is False,
    }
    failed = [name for name, passed in required_checks.items() if not passed]
    if failed:
        raise SystemExit(f"invalid traceable-gate qualification: {failed}")

    with args.dataset.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(report["components"]):
        raise ValueError("qualification/component-row count mismatch")
    x = feature_matrix(rows)
    y = np.asarray([1 - int(row["is_tp"]) for row in rows], dtype=np.int64)
    names = {str(record["model"]) for record in report["outer_folds"]}
    if len(names) != 1:
        raise ValueError(f"outer folds disagree on model family: {sorted(names)}")
    model_name = names.pop()
    choices = dict(model_grid(args.seed))
    if model_name not in choices:
        raise ValueError(f"unknown qualified model family: {model_name}")
    model = clone(choices[model_name]).fit(x, y)
    thresholds = [float(record["threshold"]) for record in report["outer_folds"]]
    threshold = float(median(thresholds))

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "gate.joblib"
    temporary = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    bundle = {
        "schema_version": 1,
        "model": model,
        "model_family": model_name,
        "features": FEATURES,
        "threshold": threshold,
        "minimum_component_mm3": float(report["min_mm3"]),
        "minimum_deletion_mm3_exclusive": float(
            report["minimum_deletion_mm3_exclusive"]
        ),
        "seed": args.seed,
    }
    joblib.dump(bundle, temporary)
    os.replace(temporary, model_path)
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "model": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "model_family": model_name,
        "features": FEATURES,
        "threshold_rule": "median of five nested outer-fold selected thresholds",
        "outer_fold_thresholds": thresholds,
        "threshold": threshold,
        "minimum_component_mm3": bundle["minimum_component_mm3"],
        "minimum_deletion_mm3_exclusive": bundle[
            "minimum_deletion_mm3_exclusive"
        ],
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "qualification": str(args.qualification.resolve()),
        "qualification_sha256": sha256(args.qualification),
        "rows": len(rows),
        "positive_false_components": int(y.sum()),
        "labels_read": True,
        "labels_role": "all-707 final fit after nested held-out qualification",
        "public_or_hidden_data_read": False,
        "submission_created": False,
    }
    atomic_json(args.output / "FIT_COMPLETE.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
