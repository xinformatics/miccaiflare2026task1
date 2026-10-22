#!/usr/bin/env python3
"""Summarize exact-707 component errors by source cohort after OOF.

This consumes the label-informed output of ``analyze_lesion_errors.py`` only
after deployment calibration is frozen.  It is diagnostic and cannot
authorize public/hidden inference or submission creation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean


METRICS = (
    "dice",
    "precision",
    "recall",
    "lesion_detection_recall",
    "component_precision",
    "fp_free_oracle_dice",
    "fn_free_oracle_dice",
)
ZERO_WHEN_UNDEFINED = {
    "precision",
    "recall",
    "lesion_detection_recall",
    "component_precision",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_mean(rows: list[dict[str, str]], key: str) -> float:
    values = []
    for row in rows:
        raw = str(row.get(key, ""))
        if raw in ("", "None"):
            if key not in ZERO_WHEN_UNDEFINED:
                raise ValueError(f"missing cohort metric: {key}")
            values.append(0.0)
        else:
            values.append(float(raw))
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"missing or non-finite cohort metric: {key}")
    return fmean(values)


def choose_priority(
    macro: dict[str, float],
    minimum_gap_separation: float = 0.02,
) -> tuple[str, dict[str, float]]:
    if minimum_gap_separation < 0:
        raise ValueError("minimum gap separation must be nonnegative")
    fp_headroom = macro["fp_free_oracle_dice"] - macro["dice"]
    fn_headroom = macro["fn_free_oracle_dice"] - macro["dice"]
    if fp_headroom >= fn_headroom + minimum_gap_separation:
        priority = "PRIORITIZE_FALSE_POSITIVE_CONTROL"
    elif fn_headroom >= fp_headroom + minimum_gap_separation:
        priority = "PRIORITIZE_RECALL_AND_CONTOUR"
    else:
        priority = "PRIORITIZE_BALANCED_CLASSIFICATION_AND_CONTOUR"
    return priority, {
        "false_positive_removal_oracle_gain": fp_headroom,
        "false_negative_recovery_oracle_gain": fn_headroom,
        "headroom_difference_fp_minus_fn": fp_headroom - fn_headroom,
    }


def summarize(
    manifest: dict,
    rows: list[dict[str, str]],
    lesion_summary: dict,
    minimum_gap_separation: float = 0.02,
) -> dict:
    items = manifest.get("items", [])
    cohort_by_name = {str(item["prediction_name"]): str(item["cohort"]) for item in items}
    if len(cohort_by_name) != len(items) or not cohort_by_name:
        raise ValueError("manifest has missing or duplicate prediction names")
    by_name = {str(row["case"]): row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != set(cohort_by_name):
        raise ValueError("failure audit does not exactly cover the manifest")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for name, row in by_name.items():
        grouped[cohort_by_name[name]].append(row)
    cohort_summary = []
    for cohort in sorted(grouped):
        cohort_rows = grouped[cohort]
        cohort_summary.append(
            {
                "cohort": cohort,
                "cases": len(cohort_rows),
                **{metric: finite_mean(cohort_rows, metric) for metric in METRICS},
            }
        )
    macro = {metric: finite_mean(rows, metric) for metric in METRICS}
    recorded_macro = lesion_summary.get("macro", {})
    for metric in ("dice", "precision", "recall", "fp_free_oracle_dice", "fn_free_oracle_dice"):
        if not math.isclose(macro[metric], float(recorded_macro[metric]), abs_tol=1e-10):
            raise ValueError(f"case CSV/summary mismatch for {metric}")
    priority, headroom = choose_priority(macro, minimum_gap_separation)
    return {
        "cases": len(rows),
        "cohorts": len(grouped),
        "macro": macro,
        "headroom": headroom,
        "priority": priority,
        "priority_interpretation": (
            "diagnostic only; it does not authorize a model, threshold, public run, or submission"
        ),
        "cohort_summary": cohort_summary,
        "gt_by_size_mm3": lesion_summary.get("gt_by_size_mm3", {}),
        "prediction_by_size_mm3": lesion_summary.get("prediction_by_size_mm3", {}),
        "minimum_gap_separation": minimum_gap_separation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--lesion-summary", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-gap-separation", type=float, default=0.02)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with args.cases.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    lesion_summary = json.loads(args.lesion_summary.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if int(calibration.get("cases", -1)) != len(rows):
        raise ValueError("calibration/failure-audit case count mismatch")
    result = summarize(manifest, rows, lesion_summary, args.minimum_gap_separation)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "lesion_analysis_implementation": lesion_summary.get(
            "configuration", {}
        ).get("implementation"),
        "lesion_analysis_implementation_sha256": lesion_summary.get(
            "configuration", {}
        ).get("implementation_sha256"),
        "selection_basis": "frozen exact-707 case-held-out full-volume OOF deployment",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "cases_csv": str(args.cases.resolve()),
        "cases_csv_sha256": sha256(args.cases),
        "lesion_summary": str(args.lesion_summary.resolve()),
        "lesion_summary_sha256": sha256(args.lesion_summary),
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256(args.calibration),
        "calibration_variant": calibration.get("variant"),
        "calibration_deployment": calibration.get("calibration_deployment", "standalone"),
        **result,
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
