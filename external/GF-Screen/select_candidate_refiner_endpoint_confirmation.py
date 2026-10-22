#!/usr/bin/env python3
"""Trigger and select the frozen exact-707 contour endpoint confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


METRICS = ("Lesion_DSC", "Lesion_NSD")
EXPECTED_THRESHOLDS = (0.95, 0.96, 0.97, 0.98)
EXPECTED_TRIGGER = {
    "anchor_currentgf_selected_segmentation_threshold": 0.95,
    "minimum_anchor_DSC_gain_0.90_to_0.95": 0.005,
    "minimum_anchor_NSD_gain_0.90_to_0.95": 0.0,
}
EXPECTED_CRITERIA = {
    "minimum_mean_DSC_gain": 0.002,
    "minimum_mean_NSD_gain": 0.0,
    "minimum_DSC_paired_bootstrap_95ci_lower": 0.0,
    "minimum_NSD_paired_bootstrap_95ci_lower": -0.001,
    "minimum_combined_case_win_rate": 0.52,
    "maximum_DSC_loss_gt_0.05_rate": 0.05,
    "minimum_per_cohort_DSC_delta": -0.005,
    "minimum_per_cohort_NSD_delta": -0.005,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("status") != "FROZEN_BEFORE_ANCHOR_RESULT":
        raise ValueError("endpoint plan was not frozen before the anchor result")
    thresholds = tuple(float(value) for value in plan["confirmation_thresholds"])
    if thresholds != EXPECTED_THRESHOLDS:
        raise ValueError(f"confirmation threshold mismatch: {thresholds}")
    if not plan.get("anchor_deployment_only"):
        raise ValueError("endpoint confirmation must be anchor-only")
    if plan.get("trigger") != EXPECTED_TRIGGER:
        raise ValueError("endpoint trigger differs from the frozen specification")
    if plan.get("promotion_vs_0.95") != EXPECTED_CRITERIA:
        raise ValueError("endpoint promotion criteria differ from the frozen specification")
    if plan.get("public_read") or plan.get("hidden_read"):
        raise ValueError("endpoint plan is not label-internal")
    return plan


def read_sweep(path: Path) -> dict[float, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    by_threshold = {round(float(row["segmentation_threshold"]), 2): row for row in rows}
    if len(by_threshold) != len(rows):
        raise ValueError("duplicate segmentation threshold in anchor sweep")
    return by_threshold


def trigger_payload(plan: dict, selection: dict, sweep: dict[float, dict[str, str]]) -> dict:
    trigger = plan["trigger"]
    selected = float(selection["selected_segmentation_threshold"])
    for threshold in (0.90, 0.95):
        if threshold not in sweep:
            raise ValueError(f"anchor sweep lacks threshold {threshold:.2f}")
    dsc_gain = float(sweep[0.95]["DSC"]) - float(sweep[0.90]["DSC"])
    nsd_gain = float(sweep[0.95]["NSD"]) - float(sweep[0.90]["NSD"])
    checks = {
        "anchor_selected_0p95": abs(
            selected - float(trigger["anchor_currentgf_selected_segmentation_threshold"])
        ) < 1e-8,
        "anchor_dsc_gain": dsc_gain
        >= float(trigger["minimum_anchor_DSC_gain_0.90_to_0.95"]),
        "anchor_nsd_nonregression": nsd_gain
        >= float(trigger["minimum_anchor_NSD_gain_0.90_to_0.95"]),
    }
    return {
        "decision": "RUN_ENDPOINT_CONFIRMATION" if all(checks.values()) else "SKIP_ENDPOINT_CONFIRMATION",
        "selected_anchor_threshold": selected,
        "anchor_gain_0p90_to_0p95": {"Lesion_DSC": dsc_gain, "Lesion_NSD": nsd_gain},
        "checks": checks,
    }


def read_metrics(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["Name"] for row in rows]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"empty or duplicate cases in {path}")
    values = np.asarray(
        [[float(row[metric]) for metric in METRICS] for row in rows], dtype=np.float64
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"invalid metrics in {path}")
    return names, values


def cohort_map(manifest: dict) -> dict[str, str]:
    result = {str(item["name"]): str(item["cohort"]) for item in manifest["items"]}
    if len(result) != int(manifest["cases"]):
        raise ValueError("manifest case mapping is incomplete or duplicated")
    return result


def stratified_bootstrap_interval(
    delta: np.ndarray,
    cohorts: np.ndarray,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = np.zeros((samples, delta.shape[1]), dtype=np.float64)
    for cohort in sorted(set(cohorts.tolist())):
        indices = np.flatnonzero(cohorts == cohort)
        drawn = rng.choice(indices, size=(samples, len(indices)), replace=True)
        means += delta[drawn].sum(axis=1) / len(delta)
    return np.quantile(means, (0.025, 0.975), axis=0)


def compare_candidate(
    anchor: np.ndarray,
    candidate: np.ndarray,
    cohorts: np.ndarray,
    criteria: dict,
    bootstrap: int,
    seed: int,
) -> dict:
    delta = candidate - anchor
    mean_gain = delta.mean(axis=0)
    interval = stratified_bootstrap_interval(delta, cohorts, bootstrap, seed)
    combined = delta.mean(axis=1)
    cohort_deltas = {
        cohort: dict(zip(METRICS, delta[cohorts == cohort].mean(axis=0).tolist()))
        for cohort in sorted(set(cohorts.tolist()))
    }
    win_rate = float(np.mean(combined > 0))
    tie_rate = float(np.mean(combined == 0))
    dsc_large_loss_rate = float(np.mean(delta[:, 0] < -0.05))
    checks = {
        "mean_dsc_gain": bool(mean_gain[0] >= float(criteria["minimum_mean_DSC_gain"])),
        "mean_nsd_gain": bool(mean_gain[1] >= float(criteria["minimum_mean_NSD_gain"])),
        "dsc_bootstrap_lower": bool(
            interval[0, 0] > float(criteria["minimum_DSC_paired_bootstrap_95ci_lower"])
        ),
        "nsd_bootstrap_lower": bool(
            interval[0, 1] >= float(criteria["minimum_NSD_paired_bootstrap_95ci_lower"])
        ),
        "combined_case_win_rate": bool(
            win_rate >= float(criteria["minimum_combined_case_win_rate"])
        ),
        "dsc_large_loss_rate": bool(
            dsc_large_loss_rate <= float(criteria["maximum_DSC_loss_gt_0.05_rate"])
        ),
        "per_cohort_dsc": all(
            values["Lesion_DSC"] >= float(criteria["minimum_per_cohort_DSC_delta"])
            for values in cohort_deltas.values()
        ),
        "per_cohort_nsd": all(
            values["Lesion_NSD"] >= float(criteria["minimum_per_cohort_NSD_delta"])
            for values in cohort_deltas.values()
        ),
    }
    return {
        "anchor_mean": dict(zip(METRICS, anchor.mean(axis=0).tolist())),
        "candidate_mean": dict(zip(METRICS, candidate.mean(axis=0).tolist())),
        "paired_gain": dict(zip(METRICS, mean_gain.tolist())),
        "cohort_stratified_bootstrap_95ci": {
            metric: [float(interval[0, index]), float(interval[1, index])]
            for index, metric in enumerate(METRICS)
        },
        "combined_case_win_rate": win_rate,
        "combined_case_tie_rate": tie_rate,
        "dsc_loss_gt_0p05_rate": dsc_large_loss_rate,
        "per_cohort_paired_gain": cohort_deltas,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def selection_payload(
    plan: dict,
    manifest: dict,
    metric_paths: dict[float, Path],
    bootstrap: int = 20000,
    seed: int = 20260824,
) -> dict:
    if tuple(sorted(metric_paths)) != EXPECTED_THRESHOLDS:
        raise ValueError("endpoint metric thresholds do not match frozen plan")
    mapping = cohort_map(manifest)
    loaded = {threshold: read_metrics(path) for threshold, path in metric_paths.items()}
    anchor_names, anchor_values = loaded[0.95]
    if len(anchor_names) != int(manifest["cases"]):
        raise ValueError("endpoint metrics do not cover exact manifest")
    if set(anchor_names) != set(mapping):
        raise ValueError("endpoint metric/manifest case mismatch")
    cohorts = np.asarray([mapping[name] for name in anchor_names])
    reports = {}
    for threshold in EXPECTED_THRESHOLDS[1:]:
        names, values = loaded[threshold]
        if set(names) != set(anchor_names):
            raise ValueError(f"case mismatch at endpoint threshold {threshold:.2f}")
        aligned = values[[names.index(name) for name in anchor_names]]
        reports[f"{threshold:.2f}"] = compare_candidate(
            anchor_values,
            aligned,
            cohorts,
            plan["promotion_vs_0.95"],
            bootstrap,
            seed + int(round(threshold * 100)),
        )
    eligible = [
        threshold for threshold in EXPECTED_THRESHOLDS[1:]
        if reports[f"{threshold:.2f}"]["eligible"]
    ]
    selected = max(
        eligible,
        key=lambda threshold: (
            reports[f"{threshold:.2f}"]["candidate_mean"]["Lesion_DSC"],
            reports[f"{threshold:.2f}"]["candidate_mean"]["Lesion_NSD"],
            -abs(threshold - 0.95),
        ),
        default=0.95,
    )
    return {
        "decision": "PROMOTE_ENDPOINT" if selected != 0.95 else "KEEP_0P95",
        "selected_segmentation_threshold": selected,
        "selection_basis": "exact-707 case-held-out anchor composition; cohort-stratified paired bootstrap",
        "anchor_threshold": 0.95,
        "candidate_reports": reports,
        "criteria": plan["promotion_vs_0.95"],
        "bootstrap_samples": bootstrap,
        "bootstrap_seed": seed,
        "cases": len(anchor_names),
    }


def common_payload(plan_path: Path, plan: dict) -> dict:
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256(plan_path),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "upload_authorized": False,
    }


def parse_metric_arguments(values: list[str]) -> dict[float, Path]:
    result = {}
    for value in values:
        threshold, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"invalid --metrics value: {value}")
        key = round(float(threshold), 2)
        if key in result:
            raise ValueError(f"duplicate metric threshold: {key}")
        result[key] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    trigger = subparsers.add_parser("trigger")
    trigger.add_argument("--plan", type=Path, required=True)
    trigger.add_argument("--selection", type=Path, required=True)
    trigger.add_argument("--sweep", type=Path, required=True)
    trigger.add_argument("--output", type=Path, required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--plan", type=Path, required=True)
    select.add_argument("--trigger", type=Path, required=True)
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--metrics", action="append", required=True)
    select.add_argument("--bootstrap", type=int, default=20000)
    select.add_argument("--seed", type=int, default=20260824)
    select.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = load_plan(args.plan)
    payload = common_payload(args.plan, plan)
    if args.command == "trigger":
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        payload.update(trigger_payload(plan, selection, read_sweep(args.sweep)))
        payload.update({
            "selection": str(args.selection.resolve()),
            "selection_sha256": sha256(args.selection),
            "sweep": str(args.sweep.resolve()),
            "sweep_sha256": sha256(args.sweep),
        })
    else:
        trigger_payload_saved = json.loads(args.trigger.read_text(encoding="utf-8"))
        if trigger_payload_saved.get("decision") != "RUN_ENDPOINT_CONFIRMATION":
            raise ValueError("endpoint selection invoked without a passing trigger")
        if args.bootstrap < 1000:
            parser.error("bootstrap must be at least 1000")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        metric_paths = parse_metric_arguments(args.metrics)
        for path in metric_paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        payload.update(
            selection_payload(plan, manifest, metric_paths, args.bootstrap, args.seed)
        )
        payload.update({
            "trigger": str(args.trigger.resolve()),
            "trigger_sha256": sha256(args.trigger),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "metrics": {f"{key:.2f}": str(path.resolve()) for key, path in metric_paths.items()},
            "metrics_sha256": {f"{key:.2f}": sha256(path) for key, path in metric_paths.items()},
        })
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
