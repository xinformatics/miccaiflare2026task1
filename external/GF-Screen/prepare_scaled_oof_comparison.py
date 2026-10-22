#!/usr/bin/env python3
"""Prepare a matched pilot-subset comparison for a scaled OOF refiner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path


METRICS = ("Lesion_DSC", "Lesion_NSD")


def parse_named(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("baseline must be NAME=CALIBRATION_JSON")
    name, path = value.split("=", 1)
    if not name or not name.replace("_", "").isalnum() or not path:
        raise ValueError(f"invalid baseline: {value!r}")
    return name, Path(path)


def resolve_metrics(calibration_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = calibration_path.parent / path
    return local if local.is_file() else path


def read_metrics(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"Name", *METRICS} <= set(rows[0]):
        raise ValueError(f"invalid metrics CSV: {path}")
    names = [str(row["Name"]) for row in rows]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"missing or duplicate metric names: {path}")
    for row in rows:
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid {metric} for {row['Name']}: {path}")
    return list(rows[0]), {str(row["Name"]): row for row in rows}


def load_calibration(path: Path, expected_cases: int) -> tuple[dict, Path, dict[str, dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("cases", -1)) != expected_cases:
        raise ValueError(f"calibration case count mismatch: {path}")
    metrics_path = resolve_metrics(path, str(payload.get("metrics", ""))).resolve(strict=True)
    _, rows = read_metrics(metrics_path)
    if len(rows) != expected_cases:
        raise ValueError(f"metrics case count mismatch: {metrics_path}")
    for metric in METRICS:
        observed = sum(float(row[metric]) for row in rows.values()) / len(rows)
        if not math.isclose(observed, float(payload[metric]), abs_tol=5e-4):
            raise ValueError(f"calibration mean mismatch for {metric}: {path}")
    return payload, metrics_path, rows


def write_subset(path: Path, fieldnames: list[str], rows: dict[str, dict[str, str]], names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[name] for name in names)


def prepare(
    subset_manifest: Path,
    scaled_calibration: Path,
    baselines: list[tuple[str, Path]],
    output: Path,
) -> dict:
    manifest = json.loads(subset_manifest.read_text(encoding="utf-8"))
    names = sorted(str(item["name"]) for item in manifest.get("items", []))
    if not names or len(names) != len(set(names)):
        raise ValueError("subset manifest must contain unique cases")
    expected = len(names)
    if not baselines or len({name for name, _ in baselines}) != len(baselines):
        raise ValueError("at least one uniquely named baseline is required")

    loaded_baselines = []
    for name, path in baselines:
        payload, metrics_path, rows = load_calibration(path, expected)
        if set(rows) != set(names):
            raise ValueError(f"baseline coverage mismatch: {name}")
        loaded_baselines.append((name, path, payload, metrics_path, rows))
    selected = max(
        loaded_baselines,
        key=lambda item: (
            float(item[2]["Lesion_DSC"]),
            float(item[2]["Lesion_NSD"]),
            item[0],
        ),
    )

    scaled_payload = json.loads(scaled_calibration.read_text(encoding="utf-8"))
    scaled_cases = int(scaled_payload.get("cases", -1))
    if scaled_cases <= expected:
        raise ValueError("scaled calibration must contain more cases than the pilot subset")
    selected_deployment = scaled_payload.get("calibration_deployment", "standalone")
    if selected_deployment not in {"standalone", "anchor_currentgf"}:
        raise ValueError(f"invalid scaled calibration deployment: {selected_deployment}")
    comparison_calibration = scaled_calibration
    if selected_deployment == "anchor_currentgf":
        value = str(scaled_payload.get("standalone_calibration", ""))
        if not value:
            raise ValueError("anchor-composed scaled calibration lacks standalone calibration")
        comparison_calibration = resolve_metrics(scaled_calibration, value).resolve(strict=True)
    comparison_payload, scaled_metrics, scaled_rows = load_calibration(
        comparison_calibration, scaled_cases
    )
    fieldnames, _ = read_metrics(scaled_metrics)
    if len(scaled_rows) != scaled_cases or not set(names) <= set(scaled_rows):
        raise ValueError("scaled metrics do not exactly cover the declared cases/subset")

    output.mkdir(parents=True, exist_ok=True)
    baseline_csv = output / "baseline_selected.metrics.csv"
    scaled_subset_csv = output / "scaled_subset.metrics.csv"
    baseline_fields, baseline_rows = read_metrics(selected[3])
    write_subset(baseline_csv, baseline_fields, baseline_rows, names)
    write_subset(scaled_subset_csv, fieldnames, scaled_rows, names)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "strongest complete exact-pilot OOF baseline by DSC then NSD",
        "subset_cases": expected,
        "scaled_cases": scaled_cases,
        "selected_baseline": selected[0],
        "selected_baseline_calibration": str(selected[1].resolve()),
        "selected_baseline_DSC": float(selected[2]["Lesion_DSC"]),
        "selected_baseline_NSD": float(selected[2]["Lesion_NSD"]),
        "baseline_metrics": str(baseline_csv.resolve()),
        "scaled_subset_metrics": str(scaled_subset_csv.resolve()),
        "scaled_calibration": str(scaled_calibration.resolve()),
        "scaled_selected_deployment": selected_deployment,
        "scaled_comparison_deployment": "standalone",
        "scaled_comparison_calibration": str(comparison_calibration.resolve()),
        "scaled_deployment_mode": comparison_payload.get("deployment_mode", "refined"),
        "scaled_classification_threshold": comparison_payload["classification_threshold"],
        "scaled_segmentation_threshold": comparison_payload["segmentation_threshold"],
        "subset_manifest": str(subset_manifest.resolve()),
        "subset_manifest_sha256": hashlib.sha256(subset_manifest.read_bytes()).hexdigest(),
        "public_or_hidden_data_read": False,
        "raw_labels_read": False,
    }
    (output / "PREPARED.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--scaled-calibration", type=Path, required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(
        args.subset_manifest,
        args.scaled_calibration,
        [parse_named(value) for value in args.baseline],
        args.output,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
