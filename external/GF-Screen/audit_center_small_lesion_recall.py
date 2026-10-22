#!/usr/bin/env python3
"""Audit whether dense-centerness candidates recover additional small lesions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def deployed_from_metrics(metrics: Path) -> Path:
    suffix = ".metrics.csv"
    if not metrics.name.endswith(suffix):
        raise ValueError(f"unexpected metrics filename: {metrics}")
    threshold_tag = metrics.name[: -len(suffix)]
    return metrics.parent / threshold_tag / "deployed_mm1000_new2k_s25"


def raw_from_metrics(metrics: Path) -> Path:
    suffix = ".metrics.csv"
    if not metrics.name.endswith(suffix):
        raise ValueError(f"unexpected metrics filename: {metrics}")
    threshold_tag = metrics.name[: -len(suffix)]
    return metrics.parent / threshold_tag / "inference"


def run_audit(audit_script, labels, predictions, audit_dir):
    members = sorted(predictions.glob("*.nii.gz"))
    if len(members) != 50:
        raise ValueError(
            f"expected 50 public predictions in {predictions}, found {len(members)}"
        )
    audit_summary = audit_dir / "summary.json"
    if not audit_summary.is_file():
        subprocess.run(
            [
                sys.executable,
                str(audit_script),
                "--labels",
                str(labels),
                "--predictions",
                str(predictions),
                "--output",
                str(audit_dir),
            ],
            check=True,
        )
    return audit_summary, json.loads(audit_summary.read_text(encoding="utf-8"))


def small_lesion_decision(anchor: dict, candidate: dict) -> dict[str, object]:
    bins = ("100to500", "500to1000")
    anchor_detected = sum(anchor["gt_by_size_mm3"][name]["detected"] for name in bins)
    candidate_detected = sum(
        candidate["gt_by_size_mm3"][name]["detected"] for name in bins
    )
    no_bin_regression = all(
        candidate["gt_by_size_mm3"][name]["detected"]
        >= anchor["gt_by_size_mm3"][name]["detected"]
        for name in bins
    )
    anchor_precision = float(anchor["components"]["precision"])
    candidate_precision = float(candidate["components"]["precision"])
    precision_guard = candidate_precision >= anchor_precision - 0.03
    qualified = (
        candidate_detected >= anchor_detected + 1
        and no_bin_regression
        and precision_guard
    )
    return {
        "anchor_detected_100to1000": anchor_detected,
        "candidate_detected_100to1000": candidate_detected,
        "small_detected_gain": candidate_detected - anchor_detected,
        "no_bin_regression": no_bin_regression,
        "anchor_component_precision": anchor_precision,
        "candidate_component_precision": candidate_precision,
        "component_precision_guard": precision_guard,
        "qualified": qualified,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--anchor-audit", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    anchor = json.loads(args.anchor_audit.read_text(encoding="utf-8"))
    with args.summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty candidate summary: {args.summary}")
    args.output.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in rows:
        metrics = Path(row["metrics"])
        predictions = deployed_from_metrics(metrics)
        raw_predictions = raw_from_metrics(metrics)
        glance_tag = str(row["glance_threshold"]).replace(".", "p")
        center_tag = str(row.get("center_threshold", "off")).replace(".", "p")
        audit_root = (
            args.output
            / row["variant"]
            / row["checkpoint_tag"]
            / f"g{glance_tag}_c{center_tag}"
        )
        audit_summary, candidate = run_audit(
            args.audit_script, args.labels, predictions, audit_root / "deployed"
        )
        raw_audit_summary, raw_candidate = run_audit(
            args.audit_script, args.labels, raw_predictions, audit_root / "raw"
        )
        decision = small_lesion_decision(anchor, candidate)
        raw_decision = small_lesion_decision(anchor, raw_candidate)
        output_rows.append(
            {
                **row,
                "audit_summary": str(audit_summary.resolve()),
                "detected_100to1000": decision["candidate_detected_100to1000"],
                "component_precision": decision["candidate_component_precision"],
                "small_detected_gain": decision["small_detected_gain"],
                "no_bin_regression": str(decision["no_bin_regression"]).lower(),
                "component_precision_guard": str(
                    decision["component_precision_guard"]
                ).lower(),
                "qualified": str(decision["qualified"]).lower(),
                "raw_audit_summary": str(raw_audit_summary.resolve()),
                "raw_detected_100to1000": raw_decision[
                    "candidate_detected_100to1000"
                ],
                "raw_component_precision": raw_decision[
                    "candidate_component_precision"
                ],
                "raw_small_detected_gain": raw_decision["small_detected_gain"],
            }
        )
    fieldnames = list(output_rows[0])
    gate_path = args.output / "small_lesion_gate.tsv"
    temporary = gate_path.with_suffix(".tmp.tsv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(gate_path)
    (args.output / "COMPLETE.txt").write_text("complete\n", encoding="utf-8")
    print(gate_path)


if __name__ == "__main__":
    main()
