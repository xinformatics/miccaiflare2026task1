#!/usr/bin/env python3
"""Select one Focus and one Glance checkpoint for a cross-module merge."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(path: Path, threshold_column: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty summary: {path}")

    parsed = []
    for row in rows:
        checkpoint = Path(row["checkpoint"])
        metrics = Path(row["metrics"])
        robust_path = Path(row["robust"])
        for required in (checkpoint, metrics, robust_path):
            if not required.is_file() or required.stat().st_size == 0:
                raise FileNotFoundError(required)
        robust = json.loads(robust_path.read_text(encoding="utf-8"))
        if int(robust.get("cases", 0)) != 50:
            raise ValueError(f"robustness report does not cover 50 cases: {robust_path}")
        parsed.append(
            {
                "variant": row["variant"],
                "checkpoint_tag": row["checkpoint_tag"],
                "checkpoint": str(checkpoint.resolve()),
                "threshold": float(row[threshold_column]),
                "DSC": float(row["DSC"]),
                "NSD": float(row["NSD"]),
                "combined": float(row["combined"]),
                "metrics": str(metrics.resolve()),
                "robust": str(robust_path.resolve()),
                "decision": robust["decision"],
                "anchor_DSC": float(robust["anchor_mean"]["Lesion_DSC"]),
                "anchor_NSD": float(robust["anchor_mean"]["Lesion_NSD"]),
                "win_rate": float(robust["win_rate"]),
                "catastrophic_regression_rate": float(
                    robust["catastrophic_regression_rate"]
                ),
            }
        )
    return parsed


def select(rows: list[dict], nsd_tolerance: float) -> tuple[dict, str]:
    eligible = [row for row in rows if row["NSD"] >= row["anchor_NSD"] - nsd_tolerance]
    if eligible:
        pool = eligible
        status = "nsd_guard_pass"
    else:
        pool = rows
        status = "fallback_best_combined"
    winner = max(pool, key=lambda row: (row["DSC"], row["NSD"], row["combined"]))
    return winner, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-summary", type=Path, required=True)
    parser.add_argument("--glance-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nsd-tolerance", type=float, default=0.002)
    args = parser.parse_args()

    focus, focus_status = select(
        load_rows(args.focus_summary, "lesion_threshold"), args.nsd_tolerance
    )
    glance, glance_status = select(
        load_rows(args.glance_summary, "glance_threshold"), args.nsd_tolerance
    )
    report = {
        "selection_rule": (
            "maximize DSC among candidates no more than nsd_tolerance below the "
            "paired champion NSD; break ties by NSD then combined"
        ),
        "nsd_tolerance": args.nsd_tolerance,
        "focus_selection_status": focus_status,
        "glance_selection_status": glance_status,
        "focus": focus,
        "glance": glance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
