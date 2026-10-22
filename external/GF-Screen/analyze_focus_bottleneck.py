#!/usr/bin/env python3
"""Summarize GT-proposal and all-crop Focus diagnostics against the anchor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    if not rows:
        raise ValueError(f"no metric rows in {path}")
    return rows


def metric_row(row: dict[str, str]) -> dict[str, object]:
    result: dict[str, object] = dict(row)
    for key in ("DSC", "NSD", "combined"):
        if key in result:
            result[key] = float(result[key])
    return result


def select(rows: list[dict[str, str]], anchor_nsd: float) -> dict[str, object]:
    parsed = [metric_row(row) for row in rows]
    best_dsc = max(parsed, key=lambda row: (row["DSC"], row["NSD"]))
    best_combined = max(parsed, key=lambda row: (row["combined"], row["DSC"]))
    nsd_safe = [row for row in parsed if row["NSD"] >= anchor_nsd]
    best_nsd_safe = (
        max(nsd_safe, key=lambda row: (row["DSC"], row["NSD"]))
        if nsd_safe
        else None
    )
    return {
        "best_dsc": best_dsc,
        "best_combined": best_combined,
        "best_dsc_with_anchor_nsd_preserved": best_nsd_safe,
    }


def classify_route(oracle_gain: float, full_gain: float | None) -> str:
    """Coarse diagnostic route; thresholds are predeclared practical effects."""
    if oracle_gain < 0.03:
        return "focus_segmentation_limited"
    if full_gain is None:
        return "await_full_coverage"
    if oracle_gain >= 0.08 and full_gain < 0.03:
        return "proposal_quality_limited"
    if oracle_gain >= 0.08 and full_gain >= 0.03:
        return "glance_recall_limited"
    return "mixed_glance_focus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--full-coverage", type=Path)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    anchor_rows = read_rows(args.anchor, delimiter=",")
    anchor_dsc = sum(float(row["Lesion_DSC"]) for row in anchor_rows) / len(anchor_rows)
    anchor_nsd = sum(float(row["Lesion_NSD"]) for row in anchor_rows) / len(anchor_rows)
    oracle = select(read_rows(args.oracle), anchor_nsd)
    oracle_safe = oracle["best_dsc_with_anchor_nsd_preserved"]
    oracle_reference = oracle_safe or oracle["best_dsc"]
    oracle_gain = float(oracle_reference["DSC"]) - anchor_dsc

    full = None
    full_gain = None
    if args.full_coverage and args.full_coverage.is_file():
        full = select(read_rows(args.full_coverage), anchor_nsd)
        full_safe = full["best_dsc_with_anchor_nsd_preserved"]
        full_reference = full_safe or full["best_dsc"]
        full_gain = float(full_reference["DSC"]) - anchor_dsc

    report = {
        "anchor": {"cases": len(anchor_rows), "DSC": anchor_dsc, "NSD": anchor_nsd},
        "oracle": oracle,
        "oracle_nsd_safe_dsc_gain": oracle_gain,
        "full_coverage": full,
        "full_coverage_nsd_safe_dsc_gain": full_gain,
        "route": classify_route(oracle_gain, full_gain),
        "route_thresholds": {
            "material_headroom": 0.03,
            "large_headroom": 0.08,
            "nsd_must_be_at_least_anchor": True,
        },
        "warning": "GT-proposal oracle is label-informed diagnostic output and must never be submitted.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
