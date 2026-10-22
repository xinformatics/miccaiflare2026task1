#!/usr/bin/env python3
"""Select whether exhaustive Focus crops justify unlabeled pseudo-labeling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODES = ("exhaustive", "gated")
THRESHOLDS = ("0.85", "0.90", "0.95", "0.97")


def number(row: dict[str, str], key: str) -> float:
    value = row[key]
    return 0.0 if value in ("", "None", "null") else float(value)


def decide(rows: list[dict[str, str]]) -> dict[str, object]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["mode"], f'{float(row["lesion_threshold"]):.2f}')
        if key in indexed:
            raise ValueError(f"duplicate teacher state: {key}")
        indexed[key] = row
    expected = {(mode, threshold) for mode in MODES for threshold in THRESHOLDS}
    if set(indexed) != expected:
        raise ValueError(
            f"teacher state mismatch: missing={sorted(expected-set(indexed))}, "
            f"extra={sorted(set(indexed)-expected)}"
        )

    comparisons = []
    for threshold in THRESHOLDS:
        exhaustive = indexed[("exhaustive", threshold)]
        gated = indexed[("gated", threshold)]
        comparisons.append(
            {
                "lesion_threshold": float(threshold),
                "detection_recall_gain": number(exhaustive, "detection_recall")
                - number(gated, "detection_recall"),
                "small_recall_100to1000_gain": number(
                    exhaustive, "small_recall_100to1000"
                )
                - number(gated, "small_recall_100to1000"),
                "component_precision_gain": number(
                    exhaustive, "component_precision"
                )
                - number(gated, "component_precision"),
                "DSC_gain": number(exhaustive, "DSC") - number(gated, "DSC"),
                "NSD_gain": number(exhaustive, "NSD") - number(gated, "NSD"),
            }
        )
    eligible = [
        row
        for row in comparisons
        if row["detection_recall_gain"] >= 0.02
        and row["small_recall_100to1000_gain"] >= 0.01
        and row["component_precision_gain"] >= -0.02
    ]
    return {
        "warning": (
            "Public-only pseudo-label teacher diagnostic; no hidden inference "
            "or submission."
        ),
        "decision_rule": (
            "retain exhaustive branch only with >=0.02 overall lesion-recall "
            "gain, >=0.01 recall gain for 100--1000 mm3 lesions, and <=0.02 "
            "component-precision loss at a predeclared threshold"
        ),
        "comparisons": comparisons,
        "decision": (
            "RETAIN_EXHAUSTIVE_FOR_UNLABELED"
            if eligible
            else "KEEP_GATED_TEACHER"
        ),
        "eligible_thresholds": [row["lesion_threshold"] for row in eligible],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.summary.open(newline="", encoding="utf-8") as stream:
        report = decide(list(csv.DictReader(stream, delimiter="\t")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
