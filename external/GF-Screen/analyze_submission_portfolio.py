#!/usr/bin/env python3
"""Audit case-level diversity among already-scored public candidates.

The case-wise maximum is an oracle diagnostic, not an ensemble score.  It is
useful for identifying candidates that make different errors when a fixed
number of hidden-evaluation slots is available.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import fmean


METRICS = ("Lesion_DSC", "Lesion_NSD")


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=METRICS.csv")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=METRICS.csv")
    return name, Path(path).resolve()


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"Name", *METRICS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        rows: dict[str, dict[str, float]] = {}
        for row in reader:
            name = row["Name"].strip()
            if not name or name in rows:
                raise ValueError(f"empty or duplicate case name in {path}: {name!r}")
            values = {metric: float(row[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"non-finite metric for {name} in {path}")
            rows[name] = values
    if not rows:
        raise ValueError(f"no cases in {path}")
    return rows


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("correlation vectors must be nonempty and equal length")
    left_mean, right_mean = fmean(left), fmean(right)
    left_sum = sum((value - left_mean) ** 2 for value in left)
    right_sum = sum((value - right_mean) ** 2 for value in right)
    if left_sum == 0 or right_sum == 0:
        return None
    covariance = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    return covariance / math.sqrt(left_sum * right_sum)


def analyze(
    anchor_name: str,
    anchor: dict[str, dict[str, float]],
    candidates: dict[str, dict[str, dict[str, float]]],
    slots: int,
) -> dict[str, object]:
    if slots < 1 or slots > len(candidates):
        raise ValueError("slots must be between one and the candidate count")
    expected = set(anchor)
    for name, rows in candidates.items():
        if set(rows) != expected:
            raise ValueError(
                f"case mismatch for {name}: "
                f"missing={sorted(expected-set(rows))[:3]} "
                f"extra={sorted(set(rows)-expected)[:3]}"
            )
    cases = sorted(expected)
    means = {
        name: {
            metric: fmean(rows[case][metric] for case in cases)
            for metric in METRICS
        }
        for name, rows in candidates.items()
    }
    pairwise = []
    for left, right in itertools.combinations(sorted(candidates), 2):
        row: dict[str, object] = {"left": left, "right": right}
        for metric in METRICS:
            left_delta = [
                candidates[left][case][metric] - anchor[case][metric] for case in cases
            ]
            right_delta = [
                candidates[right][case][metric] - anchor[case][metric] for case in cases
            ]
            row[f"delta_{metric}_correlation"] = pearson(left_delta, right_delta)
        pairwise.append(row)
    portfolios = []
    for names in itertools.combinations(sorted(candidates), slots):
        row = {"candidates": list(names)}
        for metric in METRICS:
            row[f"casewise_max_{metric}"] = fmean(
                max(candidates[name][case][metric] for name in names) for case in cases
            )
        portfolios.append(row)
    portfolios.sort(
        key=lambda row: (
            row["casewise_max_Lesion_DSC"], row["casewise_max_Lesion_NSD"]
        ),
        reverse=True,
    )
    return {
        "warning": (
            "Public case-wise maxima are oracle diversity diagnostics, not "
            "deployable ensemble scores or hidden-score predictions."
        ),
        "anchor": anchor_name,
        "cases": len(cases),
        "slots": slots,
        "candidate_means": means,
        "pairwise_delta_correlations": pairwise,
        "portfolios": portfolios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=named_path, required=True)
    parser.add_argument("--candidate", type=named_path, action="append", required=True)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    args = parser.parse_args()
    anchor_name, anchor_path = args.anchor
    candidate_paths = dict(args.candidate)
    if len(candidate_paths) != len(args.candidate):
        raise ValueError("candidate names must be unique")
    report = analyze(
        anchor_name,
        load_metrics(anchor_path),
        {name: load_metrics(path) for name, path in candidate_paths.items()},
        args.slots,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.tsv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.tsv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            ("rank", "candidates", "casewise_max_Lesion_DSC", "casewise_max_Lesion_NSD")
        )
        for rank, row in enumerate(report["portfolios"], 1):
            writer.writerow(
                (
                    rank,
                    "+".join(row["candidates"]),
                    row["casewise_max_Lesion_DSC"],
                    row["casewise_max_Lesion_NSD"],
                )
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
