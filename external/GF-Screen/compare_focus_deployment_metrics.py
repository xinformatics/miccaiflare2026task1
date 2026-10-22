#!/usr/bin/env python3
"""Verify that a regenerated Focus deployment is numerically equivalent.

The historical public metrics are rounded to four decimals.  Re-materializing
the same connected-component recipe can change a handful of isolated voxels
after library/runtime changes without changing the measured aggregate.  This
checker keeps that tolerance explicit and very small; it is not a performance
promotion test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


MAX_CASE_METRIC_DELTA = 0.0001001
MAX_CASE_VOXEL_DELTA = 8
MAX_TOTAL_ABS_VOXEL_DELTA = 25
MAX_TOTAL_ABS_VOXEL_FRACTION = 5e-6
MAX_MEAN_METRIC_DELTA = 5e-7


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"Name", "Lesion_DSC", "Lesion_NSD", "pred_fg_voxels"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"missing deployment metric columns: {path}")
    by_name = {row["Name"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError(f"duplicate case names: {path}")
    return by_name


def compare(anchor: Path, candidate: Path) -> dict[str, object]:
    left, right = _rows(anchor), _rows(candidate)
    if set(left) != set(right):
        raise ValueError("anchor and candidate case inventories differ")

    names = sorted(left)
    different_cases: list[str] = []
    max_dsc_delta = 0.0
    max_nsd_delta = 0.0
    max_voxel_delta = 0
    total_abs_voxel_delta = 0
    anchor_voxels = 0
    anchor_dsc: list[float] = []
    candidate_dsc: list[float] = []
    anchor_nsd: list[float] = []
    candidate_nsd: list[float] = []
    exact = True
    for name in names:
        a, b = left[name], right[name]
        exact &= a == b
        if a != b:
            different_cases.append(name)
        ad, bd = float(a["Lesion_DSC"]), float(b["Lesion_DSC"])
        an, bn = float(a["Lesion_NSD"]), float(b["Lesion_NSD"])
        av, bv = int(a["pred_fg_voxels"]), int(b["pred_fg_voxels"])
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (ad, bd, an, bn)):
            raise ValueError(f"invalid metric value for {name}")
        max_dsc_delta = max(max_dsc_delta, abs(ad - bd))
        max_nsd_delta = max(max_nsd_delta, abs(an - bn))
        max_voxel_delta = max(max_voxel_delta, abs(av - bv))
        total_abs_voxel_delta += abs(av - bv)
        anchor_voxels += av
        anchor_dsc.append(ad); candidate_dsc.append(bd)
        anchor_nsd.append(an); candidate_nsd.append(bn)

    mean_dsc_delta = abs(sum(anchor_dsc) / len(names) - sum(candidate_dsc) / len(names))
    mean_nsd_delta = abs(sum(anchor_nsd) / len(names) - sum(candidate_nsd) / len(names))
    voxel_fraction = total_abs_voxel_delta / max(anchor_voxels, 1)
    equivalent = (
        max_dsc_delta <= MAX_CASE_METRIC_DELTA
        and max_nsd_delta <= MAX_CASE_METRIC_DELTA
        and max_voxel_delta <= MAX_CASE_VOXEL_DELTA
        and total_abs_voxel_delta <= MAX_TOTAL_ABS_VOXEL_DELTA
        and voxel_fraction <= MAX_TOTAL_ABS_VOXEL_FRACTION
        and mean_dsc_delta <= MAX_MEAN_METRIC_DELTA
        and mean_nsd_delta <= MAX_MEAN_METRIC_DELTA
    )
    return {
        "exact_metric_rows": exact,
        "deployment_equivalent": equivalent,
        "cases": len(names),
        "different_cases": different_cases,
        "max_case_dsc_delta": max_dsc_delta,
        "max_case_nsd_delta": max_nsd_delta,
        "max_case_voxel_delta": max_voxel_delta,
        "total_abs_voxel_delta": total_abs_voxel_delta,
        "total_abs_voxel_fraction": voxel_fraction,
        "mean_dsc_delta": mean_dsc_delta,
        "mean_nsd_delta": mean_nsd_delta,
        "limits": {
            "max_case_metric_delta": MAX_CASE_METRIC_DELTA,
            "max_case_voxel_delta": MAX_CASE_VOXEL_DELTA,
            "max_total_abs_voxel_delta": MAX_TOTAL_ABS_VOXEL_DELTA,
            "max_total_abs_voxel_fraction": MAX_TOTAL_ABS_VOXEL_FRACTION,
            "max_mean_metric_delta": MAX_MEAN_METRIC_DELTA,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.anchor, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    if not payload["deployment_equivalent"]:
        raise SystemExit("deployment metrics exceed locked equivalence bounds")


if __name__ == "__main__":
    main()
