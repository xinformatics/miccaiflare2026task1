#!/usr/bin/env python3
"""Select a bounded, distinct-family fallback queue for hidden materialization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def mean_metrics(path: Path) -> tuple[float, float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No metric rows in {path}")
    return (
        sum(float(row["Lesion_DSC"]) for row in rows) / len(rows),
        sum(float(row["Lesion_NSD"]) for row in rows) / len(rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-dsc-drop", default=0.012, type=float)
    parser.add_argument("--max-nsd-drop", default=0.018, type=float)
    parser.add_argument("--max-catastrophic-rate", default=0.10, type=float)
    parser.add_argument("--count", default=2, type=int)
    args = parser.parse_args()

    anchor_dsc, anchor_nsd = mean_metrics(args.anchor)
    with args.summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    eligible = []
    for row in rows:
        robust_path = Path(row["robust"])
        if not robust_path.is_file():
            continue
        robust = json.loads(robust_path.read_text())
        dsc = float(row["DSC"])
        nsd = float(row["NSD"])
        catastrophic = float(robust["catastrophic_regression_rate"])
        if (
            dsc >= anchor_dsc - args.max_dsc_drop
            and nsd >= anchor_nsd - args.max_nsd_drop
            and catastrophic <= args.max_catastrophic_rate
        ):
            item = dict(row)
            item["anchor_dsc"] = f"{anchor_dsc:.6f}"
            item["anchor_nsd"] = f"{anchor_nsd:.6f}"
            item["dsc_drop"] = f"{anchor_dsc - dsc:.6f}"
            item["nsd_drop"] = f"{anchor_nsd - nsd:.6f}"
            item["catastrophic_rate"] = f"{catastrophic:.6f}"
            eligible.append(item)
    eligible.sort(key=lambda row: (float(row["DSC"]), float(row["combined"])), reverse=True)

    selected = []
    seen_families = set()
    for row in eligible:
        family = row["variant"]
        if family in seen_families:
            continue
        selected.append(row)
        seen_families.add(family)
        if len(selected) >= args.count:
            break

    fieldnames = list(rows[0]) if rows else [
        "variant", "checkpoint_tag", "checkpoint", "lesion_threshold",
        "DSC", "NSD", "combined", "metrics", "robust",
    ]
    fieldnames += ["anchor_dsc", "anchor_nsd", "dsc_drop", "nsd_drop", "catastrophic_rate"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)
    print(f"eligible={len(eligible)} selected={len(selected)} output={args.output}")


if __name__ == "__main__":
    main()
