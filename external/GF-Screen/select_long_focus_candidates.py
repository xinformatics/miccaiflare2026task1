#!/usr/bin/env python3
"""Select two structurally distinct Focus recipes for longer continuation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FAMILIES = {
    "gf_context24_dual_20260820": "context24",
    "gf_component_uniform_dual_20260820": "component_uniform",
    "gf_component_box_uniform_dual_20260820": "component_box_uniform",
    "gf_bbox_pseudo_positive_dual_20260820": "bbox_pseudo",
    "gf_cohort_balance_dual_20260820": "cohort_balance",
    "gf_remote_pseudo_positive_dual_20260820": "remote_pseudo",
}


def metrics(path: Path) -> dict[str, np.ndarray | list[str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        "names": [row["Name"] for row in rows],
        "dsc": np.asarray([float(row["Lesion_DSC"]) for row in rows]),
        "nsd": np.asarray([float(row["Lesion_NSD"]) for row in rows]),
    }


def paired_guard(anchor_path: Path, candidate_path: Path) -> dict[str, float]:
    anchor = metrics(anchor_path)
    candidate = metrics(candidate_path)
    if anchor["names"] != candidate["names"]:
        raise ValueError(f"case order mismatch: {candidate_path}")
    delta = np.column_stack((candidate["dsc"] - anchor["dsc"], candidate["nsd"] - anchor["nsd"]))
    combined = delta.mean(axis=1)
    return {
        "dsc_gain": float(delta[:, 0].mean()),
        "nsd_gain": float(delta[:, 1].mean()),
        "combined_gain": float(combined.mean()),
        "win_rate": float(np.mean(combined > 0)),
        "catastrophic_regression_rate": float(np.mean(combined < -0.20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=Path, default=Path("public_sweeps"))
    parser.add_argument(
        "--anchor",
        type=Path,
        default=Path("public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("public_sweeps/long_focus_selection_20260820"))
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()

    anchor = metrics(args.anchor)
    anchor_dsc = float(anchor["dsc"].mean())
    anchor_combined = float((anchor["dsc"].mean() + anchor["nsd"].mean()) / 2)
    candidates = []
    for root_name, family in FAMILIES.items():
        root = args.sweeps / root_name
        summary = root / "summary.tsv"
        if not summary.exists():
            continue
        with summary.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        best_by_variant: dict[str, dict[str, str]] = {}
        for row in rows:
            variant = row["variant"]
            if variant not in best_by_variant or float(row["combined"]) > float(best_by_variant[variant]["combined"]):
                best_by_variant[variant] = row
        for row in best_by_variant.values():
            threshold = float(row["lesion_threshold"])
            minimum = int(float(row["min_mm3"]))
            score_tag = f"{row['variant']}_{row['checkpoint_tag']}_lesion_{threshold:.2f}_mm{minimum}"
            metric_path = root / "scored" / f"{score_tag}.metrics.csv"
            if not metric_path.exists():
                raise FileNotFoundError(metric_path)
            guard = paired_guard(args.anchor, metric_path)
            item = {
                "family": family,
                **row,
                "metrics": str(metric_path),
                **{key: str(value) for key, value in guard.items()},
            }
            item["eligible"] = str(
                float(row["DSC"]) >= anchor_dsc - 0.02
                and float(row["combined"]) >= anchor_combined - 0.02
                and guard["catastrophic_regression_rate"] <= 0.10
            ).lower()
            candidates.append(item)

    if not candidates:
        raise RuntimeError("no completed Focus sweep summaries were found")
    # Select at most one arm per objective family. Prefer the strict local
    # guard, then combined score; a bounded fallback keeps compute productive
    # when every five-epoch probe is slightly below the noisy public anchor.
    family_best = []
    for family in sorted({row["family"] for row in candidates}):
        rows = [row for row in candidates if row["family"] == family]
        rows.sort(key=lambda row: (row["eligible"] == "true", float(row["combined"])), reverse=True)
        family_best.append(rows[0])
    family_best.sort(key=lambda row: (row["eligible"] == "true", float(row["combined"])), reverse=True)
    selected = family_best[: args.count]
    if len(selected) < args.count:
        raise RuntimeError(f"only {len(selected)} distinct Focus families completed")

    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(selected[0])
    with (args.output / "selected.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)
    (args.output / "selection.json").write_text(json.dumps({
        "anchor_dsc": anchor_dsc,
        "anchor_combined": anchor_combined,
        "selected": selected,
        "all_family_winners": family_best,
        "all_candidates": candidates,
    }, indent=2, sort_keys=True) + "\n")
    print((args.output / "selected.tsv").read_text(), end="")


if __name__ == "__main__":
    main()
