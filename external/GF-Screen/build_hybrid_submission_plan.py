#!/usr/bin/env python3
"""Build a diverse, strictly promoted three-candidate hybrid save plan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "rank",
    "tag",
    "family",
    "checkpoint",
    "glance_threshold",
    "lesion_threshold",
    "DSC",
    "NSD",
    "metrics",
    "robust",
)


def promoted(path: Path, family: str, threshold_column: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = []
    for row in rows:
        robust = Path(row["robust"])
        report = json.loads(robust.read_text(encoding="utf-8"))
        if report["decision"] != "PROMOTE":
            continue
        threshold = float(row[threshold_column])
        variant = row["variant"]
        state = row.get("checkpoint_tag", "cross").replace("checkpoint_epoch_", "e")
        if state == "model":
            state = "best"
        if state == "model_final":
            state = "final"
        if family == "focus":
            arm = "f08" if "complete08" in variant else "f04"
            tag = f"gf_hyb_{arm}_{state}_l{threshold:.2f}".replace(".", "")
            glance_threshold, lesion_threshold = 0.50, threshold
        elif family == "glance":
            arm = "g1e4" if "lr1e4" in variant else "g3e5"
            tag = f"gf_hyb_{arm}_{state}_g{threshold:.2f}".replace(".", "")
            glance_threshold, lesion_threshold = threshold, 0.50
        else:
            tag = "gf_hyb_cross_0822"
            glance_threshold = float(row["glance_threshold"])
            lesion_threshold = float(row["lesion_threshold"])
        result.append(
            {
                "tag": tag,
                "family": family,
                "checkpoint": str(Path(row["checkpoint"]).resolve()),
                "glance_threshold": glance_threshold,
                "lesion_threshold": lesion_threshold,
                "DSC": float(row["DSC"]),
                "NSD": float(row["NSD"]),
                "metrics": str(Path(row["metrics"]).resolve()),
                "robust": str(robust.resolve()),
            }
        )
    return sorted(result, key=lambda row: (row["DSC"], row["NSD"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-summary", type=Path, required=True)
    parser.add_argument("--glance-summary", type=Path, required=True)
    parser.add_argument("--cross-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")

    families = {
        "cross": promoted(args.cross_summary, "cross", "lesion_threshold"),
        "focus": promoted(args.focus_summary, "focus", "lesion_threshold"),
        "glance": promoted(args.glance_summary, "glance", "glance_threshold"),
    }
    selected = []
    # Prefer the tested cross-module state, then preserve one candidate from
    # each independently controlled module family.
    for family in ("cross", "focus", "glance"):
        if families[family] and len(selected) < args.count:
            selected.append(families[family].pop(0))
    remainder = sorted(
        [row for rows in families.values() for row in rows],
        key=lambda row: (row["DSC"], row["NSD"]),
        reverse=True,
    )
    selected.extend(remainder[: max(0, args.count - len(selected))])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for rank, row in enumerate(selected, 1):
            writer.writerow({"rank": rank, **row})
    print(f"selected {len(selected)} strictly promoted candidates")


if __name__ == "__main__":
    main()
