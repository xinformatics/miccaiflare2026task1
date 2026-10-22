#!/usr/bin/env python3
"""Select diverse near-anchor hybrid backups for unused submission capacity.

Strictly promoted candidates are handled by ``build_hybrid_submission_plan``.
This planner is deliberately second-line: it excludes already saved states and
admits only candidates close to the deployed public anchor with a bounded tail
regression rate.  It creates a save plan, never predictions or uploads.
"""

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
    "gate",
)


def read_saved(path: Path | None) -> set[tuple[str, str, float, float]]:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return {
        (
            row["family"],
            str(Path(row["checkpoint"]).resolve()),
            round(float(row["glance_threshold"]), 6),
            round(float(row["lesion_threshold"]), 6),
        )
        for row in rows
    }


def state_tag(checkpoint_tag: str) -> str:
    state = checkpoint_tag.replace("checkpoint_epoch_", "e")
    return {"model": "best", "model_final": "final"}.get(state, state)


def candidates(
    summary: Path,
    family: str,
    threshold_column: str,
    saved: set[tuple[str, str, float, float]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = []
    for row in rows:
        robust_path = Path(row["robust"])
        report = json.loads(robust_path.read_text(encoding="utf-8"))
        anchor_dsc = float(report["anchor_mean"]["Lesion_DSC"])
        anchor_nsd = float(report["anchor_mean"]["Lesion_NSD"])
        dsc = float(row["DSC"])
        nsd = float(row["NSD"])
        combined_drop = (anchor_dsc + anchor_nsd - dsc - nsd) / 2
        strict = report["decision"] == "PROMOTE"
        near_anchor = (
            dsc >= anchor_dsc - args.max_dsc_drop
            and nsd >= anchor_nsd - args.max_nsd_drop
            and combined_drop <= args.max_combined_drop
            and float(report["catastrophic_regression_rate"])
            <= args.max_catastrophic_rate
        )
        if not (strict or near_anchor):
            continue

        threshold = float(row[threshold_column])
        checkpoint = str(Path(row["checkpoint"]).resolve())
        state = state_tag(row.get("checkpoint_tag", "cross"))
        if family == "focus":
            arm = "f08" if "complete08" in row["variant"] else "f04"
            tag = f"gf_bak_{arm}_{state}_l{threshold:.2f}".replace(".", "")
            glance_threshold, lesion_threshold = 0.50, threshold
        elif family == "glance":
            arm = "g1e4" if "lr1e4" in row["variant"] else "g3e5"
            tag = f"gf_bak_{arm}_{state}_g{threshold:.2f}".replace(".", "")
            glance_threshold, lesion_threshold = threshold, 0.50
        else:
            tag = "gf_bak_cross_0822"
            glance_threshold = float(row["glance_threshold"])
            lesion_threshold = float(row["lesion_threshold"])
        identity = (
            family,
            checkpoint,
            round(glance_threshold, 6),
            round(lesion_threshold, 6),
        )
        if identity in saved:
            continue
        result.append(
            {
                "tag": tag,
                "family": family,
                "checkpoint": checkpoint,
                "glance_threshold": glance_threshold,
                "lesion_threshold": lesion_threshold,
                "DSC": dsc,
                "NSD": nsd,
                "metrics": str(Path(row["metrics"]).resolve()),
                "robust": str(robust_path.resolve()),
                "gate": "PROMOTE" if strict else "NEAR_ANCHOR_BACKUP",
            }
        )
    return sorted(
        result,
        key=lambda item: (
            item["gate"] == "PROMOTE",
            (float(item["DSC"]) + float(item["NSD"])) / 2,
            float(item["DSC"]),
        ),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-summary", type=Path, required=True)
    parser.add_argument("--glance-summary", type=Path, required=True)
    parser.add_argument("--cross-summary", type=Path, required=True)
    parser.add_argument("--saved", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-dsc-drop", type=float, default=0.005)
    parser.add_argument("--max-nsd-drop", type=float, default=0.005)
    parser.add_argument("--max-combined-drop", type=float, default=0.0025)
    parser.add_argument("--max-catastrophic-rate", type=float, default=0.04)
    args = parser.parse_args()
    if args.count < 0:
        raise ValueError("count must be non-negative")

    saved = read_saved(args.saved)
    families = {
        "cross": candidates(
            args.cross_summary, "cross", "lesion_threshold", saved, args
        ),
        "focus": candidates(
            args.focus_summary, "focus", "lesion_threshold", saved, args
        ),
        "glance": candidates(
            args.glance_summary, "glance", "glance_threshold", saved, args
        ),
    }
    selected: list[dict[str, object]] = []
    for family in ("cross", "focus", "glance"):
        if families[family] and len(selected) < args.count:
            selected.append(families[family].pop(0))
    remainder = sorted(
        [item for rows in families.values() for item in rows],
        key=lambda item: (
            item["gate"] == "PROMOTE",
            (float(item["DSC"]) + float(item["NSD"])) / 2,
            float(item["DSC"]),
        ),
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
    print(f"selected {len(selected)} near-anchor backup candidates")


if __name__ == "__main__":
    main()
