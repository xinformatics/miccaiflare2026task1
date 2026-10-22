#!/usr/bin/env python3
"""Rank completed short Focus families after the frozen prompt verifier.

The short probes are generated and selected without prompt postprocessing for
diagnostic comparability.  Long compute, however, should be allocated by the
performance of the method that is actually deployed.  This script applies one
pre-existing prompt configuration to one winner per objective family, scores
the resulting masks, and selects two distinct families without touching hidden
data or introducing a new inference hyperparameter.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def read_metrics(path: Path) -> dict[str, np.ndarray | list[str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        "names": [row["Name"] for row in rows],
        "dsc": np.asarray([float(row["Lesion_DSC"]) for row in rows]),
        "nsd": np.asarray([float(row["Lesion_NSD"]) for row in rows]),
    }


def prediction_dir(row: dict[str, str]) -> Path:
    metrics = Path(row["metrics"])
    minimum = int(float(row["min_mm3"]))
    if minimum > 0:
        suffix = ".metrics.csv"
        if not str(metrics).endswith(suffix):
            raise ValueError(f"unexpected metrics suffix: {metrics}")
        return Path(str(metrics)[: -len(suffix)])
    sweep = metrics.parent.parent
    threshold = float(row["lesion_threshold"])
    return sweep / row["variant"] / row["checkpoint_tag"] / f"lesion_{threshold:.2f}"


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def mean_metrics(path: Path) -> tuple[float, float]:
    data = read_metrics(path)
    return float(data["dsc"].mean()), float(data["nsd"].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prompt-support", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, required=True)
    parser.add_argument("--prompt-anchor", type=Path, required=True)
    parser.add_argument("--fusion-script", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()

    source = json.loads(args.selection_json.read_text())
    candidates = source.get("all_candidates", source["all_family_winners"])
    families = {row["family"] for row in candidates}
    if len(families) < args.count:
        raise RuntimeError(f"only {len(families)} objective families are available")
    anchor_dsc, anchor_nsd = mean_metrics(args.prompt_anchor)
    anchor_combined = (anchor_dsc + anchor_nsd) / 2
    prompted_root = args.output / "prompted"
    prompted_root.mkdir(parents=True, exist_ok=True)

    ranked: list[dict[str, str]] = []
    prepared: list[tuple[dict[str, str], str, Path]] = []
    for row in candidates:
        base = prediction_dir(row)
        if len(list(base.glob("*.nii.gz"))) != 50:
            raise RuntimeError(f"expected 50 base masks: {base}")
        tag = (
            f"{row['family']}__{row['variant']}__{row['checkpoint_tag']}"
            f"__l{float(row['lesion_threshold']):.2f}__mm{int(float(row['min_mm3']))}"
        )
        destination = prompted_root / tag
        destination.mkdir(parents=True, exist_ok=True)
        if len(list(destination.glob("*.nii.gz"))) != 50:
            run(
                [
                    sys.executable,
                    str(args.fusion_script),
                    "materialize",
                    "--base",
                    str(base),
                    "--support",
                    str(args.prompt_support),
                    "--config",
                    str(args.prompt_config),
                    "--output",
                    str(destination),
                ],
                prompted_root / f"{tag}.materialize.log",
            )
        if len(list(destination.glob("*.nii.gz"))) != 50:
            raise RuntimeError(f"expected 50 prompt-confirmed masks: {destination}")

        prepared.append((row, tag, destination))

    grid_command = [
        sys.executable,
        str(args.evaluator),
        "--gt",
        str(args.labels),
        "--output-dir",
        str(prompted_root),
        "--workers",
        str(args.workers),
    ]
    for _, tag, destination in prepared:
        grid_command.extend(("--candidate", f"{tag}={destination}"))
    run(grid_command, prompted_root / "grid.score.log")

    for row, tag, destination in prepared:
        metrics = prompted_root / f"{tag}.metrics.csv"
        if not metrics.exists():
            raise RuntimeError(f"grid evaluator did not produce metrics: {metrics}")
        robust = prompted_root / f"{tag}.robust.json"
        if not robust.exists():
            run(
                [
                    sys.executable,
                    str(args.selector),
                    "--anchor",
                    str(args.prompt_anchor),
                    "--candidate",
                    str(metrics),
                    "--anchor-name",
                    "prompt_confirmed_champion",
                    "--candidate-name",
                    tag,
                    "--output",
                    str(robust),
                ],
                prompted_root / f"{tag}.robust.log",
            )
        selection = json.loads(robust.read_text())
        dsc, nsd = mean_metrics(metrics)
        combined = (dsc + nsd) / 2
        eligible = (
            dsc >= anchor_dsc - 0.02
            and combined >= anchor_combined - 0.02
            and float(selection["catastrophic_regression_rate"]) <= 0.10
        )
        ranked.append(
            {
                **{key: str(value) for key, value in row.items()},
                "prompt_predictions": str(destination),
                "prompt_metrics": str(metrics),
                "prompt_robust": str(robust),
                "prompt_DSC": f"{dsc:.6f}",
                "prompt_NSD": f"{nsd:.6f}",
                "prompt_combined": f"{combined:.6f}",
                "prompt_decision": str(selection["decision"]),
                "prompt_eligible": str(eligible).lower(),
            }
        )

    rank_key = lambda row: (  # noqa: E731 - compact shared deterministic key
        row["prompt_eligible"] == "true",
        float(row["prompt_combined"]),
    )
    ranked.sort(key=rank_key, reverse=True)
    family_best = []
    for family in sorted(families):
        rows = [row for row in ranked if row["family"] == family]
        rows.sort(key=rank_key, reverse=True)
        family_best.append(rows[0])
    family_best.sort(key=rank_key, reverse=True)
    selected = family_best[: args.count]
    leading = ["family", "variant", "checkpoint_tag", "checkpoint"]
    fields = leading + [key for key in selected[0] if key not in leading]
    with (args.output / "selected_prompted.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)
    with (args.output / "all_prompted.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(ranked)
    (args.output / "prompted_selection.json").write_text(
        json.dumps(
            {
                "anchor_DSC": anchor_dsc,
                "anchor_NSD": anchor_nsd,
                "anchor_combined": anchor_combined,
                "selected": selected,
                "all_family_winners": family_best,
                "all_candidates": ranked,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print((args.output / "selected_prompted.tsv").read_text(), end="")


if __name__ == "__main__":
    main()
