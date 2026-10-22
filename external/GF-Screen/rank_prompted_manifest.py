#!/usr/bin/env python3
"""Score and rank a prediction manifest in its deployed prompt-confirmed form.

Raw segmentation rank is not a reliable proxy for rank after component
confirmation.  This utility materializes the frozen prompt transform for every
declared public candidate, scores the complete grid in one label-loading pass,
and emits a deterministic deployment-space ranking.  Hidden predictions are
never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {
    "variant",
    "checkpoint_tag",
    "checkpoint",
    "lesion_threshold",
    "min_mm3",
    "score_tag",
    "predictions",
    "metrics",
}


def read_metrics(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"Name", "Lesion_DSC", "Lesion_NSD"}.issubset(rows[0]):
        raise ValueError(f"invalid metric CSV: {path}")
    return (
        [row["Name"] for row in rows],
        np.asarray([float(row["Lesion_DSC"]) for row in rows]),
        np.asarray([float(row["Lesion_NSD"]) for row in rows]),
    )


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not tag:
        raise ValueError(f"empty sanitized tag from {value!r}")
    return tag


def ranking_tier(decision: str, dsc: float, nsd: float, anchor_dsc: float, anchor_nsd: float) -> str:
    if decision == "PROMOTE":
        return "PROMOTE"
    if dsc >= anchor_dsc and nsd >= anchor_nsd:
        return "NONREGRESS"
    return "FALLBACK"


def rank_key(row: dict[str, str]) -> tuple[int, float, float, float]:
    tier_order = {"PROMOTE": 2, "NONREGRESS": 1, "FALLBACK": 0}
    return (
        tier_order[row["deployment_tier"]],
        float(row["prompt_combined"]),
        float(row["prompt_DSC"]),
        float(row["prompt_NSD"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prompt-support", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, required=True)
    parser.add_argument("--prompt-anchor", type=Path, required=True)
    parser.add_argument("--fusion-script", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with args.manifest.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f"manifest is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest contains no candidates")

    tags = [safe_tag(row["score_tag"]) for row in rows]
    if len(tags) != len(set(tags)):
        raise ValueError("candidate score tags are not unique after sanitization")

    anchor_names, anchor_dsc_values, anchor_nsd_values = read_metrics(args.prompt_anchor)
    anchor_dsc = float(anchor_dsc_values.mean())
    anchor_nsd = float(anchor_nsd_values.mean())
    args.output.mkdir(parents=True, exist_ok=True)
    prompted_root = args.output / "predictions"
    score_root = args.output / "scores"
    prompted_root.mkdir(parents=True, exist_ok=True)
    score_root.mkdir(parents=True, exist_ok=True)

    prepared: list[tuple[dict[str, str], str, Path]] = []
    for row, tag in zip(rows, tags, strict=True):
        base = Path(row["predictions"])
        raw_metrics = Path(row["metrics"])
        raw_names, _, _ = read_metrics(raw_metrics)
        if raw_names != anchor_names:
            raise ValueError(f"raw metric case order differs from anchor: {raw_metrics}")
        if len(list(base.glob("*.nii.gz"))) != len(anchor_names):
            raise RuntimeError(f"expected {len(anchor_names)} base masks: {base}")
        prepared.append((row, tag, base))

    command = [
        sys.executable,
        str(args.evaluator),
        "--gt",
        str(args.labels),
        "--support",
        str(args.prompt_support),
        "--config",
        str(args.prompt_config),
        "--output-dir",
        str(score_root),
        "--workers",
        str(args.workers),
    ]
    for _, tag, base in prepared:
        command.extend(("--candidate", f"{tag}={base}"))
    run(command, args.output / "grid.score.log")

    ranked: list[dict[str, str]] = []
    for row, tag, _ in prepared:
        metrics = score_root / f"{tag}.metrics.csv"
        names, dsc_values, nsd_values = read_metrics(metrics)
        if names != anchor_names:
            raise ValueError(f"prompt metric case order differs from anchor: {metrics}")
        robust = score_root / f"{tag}.robust.json"
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
            score_root / f"{tag}.robust.log",
        )
        selection = json.loads(robust.read_text())
        dsc = float(dsc_values.mean())
        nsd = float(nsd_values.mean())
        ranked.append(
            {
                **row,
                "prompt_predictions": "",
                "prompt_metrics": str(metrics),
                "prompt_robust": str(robust),
                "prompt_decision": str(selection["decision"]),
                "prompt_DSC": f"{dsc:.6f}",
                "prompt_NSD": f"{nsd:.6f}",
                "prompt_combined": f"{(dsc + nsd) / 2:.6f}",
                "deployment_tier": ranking_tier(
                    str(selection["decision"]), dsc, nsd, anchor_dsc, anchor_nsd
                ),
            }
        )

    ranked.sort(key=rank_key, reverse=True)
    selected = ranked[0]
    destination = prompted_root / safe_tag(selected["score_tag"])
    destination.mkdir(parents=True, exist_ok=True)
    if len(list(destination.glob("*.nii.gz"))) != len(anchor_names):
        run(
            [
                sys.executable,
                str(args.fusion_script),
                "materialize",
                "--base",
                selected["predictions"],
                "--support",
                str(args.prompt_support),
                "--config",
                str(args.prompt_config),
                "--output",
                str(destination),
            ],
            args.output / f"{safe_tag(selected['score_tag'])}.materialize.log",
        )
    if len(list(destination.glob("*.nii.gz"))) != len(anchor_names):
        raise RuntimeError(f"incomplete selected prompt masks: {destination}")
    selected["prompt_predictions"] = str(destination)
    fields = list(ranked[0])
    for filename, output_rows in (("ranked.tsv", ranked), ("selected.tsv", ranked[:1])):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(output_rows)
    (args.output / "ranking.json").write_text(
        json.dumps(
            {
                "anchor_DSC": anchor_dsc,
                "anchor_NSD": anchor_nsd,
                "selection_policy": "PROMOTE, then mean-nonregression, then deployed combined",
                "selected": ranked[0],
                "candidates": ranked,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print((args.output / "selected.tsv").read_text(), end="")


if __name__ == "__main__":
    main()
