#!/usr/bin/env python3
"""Attach aggregate public-validation metrics to the training W&B run."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def flatten_numeric(prefix: str, value: object) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten_numeric(f"{prefix}/{key}" if prefix else str(key), child))
    elif isinstance(value, (int, float)):
        result[prefix] = float(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("run_id_file", type=Path)
    parser.add_argument("--challenge-csv", type=Path)
    args = parser.parse_args()
    if not args.run_id_file.is_file():
        raise SystemExit(f"W&B run ID file is missing: {args.run_id_file}")

    import wandb

    summary = json.loads(args.summary.read_text())
    metrics = flatten_numeric("public_validation", summary.get("foreground_mean", {}))
    metrics["public_validation/num_cases"] = float(len(summary.get("metric_per_case", [])))
    if args.challenge_csv is not None:
        with args.challenge_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            metrics["challenge_public/mean_dsc"] = sum(
                float(row["Lesion_DSC"]) for row in rows
            ) / len(rows)
            metrics["challenge_public/mean_nsd"] = sum(
                float(row["Lesion_NSD"]) for row in rows
            ) / len(rows)
            metrics["challenge_public/num_cases"] = float(len(rows))
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "pancancer-ct"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        id=args.run_id_file.read_text().strip(),
        resume="allow",
        dir=os.environ.get("WANDB_DIR"),
    )
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    run.finish()


if __name__ == "__main__":
    main()
