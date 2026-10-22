#!/usr/bin/env python3
"""Audit proposal-domain priors and target scale in a completed patch index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from candidate_refiner_sampling import (
    hierarchical_sample_weights,
    semantic_source_group,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_rows(
    rows: list[dict[str, str]], sampling_temperature: float
) -> tuple[dict[str, dict[str, object]], dict[str, float]]:
    if not rows:
        raise ValueError("patch index is empty")
    patch_ids = [str(row.get("patch_id", "")) for row in rows]
    if any(not value for value in patch_ids) or len(patch_ids) != len(set(patch_ids)):
        raise ValueError("patch IDs must be nonempty and unique")
    groups = [semantic_source_group(row.get("source_name", "")) for row in rows]
    labels = [int(row["label"]) for row in rows]
    if any(value not in (0, 1) for value in labels):
        raise ValueError("labels must be binary")
    standard = hierarchical_sample_weights(rows, sampling_temperature, False)
    balanced = hierarchical_sample_weights(rows, sampling_temperature, True)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    report: dict[str, dict[str, object]] = {}
    for group, indices in sorted(grouped.items()):
        positive = [index for index in indices if labels[index] == 1]
        report[group] = {
            "rows": len(indices),
            "positive": len(positive),
            "negative": len(indices) - len(positive),
            "positive_fraction": len(positive) / len(indices),
            "cases": len({rows[index]["case"] for index in indices}),
            "positive_cases": len({rows[index]["case"] for index in positive}),
            "sampling_mass": float(standard[indices].sum()),
            "sampling_mass_positive": float(standard[positive].sum()),
            "sampling_mass_negative": float(
                standard[[index for index in indices if labels[index] == 0]].sum()
            ),
            "source_balanced_sampling_mass": float(balanced[indices].sum()),
        }
    total_mass = {group: float(values["sampling_mass"]) for group, values in report.items()}
    if not math.isclose(sum(total_mass.values()), 1.0, abs_tol=1e-9):
        raise AssertionError("sampling masses do not sum to one")
    return report, total_mass


def target_scale(
    rows: list[dict[str, str]], sample_per_group: int, seed: int, cap: float
) -> dict[str, dict[str, float | int]]:
    if sample_per_group < 1 or cap <= 0:
        raise ValueError("invalid target-scale audit settings")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["label"]) == 1:
            grouped[semantic_source_group(row.get("source_name", ""))].append(row)
    output: dict[str, dict[str, float | int]] = {}
    for group, candidates in sorted(grouped.items()):
        selected = sorted(
            candidates,
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['patch_id']}".encode("utf-8")
            ).hexdigest(),
        )[:sample_per_group]
        counts: list[float] = []
        fractions: list[float] = []
        for row in selected:
            path = Path(row["path"])
            with np.load(path, allow_pickle=False) as archive:
                target = np.asarray(archive["target"]) > 0
            if target.ndim != 4 or target.shape[0] != 1 or not target.any():
                raise ValueError(f"invalid positive target: {path}")
            count = float(target.sum())
            counts.append(count)
            fractions.append(count / target.size)
        true_weights = [(1.0 - value) / value for value in fractions]
        output[group] = {
            "population": len(candidates),
            "sample": len(selected),
            "target_voxels_p10": percentile(counts, 10),
            "target_voxels_median": percentile(counts, 50),
            "target_voxels_p90": percentile(counts, 90),
            "target_fraction_p10": percentile(fractions, 10),
            "target_fraction_median": percentile(fractions, 50),
            "uncapped_positive_weight_median": percentile(true_weights, 50),
            "fraction_positive_weight_above_current_cap": float(
                np.mean(np.asarray(true_weights) > cap)
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-temperature", type=float, default=0.5)
    parser.add_argument("--sample-per-group", type=int, default=256)
    parser.add_argument("--positive-weight-cap", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if not 0 <= args.sampling_temperature <= 1:
        parser.error("sampling temperature must be in [0,1]")
    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    sources, masses = summarize_rows(rows, args.sampling_temperature)
    scales = target_scale(
        rows, args.sample_per_group, args.seed, args.positive_weight_cap
    )
    positive_fractions = [float(value["positive_fraction"]) for value in sources.values()]
    source_prior_range = max(positive_fractions) - min(positive_fractions)
    maximum_mass = max(masses.values())
    worst_small_group = max(
        scales,
        key=lambda group: float(
            scales[group]["fraction_positive_weight_above_current_cap"]
        ),
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "index": str(args.index.resolve()),
        "index_sha256": sha256(args.index),
        "cases": len({row["case"] for row in rows}),
        "patches": len(rows),
        "sampling_temperature": args.sampling_temperature,
        "positive_weight_cap": args.positive_weight_cap,
        "sources": sources,
        "target_scale": scales,
        "diagnostics": {
            "source_positive_fraction_range": source_prior_range,
            "maximum_semantic_source_sampling_mass": maximum_mass,
            "worst_small_target_group": worst_small_group,
        },
        "recommendations": {
            "test_source_conditioning": source_prior_range >= 0.20,
            "test_source_balancing": maximum_mass >= 0.65,
            "test_small_lesion_loss": bool(
                float(
                    scales[worst_small_group][
                        "fraction_positive_weight_above_current_cap"
                    ]
                )
                >= 0.50
                and float(scales[worst_small_group]["target_voxels_p10"]) <= 64
            ),
        },
        "training_labels_read": True,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
