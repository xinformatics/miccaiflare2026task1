#!/usr/bin/env python3
"""Paired DSC-first promotion with NSD as a non-regression constraint."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


METRICS = ("Lesion_DSC", "Lesion_NSD")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metrics(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no metric rows in {path}")
    names = [row["Name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate case names in {path}")
    values = np.asarray(
        [[float(row[metric]) for metric in METRICS] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"invalid metric values in {path}")
    return {"names": names, "values": values}


def align(anchor: dict[str, object], candidate: dict[str, object]) -> tuple[list[str], np.ndarray, np.ndarray]:
    anchor_names = list(anchor["names"])
    candidate_names = list(candidate["names"])
    if set(anchor_names) != set(candidate_names):
        raise ValueError(
            f"case mismatch: anchor={len(anchor_names)}, candidate={len(candidate_names)}, "
            f"paired={len(set(anchor_names) & set(candidate_names))}"
        )
    names = sorted(anchor_names)
    anchor_index = {name: index for index, name in enumerate(anchor_names)}
    candidate_index = {name: index for index, name in enumerate(candidate_names)}
    anchor_values = np.asarray(anchor["values"])[[anchor_index[name] for name in names]]
    candidate_values = np.asarray(candidate["values"])[[candidate_index[name] for name in names]]
    return names, anchor_values, candidate_values


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (100.0 - confidence) / 2.0
    low, high = np.percentile(values, (tail, 100.0 - tail))
    return float(low), float(high)


def evaluate(
    anchor_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
    min_dsc_gain: float,
    min_dsc_win_rate: float,
    min_nsd_gain: float,
    min_nsd_uncertainty_bound: float,
    max_catastrophic_dsc_rate: float,
    catastrophic_dsc_delta: float,
) -> dict[str, object]:
    delta = candidate_values - anchor_values
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty((bootstrap, 2), dtype=np.float64)
    for start in range(0, bootstrap, 2000):
        stop = min(bootstrap, start + 2000)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        bootstrap_means[start:stop] = delta[indices].mean(axis=1)

    mean = delta.mean(axis=0)
    dsc_ci = interval(bootstrap_means[:, 0], 95.0)
    # NSD is a safety constraint rather than an improvement objective. A 90%
    # interval with a small predeclared tolerance distinguishes maintenance
    # from a material loss without demanding a statistically positive NSD gain.
    nsd_ci = interval(bootstrap_means[:, 1], 90.0)
    dsc_win_rate = float(np.mean(delta[:, 0] > 0))
    catastrophic_dsc_rate = float(np.mean(delta[:, 0] < catastrophic_dsc_delta))
    checks = {
        "dsc_practical_gain": bool(mean[0] >= min_dsc_gain),
        "dsc_ci_positive": bool(dsc_ci[0] > 0),
        "dsc_win_rate": bool(dsc_win_rate >= min_dsc_win_rate),
        "nsd_mean_nonregression": bool(mean[1] >= min_nsd_gain),
        "nsd_uncertainty_nonregression": bool(nsd_ci[0] >= min_nsd_uncertainty_bound),
        "dsc_tail_guard": bool(catastrophic_dsc_rate <= max_catastrophic_dsc_rate),
    }
    return {
        "anchor_mean": dict(zip(METRICS, anchor_values.mean(axis=0).tolist())),
        "candidate_mean": dict(zip(METRICS, candidate_values.mean(axis=0).tolist())),
        "paired_gain": dict(zip(METRICS, mean.tolist())),
        "bootstrap_intervals": {
            "Lesion_DSC_95ci": dsc_ci,
            "Lesion_NSD_90ci": nsd_ci,
        },
        "dsc_win_rate": dsc_win_rate,
        "catastrophic_dsc_rate": catastrophic_dsc_rate,
        "checks": checks,
        "decision": "PROMOTE" if all(checks.values()) else "KEEP_ANCHOR",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--anchor-name", default="anchor")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--min-dsc-gain", type=float, default=0.003)
    parser.add_argument("--min-dsc-win-rate", type=float, default=0.52)
    parser.add_argument("--min-nsd-gain", type=float, default=0.0)
    parser.add_argument("--min-nsd-uncertainty-bound", type=float, default=-0.005)
    parser.add_argument("--max-catastrophic-dsc-rate", type=float, default=0.10)
    parser.add_argument("--catastrophic-dsc-delta", type=float, default=-0.20)
    args = parser.parse_args()

    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    anchor = read_metrics(args.anchor)
    candidate = read_metrics(args.candidate)
    names, anchor_values, candidate_values = align(anchor, candidate)
    result = evaluate(
        anchor_values,
        candidate_values,
        bootstrap=args.bootstrap,
        seed=args.seed,
        min_dsc_gain=args.min_dsc_gain,
        min_dsc_win_rate=args.min_dsc_win_rate,
        min_nsd_gain=args.min_nsd_gain,
        min_nsd_uncertainty_bound=args.min_nsd_uncertainty_bound,
        max_catastrophic_dsc_rate=args.max_catastrophic_dsc_rate,
        catastrophic_dsc_delta=args.catastrophic_dsc_delta,
    )
    delta = candidate_values - anchor_values
    worst = np.argsort(delta[:, 0])[: min(10, len(names))]
    result.update(
        {
            "schema_version": 1,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "objective": "maximize Lesion_DSC subject to Lesion_NSD non-regression",
            "implementation": str(Path(__file__).resolve()),
            "implementation_sha256": sha256(Path(__file__).resolve()),
            "anchor": args.anchor_name,
            "candidate": args.candidate_name,
            "anchor_metrics": str(args.anchor.resolve()),
            "anchor_metrics_sha256": sha256(args.anchor),
            "candidate_metrics": str(args.candidate.resolve()),
            "candidate_metrics_sha256": sha256(args.candidate),
            "cases": len(names),
            "criteria": {
                "bootstrap": args.bootstrap,
                "seed": args.seed,
                "min_dsc_gain": args.min_dsc_gain,
                "min_dsc_win_rate": args.min_dsc_win_rate,
                "min_nsd_gain": args.min_nsd_gain,
                "min_nsd_uncertainty_bound": args.min_nsd_uncertainty_bound,
                "max_catastrophic_dsc_rate": args.max_catastrophic_dsc_rate,
                "catastrophic_dsc_delta": args.catastrophic_dsc_delta,
            },
            "worst_dsc_regressions": [
                {
                    "case": names[index],
                    "dsc_delta": float(delta[index, 0]),
                    "nsd_delta": float(delta[index, 1]),
                }
                for index in worst
            ],
        }
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
