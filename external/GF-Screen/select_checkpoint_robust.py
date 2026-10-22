#!/usr/bin/env python3
"""Paired, uncertainty-aware checkpoint comparison for FLARE metrics.

This is intentionally conservative: a candidate must improve both DSC and NSD
by a practical margin, have a positive paired bootstrap confidence interval,
and avoid a large tail of case-level regressions. A tiny mean gain is reported
but is not allowed to replace the deployed anchor.
"""

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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metrics(path: Path) -> dict[str, np.ndarray | list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no rows in {path}")
    names = [row["Name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate case names in {path}")
    result: dict[str, np.ndarray | list[str]] = {"names": names}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(f"invalid {metric} values in {path}")
        result[metric] = values
    return result


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--anchor-name", default="anchor")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--min-dsc-gain", type=float, default=0.015)
    parser.add_argument("--min-nsd-gain", type=float, default=0.015)
    parser.add_argument("--min-win-rate", type=float, default=0.55)
    parser.add_argument("--max-catastrophic-rate", type=float, default=0.10)
    args = parser.parse_args()

    anchor = read_metrics(args.anchor)
    candidate = read_metrics(args.candidate)
    anchor_index = {name: i for i, name in enumerate(anchor["names"])}
    candidate_index = {name: i for i, name in enumerate(candidate["names"])}
    names = sorted(set(anchor_index) & set(candidate_index))
    if len(names) != len(anchor["names"]) or len(names) != len(candidate["names"]):
        raise ValueError(
            f"case mismatch: anchor={len(anchor['names'])}, candidate={len(candidate['names'])}, paired={len(names)}"
        )

    anchor_values = np.column_stack([
        np.asarray(anchor[metric])[[anchor_index[name] for name in names]] for metric in METRICS
    ])
    candidate_values = np.column_stack([
        np.asarray(candidate[metric])[[candidate_index[name] for name in names]] for metric in METRICS
    ])
    delta = candidate_values - anchor_values
    combined_delta = delta.mean(axis=1)

    rng = np.random.default_rng(args.seed)
    # Chunking avoids a large bootstrap-by-case allocation on bigger holdouts.
    bootstrap_means = np.empty((args.bootstrap, 3), dtype=np.float64)
    for start in range(0, args.bootstrap, 2000):
        stop = min(args.bootstrap, start + 2000)
        indices = rng.integers(0, len(names), size=(stop - start, len(names)))
        bootstrap_means[start:stop, :2] = delta[indices].mean(axis=1)
        bootstrap_means[start:stop, 2] = combined_delta[indices].mean(axis=1)

    dsc_gain, nsd_gain = delta.mean(axis=0)
    combined_gain = combined_delta.mean()
    dsc_ci = percentile_interval(bootstrap_means[:, 0])
    nsd_ci = percentile_interval(bootstrap_means[:, 1])
    combined_ci = percentile_interval(bootstrap_means[:, 2])
    win_rate = float((combined_delta > 0).mean())
    tie_rate = float((combined_delta == 0).mean())
    catastrophic_rate = float((combined_delta < -0.20).mean())

    checks = {
        "dsc_practical_gain": bool(dsc_gain >= args.min_dsc_gain),
        "nsd_practical_gain": bool(nsd_gain >= args.min_nsd_gain),
        "dsc_ci_positive": bool(dsc_ci[0] > 0),
        "nsd_ci_positive": bool(nsd_ci[0] > 0),
        "combined_ci_positive": bool(combined_ci[0] > 0),
        "paired_win_rate": bool(win_rate >= args.min_win_rate),
        "tail_regression_guard": bool(catastrophic_rate <= args.max_catastrophic_rate),
    }
    promote = all(checks.values())

    worst_indices = np.argsort(combined_delta)[: min(10, len(names))]
    report = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "anchor_metrics": str(args.anchor.resolve()),
        "anchor_metrics_sha256": file_sha256(args.anchor),
        "candidate_metrics": str(args.candidate.resolve()),
        "candidate_metrics_sha256": file_sha256(args.candidate),
        "anchor": args.anchor_name,
        "candidate": args.candidate_name,
        "cases": len(names),
        "anchor_mean": dict(zip(METRICS, anchor_values.mean(axis=0).tolist())),
        "candidate_mean": dict(zip(METRICS, candidate_values.mean(axis=0).tolist())),
        "paired_gain": {
            "Lesion_DSC": float(dsc_gain),
            "Lesion_NSD": float(nsd_gain),
            "combined": float(combined_gain),
        },
        "bootstrap_95ci": {
            "Lesion_DSC": dsc_ci,
            "Lesion_NSD": nsd_ci,
            "combined": combined_ci,
        },
        "win_rate": win_rate,
        "tie_rate": tie_rate,
        "catastrophic_regression_rate": catastrophic_rate,
        "delta_quantiles_combined": {
            str(q): float(np.quantile(combined_delta, q)) for q in (0.05, 0.10, 0.25, 0.50, 0.75)
        },
        "worst_regressions": [
            {
                "case": names[i],
                "dsc_delta": float(delta[i, 0]),
                "nsd_delta": float(delta[i, 1]),
                "combined_delta": float(combined_delta[i]),
            }
            for i in worst_indices
        ],
        "checks": checks,
        "criteria": {
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "min_dsc_gain": args.min_dsc_gain,
            "min_nsd_gain": args.min_nsd_gain,
            "min_win_rate": args.min_win_rate,
            "max_catastrophic_rate": args.max_catastrophic_rate,
        },
        "decision": "PROMOTE" if promote else "KEEP_ANCHOR",
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
