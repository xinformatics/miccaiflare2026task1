#!/usr/bin/env python3
"""Find the closest paired public/hidden submissions to a saved probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def nearest_neighbors(history, public_dsc, public_nsd, count):
    paired = []
    for row in history:
        required = ("public_dsc", "public_nsd", "codabench_dsc", "codabench_nsd")
        if not all(row.get(key, "").strip() for key in required):
            continue
        distance = math.hypot(
            float(row["public_dsc"]) - public_dsc,
            float(row["public_nsd"]) - public_nsd,
        )
        paired.append(
            {
                "model": row["model"],
                "configuration": row["configuration"],
                "public_dsc": float(row["public_dsc"]),
                "public_nsd": float(row["public_nsd"]),
                "hidden_dsc": float(row["codabench_dsc"]),
                "hidden_nsd": float(row["codabench_nsd"]),
                "public_euclidean_distance": distance,
            }
        )
    if len(paired) < count:
        raise ValueError(f"requested {count} neighbors from only {len(paired)} pairs")
    return sorted(paired, key=lambda row: row["public_euclidean_distance"])[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.neighbors < 1:
        raise ValueError("--neighbors must be positive")
    with args.history.open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream, delimiter="\t"))
    with args.reserved.open(newline="", encoding="utf-8") as stream:
        matches = [
            row for row in csv.DictReader(stream, delimiter="\t")
            if row["tag"] == args.tag
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one reserved row for {args.tag}, found {len(matches)}")
    candidate = matches[0]
    neighbors = nearest_neighbors(
        history,
        float(candidate["public_DSC"]),
        float(candidate["public_NSD"]),
        args.neighbors,
    )
    hidden_dsc = [row["hidden_dsc"] for row in neighbors]
    hidden_nsd = [row["hidden_nsd"] for row in neighbors]
    report = {
        "warning": (
            "Nearest-neighbor history is a slot-prioritization diagnostic, "
            "not a hidden-score estimate or leaderboard claim."
        ),
        "distance": "unscaled Euclidean distance in public (DSC, NSD)",
        "candidate": candidate,
        "neighbor_count": len(neighbors),
        "neighbors": neighbors,
        "neighbor_hidden_dsc": {
            "minimum": min(hidden_dsc),
            "mean": sum(hidden_dsc) / len(hidden_dsc),
            "maximum": max(hidden_dsc),
        },
        "neighbor_hidden_nsd": {
            "minimum": min(hidden_nsd),
            "mean": sum(hidden_nsd) / len(hidden_nsd),
            "maximum": max(hidden_nsd),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
