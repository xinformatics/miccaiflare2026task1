#!/usr/bin/env python3
"""Shortlist stable validation peaks instead of chasing every checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


PATTERN = re.compile(r"Final validation\s+(\d+)/\d+\s+dice\s+([0-9.eE+-]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-size", type=int, default=25)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--min-distance", type=int, default=15)
    args = parser.parse_args()

    observations: dict[int, float] = {}
    for epoch_text, dice_text in PATTERN.findall(args.log.read_text(encoding="utf-8")):
        # Log epoch is zero-based; checkpoint filenames are one-based.
        observations[int(epoch_text) + 1] = float(dice_text)
    if not observations:
        raise ValueError(f"no validation entries in {args.log}")

    epochs = sorted(observations)
    stable = {}
    for index, epoch in enumerate(epochs):
        neighbors = epochs[max(0, index - 1): min(len(epochs), index + 2)]
        values = [observations[item] for item in neighbors]
        # A local spike is discounted by its adjacent validation checkpoints.
        stable[epoch] = float(np.mean(values))

    phase_winners = []
    first_phase = min(epochs) // args.phase_size
    last_phase = max(epochs) // args.phase_size
    for phase in range(first_phase, last_phase + 1):
        members = [epoch for epoch in epochs if epoch // args.phase_size == phase]
        if members:
            winner = max(members, key=lambda epoch: stable[epoch])
            phase_winners.append(winner)

    ranked = sorted(set(phase_winners + epochs), key=lambda epoch: stable[epoch], reverse=True)
    # Always retain the raw validation champion, then add peaks that are
    # stable across adjacent validation checkpoints and distinct LR phases.
    selected = [max(epochs, key=lambda epoch: observations[epoch])]
    for epoch in ranked:
        if epoch in selected:
            continue
        checkpoint = args.run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        if not checkpoint.exists():
            continue
        if all(abs(epoch - prior) >= args.min_distance for prior in selected):
            selected.append(epoch)
        if len(selected) == args.top:
            break

    report = {
        "selection_rule": {
            "stable_score": "mean of checkpoint and adjacent validation observations",
            "phase_size": args.phase_size,
            "minimum_epoch_distance": args.min_distance,
        },
        "observations": [
            {"checkpoint": epoch, "dice": observations[epoch], "stable_score": stable[epoch]}
            for epoch in epochs
        ],
        "shortlist": [
            {
                "checkpoint": epoch,
                "path": str(args.run_dir / f"checkpoint_epoch_{epoch:04d}.pt"),
                "dice": observations[epoch],
                "stable_score": stable[epoch],
            }
            for epoch in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["shortlist"], indent=2))


if __name__ == "__main__":
    main()
