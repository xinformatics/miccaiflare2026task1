#!/usr/bin/env python3
"""Select a bounded set of safe component-addition configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--max-zero", type=int, default=4)
    parser.add_argument("--max-catastrophic-rate", type=float, default=0.10)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())["summary"]
    rows = [
        row for row in rows
        if row["zero_dice_cases"] <= args.max_zero
        and row["catastrophic_regression_rate"] <= args.max_catastrophic_rate
    ]
    rows.sort(key=lambda row: (row["mean_dice"], row["win_rate"]), reverse=True)
    selected = rows[: args.top]
    if not selected:
        raise RuntimeError("no component-addition configuration passed safety filters")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write("mode\tmin_mm3\tscreen_DSC\twin_rate\tzeros\tcatastrophic_rate\n")
        for row in selected:
            handle.write(
                f"{row['mode']}\t{row['min_mm3']:g}\t{row['mean_dice']:.8f}\t"
                f"{row['win_rate']:.6f}\t{row['zero_dice_cases']}\t"
                f"{row['catastrophic_regression_rate']:.6f}\n"
            )
    print(args.output.read_text(), end="")


if __name__ == "__main__":
    main()
