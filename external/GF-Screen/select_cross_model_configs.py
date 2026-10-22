#!/usr/bin/env python3
"""Select a bounded, predeclared subset from a DSC-only fusion screen."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", default=3, type=int)
    parser.add_argument("--max-zero", default=7, type=int)
    parser.add_argument("--require-finite-rescue", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    candidates = []
    for row in data["summary"]:
        rescue = float(row["rescue_min_mm3"])
        if args.require_finite_rescue and not math.isfinite(rescue):
            continue
        if int(row["zero_dice_cases"]) > args.max_zero:
            continue
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            float(row["mean_dice"]),
            -int(row["zero_dice_cases"]),
            float(row["win_rate"]),
        ),
        reverse=True,
    )
    selected = candidates[: args.top]
    if not selected:
        raise SystemExit("no cross-model configurations passed the safety screen")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        stream.write("distance_mm\trescue_min_mm3\tscreen_dsc\twin_rate\tzero_dice_cases\n")
        for row in selected:
            stream.write(
                f"{float(row['distance_mm']):g}\t"
                f"{float(row['rescue_min_mm3']):g}\t"
                f"{float(row['mean_dice']):.8f}\t"
                f"{float(row['win_rate']):.4f}\t"
                f"{int(row['zero_dice_cases'])}\n"
            )
    print(args.output.read_text(), end="")


if __name__ == "__main__":
    main()
