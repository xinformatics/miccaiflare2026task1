#!/usr/bin/env python3
"""Select a bounded safe subset from the DSC-only hysteresis screen."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", default=3, type=int)
    parser.add_argument("--max-zero", default=4, type=int)
    args = parser.parse_args()

    rows = [
        row for row in json.loads(args.input.read_text())
        if int(row["DSC_zero"]) <= args.max_zero
    ]
    rows.sort(
        key=lambda row: (float(row["DSC"]), -int(row["DSC_zero"])), reverse=True
    )
    rows = rows[: args.top]
    if not rows:
        raise SystemExit("no hysteresis configuration passed the zero-case guard")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        stream.write("low\thigh\tmin_mm3\tscreen_DSC\tzero_DSC\n")
        for row in rows:
            stream.write(
                f"{float(row['low']):g}\t{float(row['high']):g}\t"
                f"{float(row['min_mm3']):g}\t{float(row['DSC']):.8f}\t"
                f"{int(row['DSC_zero'])}\n"
            )
    print(args.output.read_text(), end="")


if __name__ == "__main__":
    main()
