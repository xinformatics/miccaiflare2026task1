#!/usr/bin/env python3
"""Compare candidate-refiner sampling temperatures without starting training."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from candidate_refiner_sampling import sampling_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, action="append", default=[])
    args = parser.parse_args()
    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    temperatures = args.temperature or [0.0, 0.5, 1.0]
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "index": str(args.index.resolve()),
        "interpretation": (
            "temperature 0 reproduces v2 label balancing; 1 fully balances "
            "label/cohort/case; intermediate values trade coverage for variance"
        ),
        "audits": [sampling_audit(rows, value) for value in temperatures],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
