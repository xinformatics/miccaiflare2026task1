#!/usr/bin/env python3
"""Rank whole fixed-public candidates by incumbent zero-case recovery.

Unlike a per-case oracle, every row is one globally deployable candidate. The
output remains a public diagnostic and does not compose, route, or package a
submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from audit_public_zero_case_bank import (
    Candidate,
    read_anchor,
    read_metric_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--anchor-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-case-dsc", type=float, default=0.1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    public_cases, zero_cases = read_anchor(args.anchor_cases)
    public_set = set(public_cases)
    candidates = []
    scanned = 0
    for path in sorted(args.search_root.rglob("*.csv")):
        scanned += 1
        values = read_metric_candidate(path)
        if values is not None and set(values) == public_set:
            candidates.append(Candidate(path=path, values=values))
    if not candidates:
        raise SystemExit("no complete public candidates found")

    rows = []
    for candidate in candidates:
        zero_dsc = [candidate.values[case][0] for case in zero_cases]
        zero_nsd = [candidate.values[case][1] for case in zero_cases]
        rows.append({
            "recovered_zero_cases": sum(value >= args.minimum_case_dsc for value in zero_dsc),
            "zero_case_dsc_sum": sum(zero_dsc),
            "zero_case_dsc_mean": sum(zero_dsc) / len(zero_dsc),
            "zero_case_nsd_mean": sum(zero_nsd) / len(zero_nsd),
            "public_macro_dsc": candidate.macro_dsc,
            "public_macro_nsd": candidate.macro_nsd,
            **{f"dsc_{case}": candidate.values[case][0] for case in zero_cases},
            "metrics_path": str(candidate.path.resolve()),
            "metrics_sha256": sha256(candidate.path),
        })
    rows.sort(
        key=lambda row: (
            -int(row["recovered_zero_cases"]),
            -float(row["zero_case_dsc_sum"]),
            -float(row["public_macro_dsc"]),
            str(row["metrics_path"]),
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    table = args.output / "candidates.tsv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "purpose": "whole-candidate incumbent zero-case recovery audit",
        "csv_files_scanned": scanned,
        "complete_public_candidates": len(candidates),
        "zero_cases": zero_cases,
        "minimum_case_dsc": args.minimum_case_dsc,
        "maximum_recovered_zero_cases_by_one_candidate": max(int(row["recovered_zero_cases"]) for row in rows),
        "top_candidates": rows[:20],
        "table": {"path": str(table.resolve()), "sha256": sha256(table)},
        "interpretation_guard": "whole candidates only; public diagnostic, not a per-case router, hidden prediction, or submission selector",
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
