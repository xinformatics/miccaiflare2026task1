#!/usr/bin/env python3
"""Audit existing fixed-public metric files for champion zero-Dice cases.

The script deliberately reads metric CSVs only.  It does not inspect hidden
data or predictions, and it does not select a deployment.  Its purpose is to
answer a narrower, reproducible question: has any already-scored candidate
shown useful overlap on a case that the current public champion misses
completely?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REQUIRED_METRIC_COLUMNS = {"Name", "Lesion_DSC", "Lesion_NSD"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--anchor-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def normalize_case(value: str) -> str:
    name = Path(value.strip()).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Candidate:
    path: Path
    values: dict[str, tuple[float, float]]

    @property
    def macro_dsc(self) -> float:
        return sum(value[0] for value in self.values.values()) / len(self.values)

    @property
    def macro_nsd(self) -> float:
        return sum(value[1] for value in self.values.values()) / len(self.values)


def read_anchor(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"case", "dice"}.issubset(rows[0]):
        raise ValueError(f"anchor must contain case,dice columns: {path}")
    cases = [normalize_case(row["case"]) for row in rows]
    if len(cases) != len(set(cases)):
        raise ValueError(f"anchor contains duplicate cases: {path}")
    zero_cases = [
        normalize_case(row["case"])
        for row in rows
        if float(row["dice"]) == 0.0
    ]
    if not zero_cases:
        raise ValueError(f"anchor contains no zero-Dice cases: {path}")
    return cases, zero_cases


def read_metric_candidate(path: Path) -> dict[str, tuple[float, float]] | None:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not REQUIRED_METRIC_COLUMNS.issubset(
                reader.fieldnames
            ):
                return None
            values: dict[str, tuple[float, float]] = {}
            for row in reader:
                case = normalize_case(row["Name"])
                if not case or case in values:
                    return None
                values[case] = (float(row["Lesion_DSC"]), float(row["Lesion_NSD"]))
            return values
    except (OSError, UnicodeDecodeError, csv.Error, TypeError, ValueError):
        return None


def audit(
    search_root: Path,
    anchor_cases: Path,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    public_cases, zero_cases = read_anchor(anchor_cases)
    public_set = set(public_cases)
    csv_paths = sorted(search_root.rglob("*.csv"))
    metric_csv_count = 0
    candidates: list[Candidate] = []
    for path in csv_paths:
        values = read_metric_candidate(path)
        if values is None:
            continue
        metric_csv_count += 1
        if set(values) == public_set:
            candidates.append(Candidate(path=path, values=values))

    if not candidates:
        raise RuntimeError("no complete fixed-public metric CSVs found")

    ranked_rows: list[dict[str, object]] = []
    best_by_case: dict[str, dict[str, object]] = {}
    for case in zero_cases:
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.values[case][0],
                -candidate.values[case][1],
                str(candidate.path),
            ),
        )
        for rank, candidate in enumerate(ranked[:top_k], start=1):
            dsc, nsd = candidate.values[case]
            row = {
                "case": case,
                "rank": rank,
                "case_dsc": dsc,
                "case_nsd": nsd,
                "public_macro_dsc": candidate.macro_dsc,
                "public_macro_nsd": candidate.macro_nsd,
                "metrics_path": str(candidate.path.resolve()),
            }
            ranked_rows.append(row)
            if rank == 1:
                best_by_case[case] = {
                    **row,
                    "metrics_sha256": sha256(candidate.path),
                }

    summary: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "fixed-public champion zero-Dice candidate-bank audit",
        "search_root": str(search_root.resolve()),
        "anchor_cases": str(anchor_cases.resolve()),
        "anchor_sha256": sha256(anchor_cases),
        "csv_files_scanned": len(csv_paths),
        "metric_csv_files": metric_csv_count,
        "complete_fixed_public_candidates": len(candidates),
        "fixed_public_case_count": len(public_cases),
        "zero_dice_cases": zero_cases,
        "best_by_case": best_by_case,
        "interpretation_guard": (
            "Per-case maxima are diagnostics selected with public labels, not a "
            "deployable router or evidence for a Codabench submission."
        ),
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    return summary, ranked_rows


def write_tsv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty ranking")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary, rows = audit(args.search_root, args.anchor_cases, args.top_k)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    ranked_path = args.output / "ranked.tsv"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_tsv(ranked_path, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
