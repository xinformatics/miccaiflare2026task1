#!/usr/bin/env python3
"""Emit validated deployment records for a mixed B32/Focus contour route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def records(summary: dict, decision: dict, expected_sources: set[str]) -> list[str]:
    if decision.get("decision") != "QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME":
        raise ValueError("mixed B32/Focus route did not qualify")
    if decision.get("index_sha256") != summary.get("index_sha256"):
        raise ValueError("mixed route and Focus summary index mismatch")
    source_to_route = decision.get("source_to_route", {})
    if set(source_to_route) != expected_sources:
        raise ValueError("mixed route source coverage mismatch")
    if set(source_to_route.values()) != {"b32", "focus"}:
        raise ValueError("mixed route must use both model families")
    input_mode = decision.get("focus_input_mode")
    if input_mode != summary.get("input_mode") or input_mode not in {
        "soft",
        "soft_seed",
        "soft_lung_seed",
    }:
        raise ValueError("mixed route Focus input mode mismatch")
    folds = summary.get("fold_records", [])
    if sorted(int(row["fold"]) for row in folds) != list(range(5)):
        raise ValueError("mixed route requires five Focus checkpoints")
    result = [f"INPUT\t-\t-\t{input_mode}"]
    result.extend(
        f"FOCUS\t{int(row['fold'])}\t-\t{row['checkpoint']}"
        for row in sorted(folds, key=lambda item: int(item["fold"]))
    )
    result.extend(
        f"SOURCE\t-\t{source}\t{route}"
        for source, route in sorted(source_to_route.items())
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-sources", required=True)
    args = parser.parse_args()
    expected = {value for value in args.expected_sources.split(",") if value}
    if not expected:
        parser.error("expected sources cannot be empty")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    print("\n".join(records(summary, decision, expected)))


if __name__ == "__main__":
    main()
