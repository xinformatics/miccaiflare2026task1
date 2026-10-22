#!/usr/bin/env python3
"""Validate and emit an arbitrary predeclared no-TTA Focus/B32 route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(payload: dict, expected_sources: set[str]) -> list[str]:
    if payload.get("decision") != "QUALIFY_ENCODER_FOCUS_B32_NOTTA_FOR_FIXED_PUBLIC":
        raise ValueError("composite route did not qualify for fixed public")
    if payload.get("focus_input_mode") != "soft_lung_seed":
        raise ValueError("invalid composite Focus input mode")
    mapping = payload.get("source_to_route", {})
    if set(mapping) != expected_sources or "b32" not in set(mapping.values()):
        raise ValueError("composite source coverage mismatch")
    folds = {int(row["fold"]): row for row in payload.get("fold_records", [])}
    if set(folds) != set(range(5)):
        raise ValueError("composite route requires five folds")
    output = ["INPUT\t-\t-\tsoft_lung_seed"]
    for fold in range(5):
        routes = folds[fold].get("routes", {})
        if set(routes) != set(mapping.values()) - {"b32"}:
            raise ValueError(f"fold {fold} route coverage mismatch")
        for name, route in sorted(routes.items()):
            path = Path(str(route["checkpoint"]))
            if not path.is_file() or sha256(path) != route.get("checkpoint_sha256"):
                raise ValueError(f"fold {fold} route checkpoint changed: {name}")
            if int(route.get("tta_views", -1)) != 1:
                raise ValueError("TTA is forbidden in this route")
            output.append(f"FOCUS\t{fold}\t{name}\t{path.resolve()}")
    output.extend(f"SOURCE\t-\t{source}\t{route}" for source, route in sorted(mapping.items()))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-sources", required=True)
    args = parser.parse_args()
    expected = {value for value in args.expected_sources.split(",") if value}
    print("\n".join(records(json.loads(args.decision.read_text(encoding="utf-8")), expected)))


if __name__ == "__main__":
    main()
