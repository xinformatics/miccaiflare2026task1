#!/usr/bin/env python3
"""Validate a gated Focus epoch-route summary and emit inference arguments.

The emitted tab-separated records are consumed by the fixed-public and
save-only controllers.  This keeps route identity, checkpoint hashes, and
source coverage identical across local OOF, public, and hidden inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROUTE_GROUPS = {"mature", "weak_sources", "lung_cells"}
INPUT_MODES = {"soft", "soft_seed", "soft_lung_seed"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_records(
    summary_path: Path,
    decision_path: Path,
    expected_sources: set[str],
) -> list[tuple[str, str, str, str]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != "QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME":
        raise ValueError("Focus epoch routes did not pass their patch-OOF gate")
    if bool(decision.get("public_read")) or bool(decision.get("hidden_read")):
        raise ValueError("Focus route decision is not local-only")
    if Path(decision.get("candidate", "")).resolve() != summary_path.resolve():
        raise ValueError("Focus route decision references a different summary")
    if decision.get("candidate_sha256") != sha256(summary_path):
        raise ValueError("Focus route summary checksum mismatch")
    if bool(summary.get("public_read")) or bool(summary.get("hidden_read")):
        raise ValueError("Focus route summary is not local-only")
    if int(summary.get("folds", 0)) != 5:
        raise ValueError("Focus route summary is not exact five-fold OOF")
    if set(summary.get("route_groups", [])) != ROUTE_GROUPS:
        raise ValueError("Focus route group coverage mismatch")
    input_mode = summary.get("training_recipe", {}).get("input_mode")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"invalid routed Focus input mode: {input_mode}")
    source_to_route = summary.get("source_to_route", {})
    if set(source_to_route) != expected_sources:
        missing = sorted(expected_sources - set(source_to_route))
        extra = sorted(set(source_to_route) - expected_sources)
        raise ValueError(
            f"Focus source route coverage mismatch: missing={missing} extra={extra}"
        )
    if set(source_to_route.values()) != ROUTE_GROUPS:
        raise ValueError("Focus source mappings do not cover every route group")

    folds = summary.get("fold_records", [])
    if sorted(int(row.get("fold", -1)) for row in folds) != list(range(5)):
        raise ValueError("invalid Focus route fold records")
    result = [("INPUT", "-", "-", str(input_mode))]
    for row in sorted(folds, key=lambda item: int(item["fold"])):
        fold = int(row["fold"])
        selected = row.get("selected", {})
        if set(selected) != ROUTE_GROUPS:
            raise ValueError(f"Focus route coverage mismatch in fold {fold}")
        for group in sorted(ROUTE_GROUPS):
            record = selected[group]
            checkpoint = Path(record["checkpoint"])
            if not checkpoint.is_file():
                raise ValueError(f"missing routed Focus checkpoint: {checkpoint}")
            if record.get("checkpoint_sha256") != sha256(checkpoint):
                raise ValueError(
                    f"routed Focus checkpoint checksum mismatch: {checkpoint}"
                )
            result.append(("ROUTE", str(fold), group, str(checkpoint.resolve())))
    for source, group in sorted(source_to_route.items()):
        result.append(("SOURCE", "-", source, group))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-sources", required=True)
    args = parser.parse_args()
    expected = {value for value in args.expected_sources.split(",") if value}
    if not expected:
        parser.error("--expected-sources must not be empty")
    for record in route_records(args.summary, args.decision, expected):
        print("\t".join(record))


if __name__ == "__main__":
    main()
