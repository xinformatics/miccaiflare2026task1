#!/usr/bin/env python3
"""Select a runtime-safe, paired-qualified Focus TTA deployment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path


EXPECTED_VARIANTS = {"notta", "tta_x", "tta_xy"}


def select_candidate(rows: list[dict[str, str]]) -> dict:
    by_variant = {row.get("variant", ""): row for row in rows}
    if len(by_variant) != len(rows) or set(by_variant) != EXPECTED_VARIANTS:
        raise ValueError("TTA summary must contain each expected variant exactly once")
    parsed: dict[str, dict[str, object]] = {}
    for variant, row in by_variant.items():
        try:
            dsc = float(row["DSC"])
            nsd = float(row["NSD"])
            combined = float(row["combined"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid TTA metric row: {variant}") from error
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (dsc, nsd, combined)):
            raise ValueError(f"invalid TTA metric value: {variant}")
        parsed[variant] = {
            "variant": variant,
            "DSC": dsc,
            "NSD": nsd,
            "combined": combined,
            "runtime_eligible": row.get("runtime_eligible") == "yes",
            "vs_notta": row.get("vs_notta", ""),
            "vs_champion": row.get("vs_champion", ""),
            "champion_anchor_match": row.get("champion_anchor_match") == "yes",
        }
    baseline = parsed["notta"]
    if baseline["vs_notta"] != "ANCHOR" or baseline["vs_champion"] != "ANCHOR":
        raise ValueError("no-TTA row is not the declared anchor")
    if baseline["champion_anchor_match"] is not True:
        raise ValueError("no-TTA deployment does not reproduce the champion within locked bounds")
    eligible = [
        row
        for variant, row in parsed.items()
        if variant != "notta"
        and row["runtime_eligible"] is True
        and row["vs_notta"] == "PROMOTE"
        and row["vs_champion"] == "PROMOTE"
        and row["DSC"] >= baseline["DSC"]
        and row["NSD"] >= baseline["NSD"]
    ]
    eligible.sort(
        key=lambda row: (row["DSC"], row["NSD"], row["combined"], row["variant"]),
        reverse=True,
    )
    return {
        "decision": "PROMOTE_TTA_HIDDEN" if eligible else "KEEP_CHAMPION",
        "selected": eligible[0] if eligible else None,
        "baseline": baseline,
        "eligible": eligible,
        "selection_order": "DSC, then NSD, then combined; paired gates are mandatory",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "summary": str(args.summary.resolve()),
        **select_candidate(rows),
        "public_read": True,
        "hidden_read": False,
        "submission_created": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
