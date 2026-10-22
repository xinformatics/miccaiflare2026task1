#!/usr/bin/env python3
"""Validate the direct Focus public-only decision before downstream work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DSC_GATE = 0.6154316379500749
NSD_FLOOR = 0.5104225107769019
DECISIONS = {"QUALIFY_SUBMISSION_PREPARATION", "KEEP_TOUCHCOMP"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    terminal_path: Path,
    route_path: Path,
    calibration_path: Path,
    summary_path: Path,
) -> str:
    payload = json.loads(terminal_path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise ValueError("direct Focus public terminal is incomplete")
    decision = payload.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"unexpected direct Focus decision: {decision}")

    for key, path in (
        ("route", route_path),
        ("calibration", calibration_path),
        ("summary", summary_path),
    ):
        identity = payload.get(key, {})
        recorded = Path(identity.get("path", ""))
        if recorded.resolve() != path.resolve() or identity.get("sha256") != sha256(path):
            raise ValueError(f"direct Focus {key} identity mismatch")

    fixed = payload.get("fixed_public", {})
    try:
        dsc = float(fixed["DSC"])
        nsd = float(fixed["NSD"])
        dsc_gate = float(fixed["DSC_gate"])
        nsd_floor = float(fixed["NSD_floor"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid direct Focus fixed-public metrics") from error
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (dsc, nsd)):
        raise ValueError("invalid direct Focus fixed-public metric range")
    if dsc_gate != DSC_GATE or nsd_floor != NSD_FLOOR:
        raise ValueError("direct Focus public gate metadata drift")

    with summary_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError("expected exactly one direct Focus public summary row")
    try:
        summary_dsc = float(rows[0]["DSC"])
        summary_nsd = float(rows[0]["NSD"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid direct Focus public summary metrics") from error
    # The shell summary is intentionally rendered to four decimals for humans,
    # while the terminal retains the exact mean from the paired-gate JSON.
    if abs(summary_dsc - dsc) > 5.1e-5 or abs(summary_nsd - nsd) > 5.1e-5:
        raise ValueError("direct Focus terminal/summary metric mismatch")

    qualified = dsc >= DSC_GATE and nsd >= NSD_FLOOR
    if qualified != (decision == "QUALIFY_SUBMISSION_PREPARATION"):
        raise ValueError("direct Focus public decision/gate mismatch")
    if (
        payload.get("public_read") is not True
        or payload.get("hidden_read") is not False
        or payload.get("submission_created") is not False
        or payload.get("uploaded") is not False
    ):
        raise ValueError("direct Focus public-only provenance mismatch")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    print(validate(args.terminal, args.route, args.calibration, args.summary))


if __name__ == "__main__":
    main()
