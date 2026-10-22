#!/usr/bin/env python3
"""Decide whether a public-audited independent proposal merits hidden materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select(
    robust: dict,
    healthy_anchor: dict | None = None,
    healthy_candidate: dict | None = None,
    min_dsc_gain: float = 0.01,
    min_nsd_gain: float = 0.0,
    min_win_rate: float = 0.52,
    max_catastrophic_rate: float = 0.05,
) -> dict:
    strict = robust.get("decision") == "PROMOTE"
    gain = robust["paired_gain"]
    win_rate = float(robust["win_rate"])
    catastrophic = float(robust["catastrophic_regression_rate"])
    checks = {
        "dsc_gain": float(gain["Lesion_DSC"]) >= min_dsc_gain,
        "nsd_noninferiority": float(gain["Lesion_NSD"]) >= min_nsd_gain,
        "paired_win_rate": win_rate >= min_win_rate,
        "catastrophic_tail": catastrophic <= max_catastrophic_rate,
    }
    exploratory = all(checks.values())
    public_materialize = strict or exploratory
    if (healthy_anchor is None) != (healthy_candidate is None):
        raise ValueError("healthy anchor and candidate must be supplied together")
    healthy_checks = None
    if healthy_anchor is not None and healthy_candidate is not None:
        if int(healthy_anchor["cases"]) != int(healthy_candidate["cases"]):
            raise ValueError("healthy anchor and candidate case counts differ")
        healthy_checks = {
            "false_positive_cases_noninferior": (
                int(healthy_candidate["false_positive_cases"])
                <= int(healthy_anchor["false_positive_cases"])
            ),
            "false_positive_volume_noninferior": (
                float(healthy_candidate["mean_false_positive_mm3"])
                <= float(healthy_anchor["mean_false_positive_mm3"]) + 1e-9
            ),
        }
    healthy_safe = healthy_checks is None or all(healthy_checks.values())
    materialize = public_materialize and healthy_safe
    if not public_materialize or not healthy_safe:
        decision = "KEEP"
    else:
        decision = "PROMOTE" if strict else "EXPLORATORY"
    return {
        "decision": decision,
        "materialize_hidden": materialize,
        "public_materialize_hidden": public_materialize,
        "strict_robust_promotion": strict,
        "checks": checks,
        "healthy_safety_checks": healthy_checks,
        "thresholds": {
            "min_dsc_gain": min_dsc_gain,
            "min_nsd_gain": min_nsd_gain,
            "min_win_rate": min_win_rate,
            "max_catastrophic_regression_rate": max_catastrophic_rate,
        },
        "evidence": {
            "dsc_gain": float(gain["Lesion_DSC"]),
            "nsd_gain": float(gain["Lesion_NSD"]),
            "win_rate": win_rate,
            "catastrophic_regression_rate": catastrophic,
            "healthy_anchor": healthy_anchor,
            "healthy_candidate": healthy_candidate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robust-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-dsc-gain", type=float, default=0.01)
    parser.add_argument("--min-nsd-gain", type=float, default=0.0)
    parser.add_argument("--min-win-rate", type=float, default=0.52)
    parser.add_argument("--max-catastrophic-rate", type=float, default=0.05)
    parser.add_argument("--healthy-anchor", type=Path)
    parser.add_argument("--healthy-candidate", type=Path)
    args = parser.parse_args()
    robust = json.loads(args.robust_selection.read_text())
    healthy_anchor = (
        json.loads(args.healthy_anchor.read_text()) if args.healthy_anchor else None
    )
    healthy_candidate = (
        json.loads(args.healthy_candidate.read_text()) if args.healthy_candidate else None
    )
    result = select(
        robust,
        healthy_anchor=healthy_anchor,
        healthy_candidate=healthy_candidate,
        min_dsc_gain=args.min_dsc_gain,
        min_nsd_gain=args.min_nsd_gain,
        min_win_rate=args.min_win_rate,
        max_catastrophic_rate=args.max_catastrophic_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["decision"])


if __name__ == "__main__":
    main()
