#!/usr/bin/env python3
"""Gate source-routed Focus epochs before any full-volume or public inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def decide(
    baseline: dict,
    candidate: dict,
    minimum_case_gain: float = 0.005,
    minimum_cohort_gain: float = 0.002,
    maximum_negative_rate_increase: float = 0.02,
) -> tuple[str, dict[str, float], dict[str, bool]]:
    baseline_case = float(baseline["case_balanced_segmentation_dice"])
    baseline_cohort = float(baseline["cohort_balanced_segmentation_dice"])
    baseline_negative = float(baseline["negative_nonempty_rate"])
    routed = candidate["segmentation"]
    candidate_case = float(routed["dice"])
    candidate_cohort = float(routed["cohort_balanced_dice"])
    candidate_negative = float(routed["negative_nonempty_rate"])
    metrics = {
        "baseline_case_dice": baseline_case,
        "candidate_case_dice": candidate_case,
        "case_dice_gain": candidate_case - baseline_case,
        "baseline_cohort_dice": baseline_cohort,
        "candidate_cohort_dice": candidate_cohort,
        "cohort_dice_gain": candidate_cohort - baseline_cohort,
        "baseline_negative_nonempty_rate": baseline_negative,
        "candidate_negative_nonempty_rate": candidate_negative,
        "negative_nonempty_rate_increase": candidate_negative - baseline_negative,
    }
    checks = {
        "meaningful_case_gain": metrics["case_dice_gain"] >= minimum_case_gain,
        "cohort_gain": metrics["cohort_dice_gain"] >= minimum_cohort_gain,
        "negative_rate_bounded": metrics["negative_nonempty_rate_increase"]
        <= maximum_negative_rate_increase,
        "route_groups_complete": set(candidate.get("route_groups", []))
        == {"mature", "weak_sources", "lung_cells"},
        "fivefold_complete": int(candidate.get("folds", 0)) == 5
        and len(candidate.get("fold_records", [])) == 5,
    }
    decision = (
        "QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME"
        if all(checks.values())
        else "KEEP_GLOBAL_FOCUS_EPOCHS"
    )
    return decision, metrics, checks


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-case-gain", type=float, default=0.005)
    parser.add_argument("--minimum-cohort-gain", type=float, default=0.002)
    parser.add_argument(
        "--maximum-negative-rate-increase", type=float, default=0.02
    )
    args = parser.parse_args()
    if min(
        args.minimum_case_gain,
        args.minimum_cohort_gain,
        args.maximum_negative_rate_increase,
    ) < 0:
        parser.error("gate tolerances must be nonnegative")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if baseline.get("index_sha256") != candidate.get("index_sha256"):
        raise ValueError("global and routed Focus summaries use different indices")
    decision, metrics, checks = decide(
        baseline,
        candidate,
        args.minimum_case_gain,
        args.minimum_cohort_gain,
        args.maximum_negative_rate_increase,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": sha256(args.baseline),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "metrics": metrics,
        "checks": checks,
        "thresholds": {
            "minimum_case_gain": args.minimum_case_gain,
            "minimum_cohort_gain": args.minimum_cohort_gain,
            "maximum_negative_rate_increase": args.maximum_negative_rate_increase,
        },
        "next_stage": (
            "matched exact707 source-routed Focus full-volume OOF"
            if decision == "QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME"
            else "none"
        ),
        "public_read": False,
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
