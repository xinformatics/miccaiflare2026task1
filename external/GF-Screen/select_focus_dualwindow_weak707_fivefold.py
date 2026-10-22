#!/usr/bin/env python3
"""Gate five-fold dual-window Focus contours against matched B32 patch OOF."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-summary", type=Path, required=True)
    parser.add_argument("--b32-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    focus = json.loads(args.focus_summary.read_text(encoding="utf-8"))
    b32 = json.loads(args.b32_summary.read_text(encoding="utf-8"))
    if int(focus.get("folds", -1)) != 5 or focus.get("input_mode") != "soft_lung_seed":
        raise ValueError("Focus summary is not exact five-fold dual-window OOF")
    coverage = b32.get("coverage", {})
    if not coverage.get("every_patch_exactly_once") or int(coverage.get("cases", 0)) < 650:
        raise ValueError("B32 summary lacks balanced-707 OOF coverage")
    if focus.get("index_sha256") != b32.get("index_sha256"):
        raise ValueError("Focus and B32 summaries use different patch indices")

    b32_segmentation = b32["segmentation"]
    metrics = {
        "focus_case_dice": float(focus["case_balanced_segmentation_dice"]),
        "focus_cohort_dice": float(focus["cohort_balanced_segmentation_dice"]),
        "focus_negative_nonempty_rate": float(focus["negative_nonempty_rate"]),
        "b32_case_dice": float(b32_segmentation["dice"]),
        "b32_cohort_dice": float(b32_segmentation["cohort_balanced_dice"]),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("non-finite five-fold metric")
    metrics["case_gain_over_b32"] = (
        metrics["focus_case_dice"] - metrics["b32_case_dice"]
    )
    metrics["cohort_gain_over_b32"] = (
        metrics["focus_cohort_dice"] - metrics["b32_cohort_dice"]
    )
    gate = {
        "minimum_case_dice_gain_over_b32": 0.01,
        "minimum_cohort_dice_gain_over_b32": 0.01,
        "maximum_negative_nonempty_rate": 0.95,
    }
    checks = {
        "case_gain_over_b32": metrics["case_gain_over_b32"]
        >= gate["minimum_case_dice_gain_over_b32"],
        "cohort_gain_over_b32": metrics["cohort_gain_over_b32"]
        >= gate["minimum_cohort_dice_gain_over_b32"],
        "negative_nonempty_rate": metrics["focus_negative_nonempty_rate"]
        <= gate["maximum_negative_nonempty_rate"],
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "decision": "PROMOTE_TO_FULL_VOLUME_OOF" if passed else "KEEP_B32",
        "passed": passed,
        "focus_summary": str(args.focus_summary.resolve()),
        "b32_summary": str(args.b32_summary.resolve()),
        "metrics": metrics,
        "gate": gate,
        "checks": checks,
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
