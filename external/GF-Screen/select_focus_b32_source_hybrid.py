#!/usr/bin/env python3
"""Predeclare a source-family B32/Focus contour hybrid from exact OOF patches.

The selector is deliberately local-only. It compares like-for-like grouped,
case-held-out patch metrics and permits Focus for a source family only when it
improves both case- and cohort-balanced contour Dice. Full-volume OOF and the
fixed public gate remain mandatory before deployment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path


EXPECTED_GROUPS = {"mature", "weak_sources", "lung_cells"}
MINIMUM_CASE_GAIN = 0.01
MINIMUM_COHORT_GAIN = 0.005
MAXIMUM_FOCUS_NEGATIVE_NONEMPTY_RATE = 0.95


def decide(focus: dict, b32: dict) -> dict:
    if int(focus.get("folds", -1)) != 5:
        raise ValueError("Focus summary is not exact five-fold OOF")
    coverage = b32.get("coverage", {})
    if (
        not coverage.get("every_patch_exactly_once")
        or int(coverage.get("cases", 0)) != 707
    ):
        raise ValueError("B32 summary is not exact 707-case OOF")
    if focus.get("index_sha256") != b32.get("index_sha256"):
        raise ValueError("Focus and B32 summaries use different patch indices")
    focus_groups = focus.get("source_segmentation_groups", {})
    b32_groups = b32.get("source_segmentation_groups", {})
    if set(focus_groups) != EXPECTED_GROUPS or set(b32_groups) != EXPECTED_GROUPS:
        raise ValueError("source-family contour coverage mismatch")
    focus_negative = float(focus.get("negative_nonempty_rate", float("nan")))
    if not math.isfinite(focus_negative):
        raise ValueError("non-finite Focus negative activation rate")

    group_records: dict[str, dict[str, object]] = {}
    selected_focus: list[str] = []
    selected_b32: list[str] = []
    source_to_route: dict[str, str] = {}
    source_segmentation_thresholds: dict[str, float] = {}
    for group in sorted(EXPECTED_GROUPS):
        focus_group = focus_groups[group]
        b32_group = b32_groups[group]
        if set(focus_group.get("sources", [])) != set(b32_group.get("sources", [])):
            raise ValueError(f"source membership mismatch for {group}")
        focus_case = float(
            focus_group.get("case_balanced_dice", focus_group.get("dice", float("nan")))
        )
        focus_cohort = float(focus_group["cohort_balanced_dice"])
        focus_group_negative = float(
            focus_group.get("negative_nonempty_rate", float("nan"))
        )
        b32_case = float(b32_group.get("dice", float("nan")))
        b32_cohort = float(b32_group["cohort_balanced_dice"])
        values = (
            focus_case,
            focus_cohort,
            focus_group_negative,
            b32_case,
            b32_cohort,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite contour metric for {group}")
        case_gain = focus_case - b32_case
        cohort_gain = focus_cohort - b32_cohort
        use_focus = bool(
            focus_negative <= MAXIMUM_FOCUS_NEGATIVE_NONEMPTY_RATE
            and focus_group_negative <= MAXIMUM_FOCUS_NEGATIVE_NONEMPTY_RATE
            and case_gain >= MINIMUM_CASE_GAIN
            and cohort_gain >= MINIMUM_COHORT_GAIN
        )
        family = "focus" if use_focus else "b32"
        # One route per model family prevents loading the same 72.76M Focus
        # checkpoint multiple times when both weak and lung groups select it.
        route = family
        selected = focus_group if use_focus else b32_group
        threshold = float(selected["threshold"])
        if not 0 < threshold < 1:
            raise ValueError(f"invalid selected contour threshold for {group}")
        (selected_focus if use_focus else selected_b32).append(group)
        for source in sorted(selected["sources"]):
            source_to_route[source] = route
            source_segmentation_thresholds[source] = threshold
        group_records[group] = {
            "selected_family": family,
            "route": route,
            "sources": sorted(selected["sources"]),
            "focus_case_dice": focus_case,
            "focus_cohort_dice": focus_cohort,
            "focus_negative_nonempty_rate": focus_group_negative,
            "b32_case_dice": b32_case,
            "b32_cohort_dice": b32_cohort,
            "case_gain": case_gain,
            "cohort_gain": cohort_gain,
            "checks": {
                "case_gain": case_gain >= MINIMUM_CASE_GAIN,
                "cohort_gain": cohort_gain >= MINIMUM_COHORT_GAIN,
                "focus_negative_nonempty_rate": focus_negative
                <= MAXIMUM_FOCUS_NEGATIVE_NONEMPTY_RATE
                and focus_group_negative <= MAXIMUM_FOCUS_NEGATIVE_NONEMPTY_RATE,
            },
            "selected_threshold": threshold,
        }

    if selected_focus and selected_b32:
        decision = "QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME"
    elif selected_focus:
        decision = "USE_GLOBAL_FOCUS_PATH"
    else:
        decision = "KEEP_B32"
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "decision": decision,
        "passed": decision == "QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME",
        "index_sha256": focus["index_sha256"],
        "folds": 5,
        "cases": 707,
        "focus_input_mode": focus.get("input_mode"),
        "focus_negative_nonempty_rate": focus_negative,
        "gate": {
            "minimum_case_gain": MINIMUM_CASE_GAIN,
            "minimum_cohort_gain": MINIMUM_COHORT_GAIN,
            "maximum_focus_negative_nonempty_rate": MAXIMUM_FOCUS_NEGATIVE_NONEMPTY_RATE,
            "requires_both_model_families": True,
        },
        "groups": group_records,
        "focus_groups": selected_focus,
        "b32_groups": selected_b32,
        "source_to_route": dict(sorted(source_to_route.items())),
        "source_segmentation_thresholds": dict(
            sorted(source_segmentation_thresholds.items())
        ),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-summary", type=Path, required=True)
    parser.add_argument("--b32-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    focus = json.loads(args.focus_summary.read_text(encoding="utf-8"))
    b32 = json.loads(args.b32_summary.read_text(encoding="utf-8"))
    payload = decide(focus, b32)
    payload["focus_summary"] = str(args.focus_summary.resolve())
    payload["b32_summary"] = str(args.b32_summary.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
