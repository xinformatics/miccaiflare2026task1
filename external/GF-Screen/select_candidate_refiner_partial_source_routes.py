#!/usr/bin/env python3
"""Select source contour routes independently on exact grouped held-out OOF.

The candidate summary is the predeclared all-source route.  Because its
sources are disjoint, each source report measures only the checkpoint change
for that source.  Sources that do not improve both case- and cohort-balanced
Dice fall back to the globally selected B32 contour.  The emitted payload is
also a complete deployment specification accepted by
``candidate_refiner_route_selection.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


EXPECTED_SOURCES = {
    "epoch105",
    "currentgf",
    "autopet_weak",
    "lndb_weak",
    *(f"lung_cell{index}" for index in range(8)),
}
LUNG_SOURCES = tuple(f"lung_cell{index}" for index in range(8))
ROUTE_UNITS = {
    "epoch105": {"sources": ("epoch105",), "route": "epoch105"},
    "currentgf": {"sources": ("currentgf",), "route": "currentgf"},
    "autopet": {"sources": ("autopet_weak",), "route": "autopet"},
    "lndb": {"sources": ("lndb_weak",), "route": "lndb"},
    "lung": {"sources": LUNG_SOURCES, "route": "lung"},
}
ROUTE_CHECKPOINTS = {
    "global": "checkpoint_best_segmentation.pt",
    "epoch105": "checkpoint_best_epoch105_segmentation.pt",
    "currentgf": "checkpoint_best_currentgf_segmentation.pt",
    "autopet": "checkpoint_best_autopet_segmentation.pt",
    "lndb": "checkpoint_best_lndb_segmentation.pt",
    "lung": "checkpoint_best_lung_segmentation.pt",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coverage(payload: dict) -> tuple[int, int, int, int, bool]:
    value = payload.get("coverage", {})
    return (
        int(value.get("cases", -1)),
        int(value.get("patches", -1)),
        int(value.get("positive", -1)),
        int(value.get("negative", -1)),
        bool(value.get("every_patch_exactly_once", False)),
    )


def _metric(report: dict, name: str) -> float:
    value = float(report[name])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"invalid {name}: {value}")
    return value


def _unit_report(payload: dict, unit: str) -> dict:
    if unit == "lung":
        reports = payload.get("source_segmentation_groups", {})
        report = reports.get("lung_cells")
        if not isinstance(report, dict) or set(report.get("sources", ())) != set(
            LUNG_SOURCES
        ):
            raise ValueError("missing or invalid pooled lung-cell report")
        return report
    source = ROUTE_UNITS[unit]["sources"][0]
    report = payload.get("source_segmentation", {}).get(source)
    if not isinstance(report, dict):
        raise ValueError(f"missing source segmentation report: {source}")
    return report


def select_routes(
    baseline: dict,
    candidate: dict,
    minimum_case_dice_gain: float = 0.002,
    minimum_cohort_dice_gain: float = 0.001,
    probability_tolerance: float = 1e-8,
    expected_cases: int = 707,
) -> tuple[str, dict[str, object], dict[str, bool]]:
    if min(
        minimum_case_dice_gain,
        minimum_cohort_dice_gain,
        probability_tolerance,
    ) < 0:
        raise ValueError("selection tolerances must be nonnegative")
    baseline_coverage = _coverage(baseline)
    candidate_coverage = _coverage(candidate)
    coverage_identical = baseline_coverage == candidate_coverage
    exact_coverage = (
        coverage_identical
        and baseline_coverage[0] == expected_cases
        and baseline_coverage[1] > 0
        and baseline_coverage[2] > 0
        and baseline_coverage[3] > 0
        and baseline_coverage[4]
    )
    same_index = bool(
        baseline.get("index_sha256")
        and baseline.get("index_sha256") == candidate.get("index_sha256")
    )
    before_ap = _metric(baseline["classification"], "average_precision")
    after_ap = _metric(candidate["classification"], "average_precision")
    classification_identical = abs(after_ap - before_ap) <= probability_tolerance

    unit_reports: dict[str, dict[str, object]] = {}
    selected_units: list[str] = []
    source_to_route: dict[str, str] = {}
    for unit, specification in ROUTE_UNITS.items():
        before = _unit_report(baseline, unit)
        after = _unit_report(candidate, unit)
        before_case = _metric(before, "dice")
        after_case = _metric(after, "dice")
        before_cohort = _metric(before, "cohort_balanced_dice")
        after_cohort = _metric(after, "cohort_balanced_dice")
        case_gain = after_case - before_case
        cohort_gain = after_cohort - before_cohort
        selected = bool(
            exact_coverage
            and same_index
            and classification_identical
            and case_gain >= minimum_case_dice_gain
            and cohort_gain >= minimum_cohort_dice_gain
        )
        if selected:
            selected_units.append(unit)
        route = str(specification["route"]) if selected else "global"
        for source in specification["sources"]:
            source_to_route[str(source)] = route
        unit_reports[unit] = {
            "sources": list(specification["sources"]),
            "candidate_route": specification["route"],
            "selected_route": route,
            "baseline_case_dice": before_case,
            "candidate_case_dice": after_case,
            "case_dice_gain": case_gain,
            "baseline_cohort_dice": before_cohort,
            "candidate_cohort_dice": after_cohort,
            "cohort_dice_gain": cohort_gain,
            "positive_cases": int(after.get("positive_cases", -1)),
            "selected": selected,
        }

    if set(source_to_route) != EXPECTED_SOURCES:
        raise AssertionError("internal source-route inventory mismatch")
    used_routes = set(source_to_route.values())
    contour_filenames = {
        route: ROUTE_CHECKPOINTS[route] for route in sorted(used_routes)
    }
    selection = {
        "selected_units": selected_units,
        "fallback_units": sorted(set(ROUTE_UNITS) - set(selected_units)),
        "unit_reports": unit_reports,
        "source_to_route": dict(sorted(source_to_route.items())),
        "used_routes": sorted(used_routes),
        "contour_checkpoint_filenames": contour_filenames,
    }
    checks = {
        "same_exact_index": same_index,
        "coverage_identical": coverage_identical,
        "exact_case_coverage": exact_coverage,
        "classification_identical": classification_identical,
        "at_least_one_source_route_selected": bool(selected_units),
        "complete_source_inventory": set(source_to_route) == EXPECTED_SOURCES,
    }
    decision = (
        "QUALIFY_GROUP_SELECTED_SOURCE_ROUTE"
        if all(checks.values())
        else "KEEP_GLOBAL_CONTOUR"
    )
    return decision, selection, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-case-dice-gain", type=float, default=0.002)
    parser.add_argument("--minimum-cohort-dice-gain", type=float, default=0.001)
    parser.add_argument("--probability-tolerance", type=float, default=1e-8)
    parser.add_argument("--expected-cases", type=int, default=707)
    args = parser.parse_args()
    if args.expected_cases < 1:
        parser.error("expected-cases must be positive")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision, selection, checks = select_routes(
        baseline,
        candidate,
        args.minimum_case_dice_gain,
        args.minimum_cohort_dice_gain,
        args.probability_tolerance,
        args.expected_cases,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "selected_strategy": "source_routed",
        "selection_basis": (
            "exact grouped held-out patch OOF; each disjoint source route must "
            "improve case- and cohort-balanced contour Dice"
        ),
        "classification_checkpoint_filename": "checkpoint_best_classification.pt",
        "contour_checkpoint_filenames": selection[
            "contour_checkpoint_filenames"
        ],
        "contour_checkpoint_run_families": {
            name: "primary"
            for name in selection["contour_checkpoint_filenames"]
        },
        "source_contour_routes": selection["source_to_route"],
        "selected_units": selection["selected_units"],
        "fallback_units": selection["fallback_units"],
        "unit_reports": selection["unit_reports"],
        "checks": checks,
        "thresholds": {
            "minimum_case_dice_gain": args.minimum_case_dice_gain,
            "minimum_cohort_dice_gain": args.minimum_cohort_dice_gain,
            "probability_tolerance": args.probability_tolerance,
            "expected_cases": args.expected_cases,
        },
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": sha256(args.baseline),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
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
