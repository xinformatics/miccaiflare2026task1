#!/usr/bin/env python3
"""Select a clearly labelled, non-primary Focus-TTA fallback.

The original TTA experiment keeps its frozen, conservative +0.015/+0.015
promotion rule.  This selector does not alter that decision.  It records
whether a TTA result clears the weaker but still paired and uncertainty-aware
evidence floor used by the exact-707 full-volume experiments.  A passing result
may be materialized locally as an exploratory fallback, but must not be called
the primary promoted model or uploaded automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_VARIANT = "tta_xy"
EXPECTED_CASES = 50

# Re-running the same CUDA model can move a few threshold-edge voxels.  These
# bounds are deliberately expressed both as aggregate metric drift and as a
# tiny fraction of the deployed foreground, so they cannot hide a recipe
# change.  The historical metric CSV is rounded to four decimals.
EQUIVALENCE_LIMITS = {
    "max_case_metric_delta": 0.001,
    "max_case_voxel_delta": 128,
    "max_total_abs_voxel_delta": 256,
    "max_total_abs_voxel_fraction": 5e-5,
    "max_mean_metric_delta": 1e-4,
}

EVIDENCE_LIMITS = {
    "minimum_dsc_gain": 0.005,
    "minimum_nsd_gain": 0.005,
    "minimum_win_rate": 0.55,
    "maximum_catastrophic_rate": 0.0,
    "maximum_runtime_seconds": 55.0,
}


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric value for {name}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value for {name}")
    return result


def _equivalence_checks(report: dict[str, Any]) -> dict[str, bool]:
    if int(report.get("cases", 0)) != EXPECTED_CASES:
        raise ValueError("deployment equivalence report is not exact-50")
    metric_delta = max(
        _finite_number(report.get("max_case_dsc_delta"), "max_case_dsc_delta"),
        _finite_number(report.get("max_case_nsd_delta"), "max_case_nsd_delta"),
    )
    mean_delta = max(
        _finite_number(report.get("mean_dsc_delta"), "mean_dsc_delta"),
        _finite_number(report.get("mean_nsd_delta"), "mean_nsd_delta"),
    )
    return {
        "case_metric_drift": metric_delta <= EQUIVALENCE_LIMITS["max_case_metric_delta"],
        "case_voxel_drift": int(report.get("max_case_voxel_delta", -1))
        <= EQUIVALENCE_LIMITS["max_case_voxel_delta"],
        "total_voxel_drift": int(report.get("total_abs_voxel_delta", -1))
        <= EQUIVALENCE_LIMITS["max_total_abs_voxel_delta"],
        "foreground_fraction_drift": _finite_number(
            report.get("total_abs_voxel_fraction"), "total_abs_voxel_fraction"
        )
        <= EQUIVALENCE_LIMITS["max_total_abs_voxel_fraction"],
        "mean_metric_drift": mean_delta <= EQUIVALENCE_LIMITS["max_mean_metric_delta"],
    }


def _robust_checks(report: dict[str, Any], expected_anchor: str) -> dict[str, bool]:
    if report.get("anchor") != expected_anchor or report.get("candidate") != EXPECTED_VARIANT:
        raise ValueError(f"unexpected paired report identity for {expected_anchor}")
    if int(report.get("cases", 0)) != EXPECTED_CASES:
        raise ValueError(f"paired report for {expected_anchor} is not exact-50")
    gains = report.get("paired_gain", {})
    intervals = report.get("bootstrap_95ci", {})
    dsc_ci = intervals.get("Lesion_DSC", [])
    nsd_ci = intervals.get("Lesion_NSD", [])
    combined_ci = intervals.get("combined", [])
    if any(not isinstance(interval, list) or len(interval) != 2 for interval in (dsc_ci, nsd_ci, combined_ci)):
        raise ValueError(f"incomplete bootstrap intervals for {expected_anchor}")
    return {
        "dsc_gain": _finite_number(gains.get("Lesion_DSC"), "DSC gain")
        >= EVIDENCE_LIMITS["minimum_dsc_gain"],
        "nsd_gain": _finite_number(gains.get("Lesion_NSD"), "NSD gain")
        >= EVIDENCE_LIMITS["minimum_nsd_gain"],
        "dsc_ci_positive": _finite_number(dsc_ci[0], "DSC CI lower") > 0,
        "nsd_ci_positive": _finite_number(nsd_ci[0], "NSD CI lower") > 0,
        "combined_ci_positive": _finite_number(combined_ci[0], "combined CI lower") > 0,
        "win_rate": _finite_number(report.get("win_rate"), "win rate")
        >= EVIDENCE_LIMITS["minimum_win_rate"],
        "tail_safety": _finite_number(
            report.get("catastrophic_regression_rate"), "catastrophic regression rate"
        )
        <= EVIDENCE_LIMITS["maximum_catastrophic_rate"],
    }


def _runtime_checks(rows: list[dict[str, str]]) -> tuple[dict[str, bool], dict[str, str]]:
    matches = [row for row in rows if row.get("variant") == EXPECTED_VARIANT]
    if len(matches) != 1:
        raise ValueError("runtime table must contain tta_xy exactly once")
    row = matches[0]
    maximum = _finite_number(row.get("max_seconds"), "maximum runtime")
    checks = {
        "runtime_declared_eligible": row.get("runtime_eligible") == "yes",
        "runtime_below_challenge_limit": maximum <= EVIDENCE_LIMITS["maximum_runtime_seconds"],
    }
    return checks, row


def select_evidence_floor(
    equivalence: dict[str, Any],
    versus_notta: dict[str, Any],
    versus_champion: dict[str, Any],
    runtime_rows: list[dict[str, str]],
) -> dict[str, Any]:
    equivalence_checks = _equivalence_checks(equivalence)
    notta_checks = _robust_checks(versus_notta, "notta")
    champion_checks = _robust_checks(versus_champion, "gf_champ_new2k_s25")
    runtime_checks, runtime_row = _runtime_checks(runtime_rows)
    qualified = all(
        list(equivalence_checks.values())
        + list(notta_checks.values())
        + list(champion_checks.values())
        + list(runtime_checks.values())
    )
    return {
        "decision": "SAVE_EXPLORATORY_TTA_FALLBACK" if qualified else "KEEP_CHAMPION",
        "selected_variant": EXPECTED_VARIANT if qualified else None,
        "status_class": "POST_HOC_EVIDENCE_FLOOR_NOT_PRIMARY_PROMOTION",
        "checks": {
            "numerical_deployment_equivalence": equivalence_checks,
            "paired_vs_rerun_notta": notta_checks,
            "paired_vs_historical_champion": champion_checks,
            "runtime": runtime_checks,
        },
        "limits": {
            "numerical_deployment_equivalence": EQUIVALENCE_LIMITS,
            "paired_evidence": EVIDENCE_LIMITS,
        },
        "runtime_row": runtime_row,
        "automatic_upload_authorized": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equivalence", type=Path, required=True)
    parser.add_argument("--versus-notta", type=Path, required=True)
    parser.add_argument("--versus-champion", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = (args.equivalence, args.versus_notta, args.versus_champion, args.runtime)
    if any(not path.is_file() for path in inputs):
        raise SystemExit("missing TTA evidence input")
    with args.runtime.open(newline="", encoding="utf-8") as stream:
        runtime_rows = list(csv.DictReader(stream, delimiter="\t"))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": (
            "post-hoc exploratory fallback using the pre-existing fixed public TTA audit; "
            "the original strict +0.015 promotion decision remains unchanged"
        ),
        **select_evidence_floor(
            json.loads(args.equivalence.read_text(encoding="utf-8")),
            json.loads(args.versus_notta.read_text(encoding="utf-8")),
            json.loads(args.versus_champion.read_text(encoding="utf-8")),
            runtime_rows,
        ),
        "inputs": {
            path.stem: {"path": str(path.resolve()), "sha256": _sha256(path)} for path in inputs
        },
        "public_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
