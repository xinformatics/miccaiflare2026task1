#!/usr/bin/env python3
"""Gate selected Focus source routes for a full five-fold TTA audit."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


PILOT_GROUPS = ("weak_sources", "lung_cells")
MIN_CASE_GAIN = 0.002
MIN_COHORT_GAIN = 0.002
MIN_SELECTION_GAIN = 0.001
MAX_NEGATIVE_INCREASE = 0.03


def select_tta(route_path: Path, tta_audits: list[Path]) -> dict[str, object]:
    route = json.loads(route_path.read_text(encoding="utf-8"))
    if (
        route.get("public_read")
        or route.get("hidden_read")
        or route.get("submission_created")
        or set(route.get("selected", {})) != {"mature", *PILOT_GROUPS}
    ):
        raise ValueError("invalid local-only Focus route selection")
    by_epoch = {}
    for path in tta_audits:
        payload = json.loads(path.read_text(encoding="utf-8"))
        epoch = int(payload["epoch"])
        if epoch in by_epoch:
            raise ValueError("duplicate TTA epoch audit")
        if (
            int(payload["fold"]) != int(route["fold"])
            or int(payload["seed"]) != int(route["seed"])
            or payload["index_sha256"] != route["index_sha256"]
            or payload.get("public_read")
            or payload.get("hidden_read")
            or payload.get("submission_created")
        ):
            raise ValueError("TTA audit identity mismatch")
        by_epoch[epoch] = (path, payload)

    reports = {}
    eligible = []
    for group in PILOT_GROUPS:
        baseline = route["selected"][group]
        epoch = int(baseline["epoch"])
        if epoch not in by_epoch:
            raise ValueError(f"missing selected epoch-{epoch} TTA audit")
        path, tta = by_epoch[epoch]
        if tta["checkpoint_sha256"] != baseline["checkpoint_sha256"]:
            raise ValueError("TTA checkpoint does not match selected no-TTA route")
        before = baseline["metrics"]
        after = tta["groups"][group]
        gains = {
            "case_balanced_dice": float(after["case_balanced_dice"])
            - float(before["case_balanced_dice"]),
            "cohort_balanced_dice": float(after["cohort_balanced_dice"])
            - float(before["cohort_balanced_dice"]),
            "selection_score": float(after["selection_score"])
            - float(before["selection_score"]),
            "negative_nonempty_rate": float(after["negative_nonempty_rate"])
            - float(before["negative_nonempty_rate"]),
        }
        checks = {
            "case_gain": gains["case_balanced_dice"] >= MIN_CASE_GAIN,
            "cohort_gain": gains["cohort_balanced_dice"] >= MIN_COHORT_GAIN,
            "selection_gain": gains["selection_score"] >= MIN_SELECTION_GAIN,
            "negative_increase": gains["negative_nonempty_rate"]
            <= MAX_NEGATIVE_INCREASE,
        }
        qualified = all(checks.values())
        if qualified:
            eligible.append(group)
        reports[group] = {
            "epoch": epoch,
            "baseline": before,
            "tta_xy": after,
            "gains": gains,
            "checks": checks,
            "qualified_for_fivefold": qualified,
            "tta_audit": str(path.resolve()),
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "one held-out-fold source-routed x+y flip-TTA pilot",
        "decision": "PROMOTE_TTA_TO_FIVEFOLD" if eligible else "KEEP_NOTTA",
        "eligible_groups": eligible,
        "thresholds": {
            "minimum_case_gain": MIN_CASE_GAIN,
            "minimum_cohort_gain": MIN_COHORT_GAIN,
            "minimum_selection_gain": MIN_SELECTION_GAIN,
            "maximum_negative_nonempty_increase": MAX_NEGATIVE_INCREASE,
        },
        "groups": reports,
        "route_selection": str(route_path.resolve()),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-selection", type=Path, required=True)
    parser.add_argument("--tta-audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = select_tta(args.route_selection, args.tta_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
