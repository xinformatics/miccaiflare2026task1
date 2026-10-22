#!/usr/bin/env python3
"""Compare fold-4 complete xy flip-group TTA against three-view xy TTA."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


GROUPS = ("weak_sources", "lung_cells")
MIN_CASE_GAIN = 0.001
MIN_COHORT_GAIN = 0.001
MIN_SELECTION_GAIN = 0.0005
MAX_NEGATIVE_INCREASE = 0.02


def select(xy: dict, route: dict, audits: list[dict]) -> dict[str, object]:
    for name, payload in (("xy", xy), ("route", route), *[("xy4", audit) for audit in audits]):
        if any(bool(payload.get(key)) for key in ("public_read", "hidden_read", "submission_created")):
            raise ValueError(f"{name} input is not local-only")
    if xy.get("decision") != "PROMOTE_TTA_TO_FIVEFOLD":
        raise ValueError("three-view xy pilot did not qualify")
    selected = route.get("selected", {})
    if not set(GROUPS) <= set(selected):
        raise ValueError("fold-4 route selection is incomplete")
    if int(route.get("fold", -1)) != 4:
        raise ValueError("route selection is not fold 4")
    route_seed = int(route.get("seed", -1))
    route_index_sha256 = str(route.get("index_sha256", ""))
    if route_seed < 0 or not route_index_sha256:
        raise ValueError("route selection identity is incomplete")
    by_epoch = {}
    for audit in audits:
        if int(audit.get("fold", -1)) != 4:
            raise ValueError("xy4 audit is not fold 4")
        if (
            int(audit.get("seed", -1)) != route_seed
            or str(audit.get("index_sha256", "")) != route_index_sha256
        ):
            raise ValueError("xy4 audit seed/index identity mismatch")
        epoch = int(audit["epoch"])
        if epoch in by_epoch:
            raise ValueError("duplicate xy4 epoch audit")
        by_epoch[epoch] = audit

    reports = {}
    promoted = []
    for group in GROUPS:
        epoch = int(selected[group]["epoch"])
        if int(xy.get("groups", {}).get(group, {}).get("epoch", -1)) != epoch:
            raise ValueError(f"three-view xy epoch mismatch for {group}")
        if epoch not in by_epoch:
            raise ValueError(f"missing xy4 audit for {group} epoch {epoch}")
        audit = by_epoch[epoch]
        if str(audit.get("checkpoint_sha256", "")) != str(
            selected[group].get("checkpoint_sha256", "")
        ):
            raise ValueError(f"xy4 checkpoint identity mismatch for {group}")
        candidate = audit["groups"][group]
        anchor = xy["groups"][group]["tta_xy"]
        gains = {
            "case_balanced_dice": float(candidate["case_balanced_dice"])
            - float(anchor["case_balanced_dice"]),
            "cohort_balanced_dice": float(candidate["cohort_balanced_dice"])
            - float(anchor["cohort_balanced_dice"]),
            "selection_score": float(candidate["selection_score"])
            - float(anchor["selection_score"]),
            "negative_nonempty_rate": float(candidate["negative_nonempty_rate"])
            - float(anchor["negative_nonempty_rate"]),
        }
        checks = {
            "case_gain": gains["case_balanced_dice"] >= MIN_CASE_GAIN,
            "cohort_gain": gains["cohort_balanced_dice"] >= MIN_COHORT_GAIN,
            "selection_gain": gains["selection_score"] >= MIN_SELECTION_GAIN,
            "negative_increase": gains["negative_nonempty_rate"] <= MAX_NEGATIVE_INCREASE,
        }
        qualifies = all(checks.values())
        if qualifies:
            promoted.append(group)
        reports[group] = {
            "epoch": epoch,
            "xy_three_view": anchor,
            "xy_four_view": candidate,
            "gains": gains,
            "checks": checks,
            "qualified_for_fivefold": qualifies,
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "held-out fold-4 complete xy flip group versus frozen three-view xy TTA",
        "decision": "PROMOTE_XY4_TO_FIVEFOLD" if promoted else "KEEP_XY3",
        "eligible_groups": promoted,
        "groups": reports,
        "thresholds": {
            "minimum_case_gain": MIN_CASE_GAIN,
            "minimum_cohort_gain": MIN_COHORT_GAIN,
            "minimum_selection_gain": MIN_SELECTION_GAIN,
            "maximum_negative_nonempty_increase": MAX_NEGATIVE_INCREASE,
        },
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xy-decision", type=Path, required=True)
    parser.add_argument("--route-selection", type=Path, required=True)
    parser.add_argument("--xy4-audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = select(
        json.loads(args.xy_decision.read_text(encoding="utf-8")),
        json.loads(args.route_selection.read_text(encoding="utf-8")),
        [json.loads(path.read_text(encoding="utf-8")) for path in args.xy4_audit],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
