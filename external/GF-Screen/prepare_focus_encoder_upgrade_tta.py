#!/usr/bin/env python3
"""Bind encoder-upgrade route eligibility before exact five-fold TTA."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from summarize_focus_epoch_routes import sha256


DEPLOYED_ROUTES = {"weak_sources", "lung_cells"}


def prepare(route_path: Path, decision_path: Path) -> dict[str, object]:
    route = json.loads(route_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        route.get("public_read")
        or route.get("hidden_read")
        or route.get("submission_created")
        or decision.get("public_read")
        or decision.get("hidden_read")
        or decision.get("submission_created")
    ):
        raise ValueError("encoder-upgrade TTA eligibility must be local-only")
    if (
        decision.get("decision")
        != "PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF"
        or not decision.get("passed")
        or decision.get("candidate_summary_sha256") != sha256(route_path)
        or Path(str(decision.get("candidate_summary", ""))).resolve()
        != route_path.resolve()
        or decision.get("mature_route") != "B32_UNCHANGED"
    ):
        raise ValueError("encoder-upgrade decision is not bound to the route summary")
    eligible = list(decision.get("qualified_routes", []))
    if not eligible or len(eligible) != len(set(eligible)) or not set(eligible) <= DEPLOYED_ROUTES:
        raise ValueError("invalid encoder-upgrade TTA route eligibility")
    if (
        int(route.get("folds", -1)) != 5
        or int(route.get("seed", -1)) != 20260823
        or int(route.get("expected_epochs", -1)) != 11
        or route.get("training_recipe", {}).get("freeze_encoder") is not False
        or float(route.get("training_recipe", {}).get("encoder_lr_scale", -1.0))
        != 0.1
    ):
        raise ValueError("route summary is not the matched encoder-0.1 five-fold run")
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": (
            "exact five-fold no-TTA encoder route qualification; TTA itself "
            "remains unobserved when eligibility is frozen"
        ),
        # Retain the interface consumed by summarize_focus_routed_tta.py.
        "decision": "PROMOTE_TTA_TO_FIVEFOLD",
        "eligible_groups": eligible,
        "route_summary": str(route_path.resolve()),
        "route_summary_sha256": sha256(route_path),
        "encoder_upgrade_decision": str(decision_path.resolve()),
        "encoder_upgrade_decision_sha256": sha256(decision_path),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-summary", type=Path, required=True)
    parser.add_argument("--upgrade-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(args.route_summary, args.upgrade_decision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
