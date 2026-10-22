#!/usr/bin/env python3
"""Replace qualified frozen Focus routes with encoder-adapted routes."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from summarize_focus_epoch_routes import sha256


GROUPS = {"mature", "weak_sources", "lung_cells"}
DEPLOYED_FOCUS_GROUPS = {"weak_sources", "lung_cells"}


def local_only(name: str, payload: dict) -> None:
    if any(
        bool(payload.get(key))
        for key in ("public_read", "hidden_read", "submission_created")
    ):
        raise ValueError(f"{name} is not local-only")


def checkpoint_route(group: str, selected: dict, use_tta: bool, input_mode: str) -> tuple[str, dict]:
    checkpoint = Path(str(selected["checkpoint"]))
    expected_hash = str(selected["checkpoint_sha256"])
    if not checkpoint.is_file() or sha256(checkpoint) != expected_hash:
        raise ValueError(f"encoder-upgrade {group} checkpoint changed")
    route_name = f"focus_encoder01_{group}_{'tta_xy' if use_tta else 'notta'}"
    return route_name, {
        "family": "gf_focus",
        "variant": "encoder_lr_scale_0.1",
        "group": group,
        "epoch": int(selected["epoch"]),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": expected_hash,
        "tta_axes": "xy" if use_tta else "",
        "tta_views": 3 if use_tta else 1,
        "input_mode": input_mode,
    }


def compose(base: dict, candidate: dict, upgrade: dict, tta: dict) -> dict[str, object]:
    for name, payload in (
        ("base composite", base),
        ("candidate routes", candidate),
        ("encoder decision", upgrade),
        ("encoder TTA", tta),
    ):
        local_only(name, payload)
    if base.get("decision") not in {
        "QUALIFY_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME",
        "QUALIFY_COMPOSITE_FOCUS_B32_NOTTA_FULL_VOLUME",
    }:
        raise ValueError("base B32/Focus composite did not qualify")
    index_hash = str(candidate.get("index_sha256", ""))
    if (
        not index_hash
        or base.get("index_sha256") != index_hash
        or int(candidate.get("folds", -1)) != 5
        or int(base.get("folds", -1)) != 5
        or candidate.get("training_recipe", {}).get("freeze_encoder") is not False
        or float(candidate.get("training_recipe", {}).get("encoder_lr_scale", -1.0))
        != 0.1
    ):
        raise ValueError("base and encoder candidate identities do not align")
    qualified = set(upgrade.get("qualified_routes", []))
    if (
        upgrade.get("decision")
        != "PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF"
        or not upgrade.get("passed")
        or not qualified
        or not qualified <= DEPLOYED_FOCUS_GROUPS
        or upgrade.get("mature_route") != "B32_UNCHANGED"
    ):
        raise ValueError("encoder-upgrade route decision did not qualify")
    tta_qualified = set(tta.get("qualified_groups", []))
    if (
        tta.get("decision") not in {"QUALIFY_ROUTED_TTA_FULL_VOLUME", "KEEP_NOTTA"}
        or not tta_qualified <= qualified
        or (tta.get("decision") == "KEEP_NOTTA" and tta_qualified)
    ):
        raise ValueError("encoder-upgrade TTA decision is invalid")
    source_groups = candidate.get("source_to_route", {})
    if set(source_groups.values()) != GROUPS or set(source_groups) != set(base.get("source_to_route", {})):
        raise ValueError("candidate/base source inventories do not align")
    if set(candidate.get("source_segmentation_groups", {})) != GROUPS:
        raise ValueError("candidate route calibration coverage mismatch")

    source_to_route: dict[str, str] = {}
    source_thresholds: dict[str, float] = {}
    for source, group in sorted(source_groups.items()):
        if group in qualified:
            use_tta = group in tta_qualified
            route_name = f"focus_encoder01_{group}_{'tta_xy' if use_tta else 'notta'}"
            if use_tta:
                threshold = float(tta["groups"][group]["tta_xy"]["threshold"])
            else:
                threshold = float(candidate["source_segmentation_groups"][group]["threshold"])
        else:
            route_name = str(base["source_to_route"][source])
            threshold = float(base["source_segmentation_thresholds"][source])
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"invalid composed threshold for {source}")
        source_to_route[source] = route_name
        source_thresholds[source] = threshold

    candidate_records = {
        int(record["fold"]): record for record in candidate.get("fold_records", [])
    }
    base_records = {int(record["fold"]): record for record in base.get("fold_records", [])}
    if set(candidate_records) != set(range(5)) or set(base_records) != set(range(5)):
        raise ValueError("composite fold coverage mismatch")
    fold_records = []
    for fold in range(5):
        base_routes = base_records[fold]["routes"]
        routes: dict[str, dict] = {}
        used_names = set(source_to_route.values())
        for name, details in base_routes.items():
            if name in used_names:
                if details.get("family") == "gf_focus":
                    checkpoint = Path(str(details["checkpoint"]))
                    if not checkpoint.is_file() or sha256(checkpoint) != details.get(
                        "checkpoint_sha256"
                    ):
                        raise ValueError(f"retained fold-{fold} Focus checkpoint changed")
                routes[name] = details
        selected = candidate_records[fold].get("selected", {})
        if set(selected) != GROUPS:
            raise ValueError("candidate selected-route fold coverage mismatch")
        for group in sorted(qualified):
            name, details = checkpoint_route(
                group,
                selected[group],
                group in tta_qualified,
                str(candidate["training_recipe"]["input_mode"]),
            )
            if name not in used_names:
                raise ValueError("candidate route is not referenced by a source")
            routes[name] = details
        if set(routes) != used_names:
            raise ValueError(f"fold-{fold} route materialization coverage mismatch")
        fold_records.append({"fold": fold, "routes": routes})

    applied_tta = sorted(
        {
            source_groups[source]
            for source, route_name in source_to_route.items()
            if route_name.endswith("_tta_xy")
        }
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": (
            "qualified frozen B32/Focus composite with exact five-fold "
            "encoder-route replacement and separately requalified x/y TTA"
        ),
        "decision": (
            "QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME"
            if applied_tta
            else "QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_NOTTA_FULL_VOLUME"
        ),
        "index_sha256": index_hash,
        "folds": 5,
        "focus_input_mode": str(candidate["training_recipe"]["input_mode"]),
        "encoder_upgraded_groups": sorted(qualified),
        "encoder_tta_groups": sorted(tta_qualified),
        "tta_groups": applied_tta,
        "b32_groups": list(base.get("b32_groups", [])),
        "source_to_route": source_to_route,
        "source_segmentation_thresholds": source_thresholds,
        "fold_records": fold_records,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-composite", type=Path, required=True)
    parser.add_argument("--candidate-routes", type=Path, required=True)
    parser.add_argument("--upgrade-decision", type=Path, required=True)
    parser.add_argument("--candidate-tta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_composite.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_routes.read_text(encoding="utf-8"))
    upgrade = json.loads(args.upgrade_decision.read_text(encoding="utf-8"))
    tta = json.loads(args.candidate_tta.read_text(encoding="utf-8"))
    if (
        upgrade.get("candidate_summary_sha256") != sha256(args.candidate_routes)
        or Path(str(upgrade.get("candidate_summary", ""))).resolve()
        != args.candidate_routes.resolve()
        or tta.get("route_summary_sha256") != sha256(args.candidate_routes)
        or Path(str(tta.get("route_summary", ""))).resolve()
        != args.candidate_routes.resolve()
    ):
        raise ValueError("encoder route decisions are not bound to the candidate")
    payload = compose(base, candidate, upgrade, tta)
    payload.update(
        {
            "base_composite": str(args.base_composite.resolve()),
            "base_composite_sha256": sha256(args.base_composite),
            "candidate_routes": str(args.candidate_routes.resolve()),
            "candidate_routes_sha256": sha256(args.candidate_routes),
            "upgrade_decision": str(args.upgrade_decision.resolve()),
            "upgrade_decision_sha256": sha256(args.upgrade_decision),
            "candidate_tta": str(args.candidate_tta.resolve()),
            "candidate_tta_sha256": sha256(args.candidate_tta),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
