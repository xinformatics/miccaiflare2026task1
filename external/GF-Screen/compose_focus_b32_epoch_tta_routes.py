#!/usr/bin/env python3
"""Compose the strongest locally qualified B32/Focus deployment policy.

The mixed-capacity selector decides *which architecture* owns each proposal
family.  The epoch-route summary independently selects the best held-out Focus
state for each source group and fold.  Finally, the routed-TTA audit decides
which Focus groups may use x/y probability averaging.  This module joins those
orthogonal decisions without reading public or hidden cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


GROUPS = {"mature", "weak_sources", "lung_cells"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _local_only(name: str, payload: dict) -> None:
    if any(
        bool(payload.get(key))
        for key in ("public_read", "hidden_read", "submission_created")
    ):
        raise ValueError(f"{name} is not a local-only decision")


def compose(epoch: dict, mixed: dict, tta: dict) -> dict[str, object]:
    for name, payload in (("epoch route", epoch), ("mixed route", mixed), ("TTA", tta)):
        _local_only(name, payload)
    if int(epoch.get("folds", -1)) != 5:
        raise ValueError("Focus epoch routing is not exact five-fold OOF")
    index_hash = str(epoch.get("index_sha256", ""))
    if not index_hash or mixed.get("index_sha256") != index_hash:
        raise ValueError("mixed and epoch routes use different patch indices")
    if mixed.get("decision") != "QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME":
        raise ValueError("mixed B32/Focus architecture route did not qualify")
    epoch_source_map = epoch.get("source_to_route", {})
    mixed_source_map = mixed.get("source_to_route", {})
    if (
        not isinstance(epoch_source_map, dict)
        or set(epoch_source_map.values()) != GROUPS
        or set(mixed_source_map) != set(epoch_source_map)
        or set(mixed_source_map.values()) != {"b32", "focus"}
    ):
        raise ValueError("architecture/epoch source inventories do not align")
    if set(mixed.get("groups", {})) != GROUPS:
        raise ValueError("mixed architecture group coverage mismatch")

    tta_decision = str(tta.get("decision", ""))
    if tta_decision not in {"QUALIFY_ROUTED_TTA_FULL_VOLUME", "KEEP_NOTTA"}:
        raise ValueError("routed TTA decision is not terminal")
    qualified_tta = set(tta.get("qualified_groups", []))
    if not qualified_tta <= {"weak_sources", "lung_cells"}:
        raise ValueError("invalid TTA-qualified source groups")
    if tta_decision == "KEEP_NOTTA" and qualified_tta:
        raise ValueError("KEEP_NOTTA cannot contain qualified groups")
    source_to_tta = tta.get("source_to_tta", {})
    if set(source_to_tta) != set(epoch_source_map):
        raise ValueError("TTA and epoch source inventories do not align")
    for source, group in epoch_source_map.items():
        expected = group in qualified_tta
        if bool(source_to_tta[source]) != expected:
            raise ValueError("TTA source flags disagree with qualified groups")

    recipe = epoch.get("training_recipe", {})
    input_mode = str(recipe.get("input_mode", ""))
    if input_mode not in {"soft", "soft_seed", "soft_lung_seed"}:
        raise ValueError("invalid routed Focus input mode")
    epoch_groups = epoch.get("source_segmentation_groups", {})
    tta_groups = tta.get("groups", {})
    if set(epoch_groups) != GROUPS:
        raise ValueError("epoch route group calibration mismatch")

    source_to_route: dict[str, str] = {}
    source_thresholds: dict[str, float] = {}
    used_focus_groups: set[str] = set()
    for source, group in sorted(epoch_source_map.items()):
        family = mixed_source_map[source]
        if family == "b32":
            route = "b32"
            threshold = float(mixed["source_segmentation_thresholds"][source])
        else:
            used_focus_groups.add(group)
            use_tta = group in qualified_tta
            route = f"focus_{group}_{'tta_xy' if use_tta else 'notta'}"
            if use_tta:
                threshold = float(tta_groups[group]["tta_xy"]["threshold"])
            else:
                threshold = float(epoch_groups[group]["threshold"])
        if not 0 < threshold < 1:
            raise ValueError(f"invalid contour threshold for {source}")
        source_to_route[source] = route
        source_thresholds[source] = threshold

    fold_records = epoch.get("fold_records", [])
    if sorted(int(record["fold"]) for record in fold_records) != list(range(5)):
        raise ValueError("epoch route fold coverage mismatch")
    deployment_folds = []
    b32_groups = sorted(
        group
        for group in GROUPS
        if mixed["groups"][group]["selected_family"] == "b32"
    )
    for record in sorted(fold_records, key=lambda value: int(value["fold"])):
        selected = record.get("selected", {})
        if set(selected) != GROUPS:
            raise ValueError("selected Focus epoch group coverage mismatch")
        routes: dict[str, dict[str, object]] = {
            "b32": {"family": "candidate_refiner", "groups": b32_groups}
        }
        for group in sorted(used_focus_groups):
            choice = selected[group]
            checkpoint = Path(str(choice["checkpoint"]))
            recorded_hash = str(choice.get("checkpoint_sha256", ""))
            if not checkpoint.is_file() or sha256(checkpoint) != recorded_hash:
                raise ValueError(f"fold {record['fold']} {group} checkpoint changed")
            use_tta = group in qualified_tta
            name = f"focus_{group}_{'tta_xy' if use_tta else 'notta'}"
            routes[name] = {
                "family": "gf_focus",
                "group": group,
                "epoch": int(choice["epoch"]),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": recorded_hash,
                "tta_axes": "xy" if use_tta else "",
                "tta_views": 3 if use_tta else 1,
                "input_mode": input_mode,
            }
        deployment_folds.append({"fold": int(record["fold"]), "routes": routes})

    applied_tta_groups = sorted(used_focus_groups & qualified_tta)
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": (
            "case-held-out exact707 architecture family gate + per-fold source "
            "epoch routing + pooled five-fold x/y TTA gate"
        ),
        "decision": (
            "QUALIFY_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME"
            if applied_tta_groups
            else "QUALIFY_COMPOSITE_FOCUS_B32_NOTTA_FULL_VOLUME"
        ),
        "index_sha256": index_hash,
        "folds": 5,
        "focus_input_mode": input_mode,
        "focus_groups": sorted(used_focus_groups),
        "b32_groups": b32_groups,
        "tta_groups": applied_tta_groups,
        "source_to_route": source_to_route,
        "source_segmentation_thresholds": source_thresholds,
        "fold_records": deployment_folds,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-routes", type=Path, required=True)
    parser.add_argument("--epoch-gate", type=Path, required=True)
    parser.add_argument("--mixed-route", type=Path, required=True)
    parser.add_argument("--tta-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    epoch = json.loads(args.epoch_routes.read_text(encoding="utf-8"))
    epoch_gate = json.loads(args.epoch_gate.read_text(encoding="utf-8"))
    mixed = json.loads(args.mixed_route.read_text(encoding="utf-8"))
    tta = json.loads(args.tta_decision.read_text(encoding="utf-8"))
    if (
        tta.get("route_summary_sha256") != sha256(args.epoch_routes)
        or Path(str(tta.get("route_summary", ""))).resolve()
        != args.epoch_routes.resolve()
    ):
        raise ValueError("TTA decision is not bound to the supplied epoch routes")
    _local_only("epoch gate", epoch_gate)
    epoch_gate_decision = str(epoch_gate.get("decision", ""))
    if epoch_gate_decision not in {
        "QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME",
        "KEEP_GLOBAL_FOCUS_EPOCHS",
    }:
        raise ValueError("Focus epoch gate is not a terminal local decision")
    if (
        epoch_gate.get("candidate_sha256") != sha256(args.epoch_routes)
        or Path(str(epoch_gate.get("candidate", ""))).resolve()
        != args.epoch_routes.resolve()
    ):
        raise ValueError("epoch gate is not bound to the supplied epoch routes")
    payload = compose(epoch, mixed, tta)
    payload.update(
        {
            "epoch_routes": str(args.epoch_routes.resolve()),
            "epoch_routes_sha256": sha256(args.epoch_routes),
            "epoch_gate": str(args.epoch_gate.resolve()),
            "epoch_gate_sha256": sha256(args.epoch_gate),
            "epoch_gate_decision": epoch_gate_decision,
            "mixed_route": str(args.mixed_route.resolve()),
            "mixed_route_sha256": sha256(args.mixed_route),
            "tta_decision": str(args.tta_decision.resolve()),
            "tta_decision_sha256": sha256(args.tta_decision),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
