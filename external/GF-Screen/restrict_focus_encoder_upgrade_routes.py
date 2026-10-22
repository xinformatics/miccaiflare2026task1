#!/usr/bin/env python3
"""Restrict fivefold encoder promotion to routes admitted by the pilot gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


ROUTES = {"weak_sources", "lung_cells"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bound_payload(terminal: dict, path_key: str, hash_key: str) -> tuple[dict, Path]:
    path = Path(str(terminal.get(path_key, "")))
    expected = str(terminal.get(hash_key, ""))
    if not path.is_file() or not expected or sha256(path) != expected:
        raise ValueError(f"terminal does not bind {path_key}")
    return json.loads(path.read_text(encoding="utf-8")), path


def restrict(fivefold_terminal: dict, pilot_terminal: dict) -> dict:
    fivefold, fivefold_path = bound_payload(
        fivefold_terminal, "fivefold_decision", "fivefold_decision_sha256"
    )
    pilot, pilot_path = bound_payload(
        pilot_terminal, "route_decision", "route_decision_sha256"
    )
    if (
        fivefold_terminal.get("status") != "COMPLETE"
        or fivefold_terminal.get("decision") != fivefold.get("decision")
        or pilot_terminal.get("status") != "COMPLETE"
        or pilot_terminal.get("decision") != pilot.get("decision")
        or any(
            payload.get(key)
            for payload in (fivefold, pilot, fivefold_terminal, pilot_terminal)
            for key in ("public_read", "hidden_read", "submission_created")
        )
    ):
        raise ValueError("route terminals are not matched local-only decisions")

    pilot_routes = set(pilot.get("qualified_routes", []))
    upstream_routes = set(fivefold.get("qualified_routes", []))
    if (
        pilot.get("decision") != "PROMOTE_ROUTES_TO_FIVEFOLD"
        or not pilot_routes
        or not pilot_routes <= ROUTES
        or not upstream_routes <= ROUTES
        or set(fivefold.get("routes", {})) != ROUTES
    ):
        raise ValueError("invalid pilot or fivefold route inventory")

    qualified = sorted(pilot_routes & upstream_routes)
    payload = copy.deepcopy(fivefold)
    for route, report in payload["routes"].items():
        report["pilot_qualified"] = route in pilot_routes
        report["upstream_fivefold_qualified"] = route in upstream_routes
        report["qualified_for_full_volume"] = route in qualified
        if route not in pilot_routes:
            report["restriction_reason"] = "route did not pass the predeclared fold-0 pilot gate"
        elif route not in upstream_routes:
            report["restriction_reason"] = "route did not pass the pooled fivefold gate"
        else:
            report["restriction_reason"] = "passed pilot and pooled fivefold gates"
    payload.update(
        {
            "schema_version": 3,
            "created_at": datetime.now().astimezone().isoformat(),
            "selection_basis": (
                "intersection of predeclared fold-0 pilot routes and exact pooled "
                "fivefold route qualification"
            ),
            "decision": (
                "PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF"
                if qualified
                else "KEEP_FROZEN_FOCUS"
            ),
            "passed": bool(qualified),
            "qualified_routes": qualified,
            "pilot_qualified_routes": sorted(pilot_routes),
            "upstream_fivefold_qualified_routes": sorted(upstream_routes),
            "pilot_decision": str(pilot_path.resolve()),
            "pilot_decision_sha256": sha256(pilot_path),
            "upstream_fivefold_decision": str(fivefold_path.resolve()),
            "upstream_fivefold_decision_sha256": sha256(fivefold_path),
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fivefold-terminal", type=Path, required=True)
    parser.add_argument("--pilot-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fivefold_terminal = json.loads(args.fivefold_terminal.read_text(encoding="utf-8"))
    pilot_terminal = json.loads(args.pilot_terminal.read_text(encoding="utf-8"))
    payload = restrict(fivefold_terminal, pilot_terminal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
