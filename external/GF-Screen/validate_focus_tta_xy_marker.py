#!/usr/bin/env python3
"""Validate a route-selective Focus xy-TTA inference marker and cache plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from apply_candidate_refiner_focus_tta_xy import (
    EXPECTED_DEPLOYMENT_SHA256,
    TTA_MARKER_NAME,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate(work: Path, expected_routes: set[str]) -> dict[str, bool]:
    marker_path = work / TTA_MARKER_NAME
    plan_path = work / "INFERENCE_PLAN.json"
    inference_path = work / "INFERENCE_COMPLETE.json"
    for path in (marker_path, plan_path, inference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    wrapper = Path(__file__).with_name(
        "apply_candidate_refiner_focus_tta_xy.py"
    ).resolve()
    deployment = wrapper.with_name("apply_candidate_refiner.py")
    implementation = plan.get("identity", {}).get("implementation", [])
    checks = {
        "complete": marker.get("status") == "COMPLETE",
        "axes": marker.get("tta_axes") == "xy"
        and int(marker.get("tta_views", -1)) == 3,
        "routes": set(marker.get("tta_focus_routes", [])) == expected_routes
        and bool(expected_routes)
        and all(name == "global" or name.endswith("_tta_xy") for name in expected_routes),
        "local_only": not any(
            bool(marker.get(key))
            for key in ("public_read", "hidden_read", "submission_created")
        ),
        "plan_path": marker.get("inference_plan") == str(plan_path.resolve()),
        "plan_hash": marker.get("inference_plan_sha256") == sha256(plan_path),
        "inference_path": marker.get("inference_complete")
        == str(inference_path.resolve()),
        "inference_hash": marker.get("inference_complete_sha256")
        == sha256(inference_path),
        "fingerprint": bool(marker.get("inference_fingerprint"))
        and marker.get("inference_fingerprint") == plan.get("inference_fingerprint")
        == inference.get("inference_fingerprint"),
        "wrapper": marker.get("implementation") == str(wrapper)
        and marker.get("implementation_sha256") == sha256(wrapper)
        and isinstance(implementation, list)
        and bool(implementation)
        and implementation[0].get("path") == str(wrapper)
        and implementation[0].get("sha256") == sha256(wrapper),
        "deployment": marker.get("underlying_deployment") == str(deployment)
        and marker.get("underlying_deployment_sha256")
        == EXPECTED_DEPLOYMENT_SHA256
        and sha256(deployment) == EXPECTED_DEPLOYMENT_SHA256,
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--tta-route", action="append", default=[])
    args = parser.parse_args()
    checks = validate(args.work, set(args.tta_route))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Focus xy-TTA marker validation failed: {failed}")
    print(json.dumps({"work": str(args.work.resolve()), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
