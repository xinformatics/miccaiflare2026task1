#!/usr/bin/env python3
"""Emit a hash-validated deployment manifest from a Focus composite decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_focus_epoch_routes import sha256


ALLOWED_DECISIONS = {
    "QUALIFY_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME",
    "QUALIFY_COMPOSITE_FOCUS_B32_NOTTA_FULL_VOLUME",
    "QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME",
    "QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_NOTTA_FULL_VOLUME",
}


def select(payload: dict, expected_sources: set[str]) -> list[tuple[str, str, str, str]]:
    if (
        payload.get("decision") not in ALLOWED_DECISIONS
        or int(payload.get("folds", -1)) != 5
        or payload.get("public_read")
        or payload.get("hidden_read")
        or payload.get("submission_created")
    ):
        raise ValueError("composite route is not a qualified local-only decision")
    input_mode = str(payload.get("focus_input_mode", ""))
    if input_mode not in {"soft", "soft_seed", "soft_lung_seed"}:
        raise ValueError("invalid composite Focus input mode")
    source_to_route = payload.get("source_to_route", {})
    source_thresholds = payload.get("source_segmentation_thresholds", {})
    if set(source_to_route) != expected_sources or set(source_thresholds) != expected_sources:
        raise ValueError("composite source inventory mismatch")
    if "b32" not in set(source_to_route.values()):
        raise ValueError("composite deployment must retain the B32 route")
    for source, threshold in source_thresholds.items():
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError(f"invalid contour threshold for {source}")

    records = {int(record["fold"]): record for record in payload.get("fold_records", [])}
    if set(records) != set(range(5)):
        raise ValueError("composite fold coverage mismatch")
    lines: list[tuple[str, str, str, str]] = [("INPUT", "-", "-", input_mode)]
    hash_cache: dict[Path, str] = {}
    route_names = set(source_to_route.values())
    for fold in range(5):
        routes = records[fold].get("routes", {})
        if set(routes) != route_names:
            raise ValueError(f"fold-{fold} composite route inventory mismatch")
        b32 = routes["b32"]
        if b32.get("family") != "candidate_refiner":
            raise ValueError("b32 route is not the candidate refiner")
        lines.append(("B32", str(fold), "b32", "-"))
        for name, route in sorted(routes.items()):
            if name == "b32":
                continue
            checkpoint = Path(str(route.get("checkpoint", "")))
            expected_hash = str(route.get("checkpoint_sha256", ""))
            if route.get("family") != "gf_focus" or not checkpoint.is_file():
                raise ValueError(f"invalid fold-{fold} Focus route {name}")
            observed_hash = hash_cache.setdefault(checkpoint, sha256(checkpoint))
            if observed_hash != expected_hash:
                raise ValueError(f"fold-{fold} Focus route checkpoint changed: {name}")
            use_tta = name.endswith("_tta_xy")
            if (
                str(route.get("tta_axes", "")) != ("xy" if use_tta else "")
                or int(route.get("tta_views", -1)) != (3 if use_tta else 1)
                or route.get("input_mode") != input_mode
            ):
                raise ValueError(f"fold-{fold} Focus route recipe mismatch: {name}")
            lines.append(("FOCUS", str(fold), name, str(checkpoint.resolve())))
    for name in sorted(route_names):
        if name.endswith("_tta_xy"):
            lines.append(("TTA", "-", name, "xy"))
    for source in sorted(expected_sources):
        lines.append(("SOURCE", "-", source, str(source_to_route[source])))
        lines.append(
            ("THRESHOLD", "-", source, f"{float(source_thresholds[source]):.8g}")
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-sources", required=True)
    args = parser.parse_args()
    expected_sources = {
        source.strip() for source in args.expected_sources.split(",") if source.strip()
    }
    if not expected_sources:
        parser.error("expected source inventory is empty")
    payload = json.loads(args.decision.read_text(encoding="utf-8"))
    for row in select(payload, expected_sources):
        print("\t".join(row))


if __name__ == "__main__":
    main()
