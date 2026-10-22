#!/usr/bin/env python3
"""Prepare label-free component confidence overrides for one inference cache."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from candidate_refiner_sampling import semantic_source_group
from component_meta_classifier import (
    component_features,
    load_qualified_component_classifier,
    sha256,
)


def group_name(source: str) -> str:
    group = semantic_source_group(source)
    if group in {"epoch105", "currentgf"}:
        return "mature"
    if group in {"autopet_weak", "lndb_weak"}:
        return "weak_sources"
    if group != "lung_cells":
        raise ValueError(f"unexpected proposal source: {source!r}")
    return group


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--component-decision", type=Path, required=True)
    parser.add_argument("--patch-fusion-decision", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.work / "manifest.json"
    inference_path = args.work / "INFERENCE_COMPLETE.json"
    for required in (manifest_path, inference_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("component_policy") != "source_separate":
        raise ValueError("component override requires source-separated proposals")
    records = list(manifest.get("records", []))
    if not records or len({str(record["patch_id"]) for record in records}) != len(records):
        raise ValueError("invalid or duplicate inference records")
    model, component_payload = load_qualified_component_classifier(
        args.component_decision, fold=args.fold
    )
    patch_payload = json.loads(args.patch_fusion_decision.read_text(encoding="utf-8"))
    if (
        patch_payload.get("decision") != "QUALIFY_FOCUS_CLASSIFIER_FUSION"
        or patch_payload.get("public_read") is not False
        or patch_payload.get("hidden_read") is not False
    ):
        raise ValueError("patch classifier fusion is not qualified local OOF evidence")
    reports = patch_payload.get("by_source", {})
    if set(reports) != {"mature", "weak_sources", "lung_cells"}:
        raise ValueError("patch classifier fusion source calibration mismatch")
    tile_thresholds = {
        name: float(report["candidate"]["threshold"]) for name, report in reports.items()
    }
    if any(not 0 < value < 1 for value in tile_thresholds.values()):
        raise ValueError("invalid source-specific tile threshold")

    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    patch_probability = {}
    for record in records:
        prediction_path = args.work / "predictions" / f"{record['patch_id']}.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        with np.load(prediction_path, allow_pickle=False) as prediction:
            value = float(prediction["classification_probability"])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid patch probability: {record['patch_id']}")
        patch_probability[str(record["patch_id"])] = value
        key = (
            str(record["case"]),
            str(record["source_name"]),
            int(record["source_component_id"]),
        )
        grouped[key].append(record)

    override = {}
    component_rows = []
    for (case, source, component_id), component_records in sorted(grouped.items()):
        ordered = sorted(component_records, key=lambda record: int(record["tile_index"]))
        values = [patch_probability[str(record["patch_id"])] for record in ordered]
        features = component_features(ordered, values)
        component_probability = float(model.predict_proba(features[None])[0, 1])
        if not np.isfinite(component_probability) or not 0 <= component_probability <= 1:
            raise ValueError("component classifier returned invalid probability")
        tile_threshold = tile_thresholds[group_name(source)]
        eligible_tiles = 0
        for record, value in zip(ordered, values):
            eligible = value >= tile_threshold
            eligible_tiles += int(eligible)
            override[str(record["patch_id"])] = component_probability if eligible else 0.0
        component_rows.append(
            (
                f"{case}|{source}|{component_id}",
                case,
                source,
                component_probability,
                len(ordered),
                eligible_tiles,
            )
        )
    if set(override) != {str(record["patch_id"]) for record in records}:
        raise RuntimeError("component override lacks exact patch coverage")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp.npz")
    ordered_ids = sorted(override)
    np.savez_compressed(
        temporary,
        patch_id=np.asarray(ordered_ids),
        classification_probability=np.asarray(
            [override[patch_id] for patch_id in ordered_ids], dtype=np.float32
        ),
        component=np.asarray([row[0] for row in component_rows]),
        component_case=np.asarray([row[1] for row in component_rows]),
        component_source=np.asarray([row[2] for row in component_rows]),
        component_probability=np.asarray([row[3] for row in component_rows], dtype=np.float32),
        component_tiles=np.asarray([row[4] for row in component_rows], dtype=np.int32),
        component_eligible_tiles=np.asarray([row[5] for row in component_rows], dtype=np.int32),
    )
    os.replace(temporary, args.output)
    terminal = args.output.with_suffix(".json")
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "selection_basis": "frozen qualified component model; label-free inference cache only",
        "work": str(args.work.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "inference_marker": str(inference_path.resolve()),
        "inference_marker_sha256": sha256(inference_path),
        "component_decision": str(args.component_decision.resolve()),
        "component_decision_sha256": sha256(args.component_decision),
        "component_model_sha256": component_payload["model_bundle_sha256"],
        "patch_fusion_decision": str(args.patch_fusion_decision.resolve()),
        "patch_fusion_decision_sha256": sha256(args.patch_fusion_decision),
        "fold": args.fold,
        "patches": len(records),
        "components": len(component_rows),
        "exact_patch_coverage": True,
        "tile_thresholds": tile_thresholds,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "labels_read": False,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    atomic_json(terminal, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
