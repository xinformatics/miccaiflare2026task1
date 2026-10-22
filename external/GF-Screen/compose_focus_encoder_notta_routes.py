#!/usr/bin/env python3
"""Compose the pre-public B32/frozen-Focus/encoder-Focus no-TTA route."""

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
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(record: dict, label: str) -> dict:
    path = Path(str(record["checkpoint"]))
    expected = str(record["checkpoint_sha256"])
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"{label} checkpoint identity drift")
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": expected,
        "epoch": int(record["epoch"]),
    }


def compose(frozen: dict, encoder: dict, upgrade: dict, mixed: dict) -> dict:
    for name, payload in (
        ("frozen routes", frozen),
        ("encoder routes", encoder),
        ("encoder decision", upgrade),
        ("mixed decision", mixed),
    ):
        if any(payload.get(key) is not False for key in ("public_read", "hidden_read", "submission_created")):
            raise ValueError(f"{name} is not local-only")
    index_hash = str(frozen.get("index_sha256", ""))
    if not index_hash or encoder.get("index_sha256") != index_hash or mixed.get("index_sha256") != index_hash:
        raise ValueError("route index identities do not match")
    if upgrade.get("decision") != "PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF" or upgrade.get("qualified_routes") != ["lung_cells"]:
        raise ValueError("encoder upgrade is not restricted to lung cells")
    if mixed.get("decision") != "QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME":
        raise ValueError("mixed architecture route did not qualify")
    if set(frozen.get("source_segmentation_groups", {})) != GROUPS or set(encoder.get("source_segmentation_groups", {})) != GROUPS:
        raise ValueError("source threshold coverage mismatch")
    if encoder.get("training_recipe", {}).get("freeze_encoder") is not False or float(encoder.get("training_recipe", {}).get("encoder_lr_scale", -1)) != 0.1:
        raise ValueError("encoder training recipe drift")
    if frozen.get("training_recipe", {}).get("input_mode") != "soft_lung_seed" or encoder.get("training_recipe", {}).get("input_mode") != "soft_lung_seed":
        raise ValueError("Focus input mode drift")

    sources = mixed.get("source_to_route", {})
    if set(sources.values()) != {"b32", "focus"}:
        raise ValueError("mixed source route must contain B32 and Focus")
    encoder_groups = encoder.get("source_to_route", {})
    if set(encoder_groups) != set(sources) or set(encoder_groups.values()) != GROUPS:
        raise ValueError("encoder source inventory mismatch")

    source_to_route = {}
    thresholds = {}
    mixed_thresholds = mixed.get("source_segmentation_thresholds", {})
    for source, family in sorted(sources.items()):
        group = encoder_groups[source]
        if family == "b32":
            if group != "mature":
                raise ValueError("only mature sources may retain B32")
            route = "b32"
            threshold = float(mixed_thresholds[source])
        elif group == "weak_sources":
            route = "focus_weak_sources_notta"
            threshold = float(frozen["source_segmentation_groups"][group]["threshold"])
        elif group == "lung_cells":
            route = "focus_encoder01_lung_cells_notta"
            threshold = float(encoder["source_segmentation_groups"][group]["threshold"])
        else:
            raise ValueError(f"unsupported Focus group for {source}: {group}")
        if not 0 < threshold < 1:
            raise ValueError(f"invalid source threshold for {source}")
        source_to_route[source] = route
        thresholds[source] = threshold

    frozen_folds = {int(row["fold"]): row for row in frozen.get("fold_records", [])}
    encoder_folds = {int(row["fold"]): row for row in encoder.get("fold_records", [])}
    if set(frozen_folds) != set(range(5)) or set(encoder_folds) != set(range(5)):
        raise ValueError("five-fold route coverage is required")
    fold_records = []
    for fold in range(5):
        frozen_selected = frozen_folds[fold].get("selected", {})
        encoder_selected = encoder_folds[fold].get("selected", {})
        if set(frozen_selected) != GROUPS or set(encoder_selected) != GROUPS:
            raise ValueError(f"fold {fold} selected-route coverage mismatch")
        weak = checkpoint(frozen_selected["weak_sources"], f"fold {fold} weak")
        lung = checkpoint(encoder_selected["lung_cells"], f"fold {fold} lung")
        fold_records.append(
            {
                "fold": fold,
                "routes": {
                    "focus_weak_sources_notta": {
                        "family": "gf_focus",
                        "variant": "frozen_encoder",
                        "group": "weak_sources",
                        "tta_views": 1,
                        **weak,
                    },
                    "focus_encoder01_lung_cells_notta": {
                        "family": "gf_focus",
                        "variant": "encoder_lr_scale_0.1",
                        "group": "lung_cells",
                        "tta_views": 1,
                        **lung,
                    },
                },
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "pre-public trained-model route: B32 mature, frozen dual-window Focus weak, encoder-adapted Focus lung; no TTA",
        "decision": "QUALIFY_ENCODER_FOCUS_B32_NOTTA_FOR_FIXED_PUBLIC",
        "index_sha256": index_hash,
        "folds": 5,
        "focus_input_mode": "soft_lung_seed",
        "source_to_route": source_to_route,
        "source_segmentation_thresholds": thresholds,
        "fold_records": fold_records,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-routes", type=Path, required=True)
    parser.add_argument("--encoder-routes", type=Path, required=True)
    parser.add_argument("--encoder-decision", type=Path, required=True)
    parser.add_argument("--mixed-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.frozen_routes, args.encoder_routes, args.encoder_decision, args.mixed_decision]
    payload = compose(*(json.loads(path.read_text(encoding="utf-8")) for path in paths))
    payload["provenance"] = [
        {"path": str(path.resolve()), "sha256": sha256(path)} for path in paths
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
