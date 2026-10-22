#!/usr/bin/env python3
"""Compose source-routed contour OOF predictions with strict provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def parse_named_path(values: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must use NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"invalid or duplicate {option} name: {name!r}")
        result[name] = Path(raw_path)
    return result


def parse_source_routes(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"source-route must use SOURCE=ROUTE: {value}")
        source, route = (part.strip() for part in value.split("=", 1))
        if not source or not route or source in result:
            raise ValueError(f"invalid or duplicate source route: {value}")
        result[source] = route
    return result


def require_equal(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], key: str
) -> None:
    if key not in reference or key not in candidate:
        raise ValueError(f"validation archive lacks identity field: {key}")
    if not np.array_equal(reference[key], candidate[key]):
        raise ValueError(f"classification/contour validation mismatch: {key}")


def compose_source_routed(
    classification: dict[str, np.ndarray],
    contours: dict[str, dict[str, np.ndarray]],
    source_by_patch: dict[str, str],
    source_to_route: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if not contours:
        raise ValueError("at least one contour route is required")
    for required in ("patch_id", "label", "classification_probability"):
        if required not in classification:
            raise ValueError(f"classification archive lacks {required}")
    if len(classification["classification_probability"]) != len(
        classification["patch_id"]
    ):
        raise ValueError("classification probability coverage mismatch")

    reference = next(iter(contours.values()))
    for route, contour in contours.items():
        for key in (
            "patch_id",
            "label",
            "positive_patch_id",
            "segmentation_thresholds",
        ):
            require_equal(classification, contour, key)
            require_equal(reference, contour, key)
        grid = contour.get("segmentation_dice_grid")
        expected_shape = (
            len(contour["positive_patch_id"]),
            len(contour["segmentation_thresholds"]),
        )
        if grid is None or grid.shape != expected_shape:
            raise ValueError(f"contour Dice-grid coverage mismatch: {route}")

    patch_ids = classification["patch_id"].astype(str)
    if len(patch_ids) != len(set(patch_ids)):
        raise ValueError("classification patch IDs are not unique")
    missing = sorted(set(patch_ids) - set(source_by_patch))[:5]
    if missing:
        raise ValueError(f"held-out patch absent from full index: {missing}")
    index_sources = set(source_by_patch.values())
    if set(source_to_route) != index_sources:
        missing = sorted(index_sources - set(source_to_route))
        extra = sorted(set(source_to_route) - index_sources)
        raise ValueError(f"source-route coverage mismatch missing={missing} extra={extra}")
    unknown_routes = sorted(set(source_to_route.values()) - set(contours))
    if unknown_routes:
        raise ValueError(f"source routes reference unknown contours: {unknown_routes}")
    unused_routes = sorted(set(contours) - set(source_to_route.values()))
    if unused_routes:
        raise ValueError(f"unused contour routes: {unused_routes}")

    positive_ids = reference["positive_patch_id"].astype(str)
    output_grid = np.empty_like(reference["segmentation_dice_grid"])
    route_counts = {route: 0 for route in contours}
    for index, patch_id in enumerate(positive_ids):
        source = source_by_patch.get(patch_id)
        if source is None:
            raise ValueError(f"positive patch absent from index: {patch_id}")
        route = source_to_route[source]
        output_grid[index] = contours[route]["segmentation_dice_grid"][index]
        route_counts[route] += 1

    return (
        {
            "patch_id": classification["patch_id"],
            "label": classification["label"],
            "classification_probability": classification[
                "classification_probability"
            ],
            "positive_patch_id": reference["positive_patch_id"],
            "segmentation_thresholds": reference["segmentation_thresholds"],
            "segmentation_dice_grid": output_grid,
        },
        route_counts,
    )


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checkpoint_identity(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    identity = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "fold": int(payload.get("fold", config.get("fold", -1))),
        "folds": int(payload.get("folds", config.get("folds", -1))),
        "seed": int(config.get("seed", payload.get("seed", -1))),
        "epoch": int(payload.get("epoch", -1)),
        "index_sha256": str(payload.get("index_sha256", "")),
    }
    if (
        identity["folds"] < 2
        or not 0 <= identity["fold"] < identity["folds"]
        or identity["seed"] < 0
        or identity["epoch"] < 1
        or len(identity["index_sha256"]) != 64
    ):
        raise ValueError(f"invalid checkpoint identity: {path}")
    return identity


def read_sources(index_path: Path) -> tuple[dict[str, str], str]:
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "patch_id" not in rows[0] or "source_name" not in rows[0]:
        raise ValueError("index must contain patch_id and source_name")
    source_by_patch: dict[str, str] = {}
    for row in rows:
        patch_id, source = row["patch_id"].strip(), row["source_name"].strip()
        if not patch_id or not source or patch_id in source_by_patch:
            raise ValueError(f"invalid or duplicate index patch: {patch_id!r}")
        source_by_patch[patch_id] = source
    return source_by_patch, sha256(index_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--classification-checkpoint", type=Path, required=True)
    parser.add_argument("--contour", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--contour-checkpoint", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument(
        "--source-route", action="append", required=True, metavar="SOURCE=NAME"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    contour_paths = parse_named_path(args.contour, "contour")
    checkpoint_paths = parse_named_path(args.contour_checkpoint, "contour-checkpoint")
    if set(contour_paths) != set(checkpoint_paths):
        raise ValueError("contour and contour-checkpoint route names differ")
    source_to_route = parse_source_routes(args.source_route)
    source_by_patch, index_digest = read_sources(args.index)
    classification = load(args.classification)
    contours = {name: load(path) for name, path in contour_paths.items()}
    classification_checkpoint = checkpoint_identity(args.classification_checkpoint)
    route_checkpoints = {
        name: checkpoint_identity(path) for name, path in checkpoint_paths.items()
    }
    for name, identity in route_checkpoints.items():
        for key in ("fold", "folds", "seed", "index_sha256"):
            if identity[key] != classification_checkpoint[key]:
                raise ValueError(f"classification/{name} checkpoint mismatch: {key}")
    if classification_checkpoint["index_sha256"] != index_digest:
        raise ValueError("checkpoint/index digest mismatch")

    arrays, route_counts = compose_source_routed(
        classification, contours, source_by_patch, source_to_route
    )
    atomic_npz(args.output, arrays)
    routes = {
        name: {
            "sources": sorted(
                source for source, route in source_to_route.items() if route == name
            ),
            "contour": str(contour_paths[name].resolve()),
            "contour_sha256": sha256(contour_paths[name]),
            "contour_checkpoint": route_checkpoints[name],
            "positive_patches": route_counts[name],
        }
        for name in sorted(contour_paths)
    }
    payload = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "index": str(args.index.resolve()),
        "index_sha256": index_digest,
        "classification": str(args.classification.resolve()),
        "classification_sha256": sha256(args.classification),
        "classification_checkpoint": classification_checkpoint,
        "contour_routes": routes,
        "source_to_route": dict(sorted(source_to_route.items())),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "patches": len(arrays["patch_id"]),
        "positive_patches": len(arrays["positive_patch_id"]),
        "labels_read_for_identity_only": True,
        "labels_used_for_routing": False,
        "public_read": False,
        "hidden_read": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.audit.with_name(f".{args.audit.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.audit)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
