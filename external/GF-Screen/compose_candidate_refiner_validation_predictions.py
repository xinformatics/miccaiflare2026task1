#!/usr/bin/env python3
"""Compose classification and contour validation predictions without leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def require_equal(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    key: str,
) -> None:
    if key not in left or key not in right or not np.array_equal(left[key], right[key]):
        raise ValueError(f"classification/contour validation mismatch: {key}")


def compose(
    classification: dict[str, np.ndarray], contour: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    for key in ("patch_id", "label", "positive_patch_id", "segmentation_thresholds"):
        require_equal(classification, contour, key)
    required_classification = {"classification_probability"}
    required_contour = {"segmentation_dice_grid"}
    if not required_classification.issubset(classification):
        raise ValueError("classification archive lacks probabilities")
    if not required_contour.issubset(contour):
        raise ValueError("contour archive lacks segmentation grid")
    if len(classification["classification_probability"]) != len(classification["patch_id"]):
        raise ValueError("classification probability coverage mismatch")
    if contour["segmentation_dice_grid"].shape != (
        len(contour["positive_patch_id"]),
        len(contour["segmentation_thresholds"]),
    ):
        raise ValueError("contour Dice-grid coverage mismatch")
    return {
        "patch_id": classification["patch_id"],
        "label": classification["label"],
        "classification_probability": classification["classification_probability"],
        "positive_patch_id": contour["positive_patch_id"],
        "segmentation_thresholds": contour["segmentation_thresholds"],
        "segmentation_dice_grid": contour["segmentation_dice_grid"],
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--contour", type=Path, required=True)
    parser.add_argument("--classification-checkpoint", type=Path, required=True)
    parser.add_argument("--contour-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    classification = load(args.classification)
    contour = load(args.contour)
    classification_checkpoint = checkpoint_identity(args.classification_checkpoint)
    contour_checkpoint = checkpoint_identity(args.contour_checkpoint)
    for key in ("fold", "folds", "seed", "index_sha256"):
        if classification_checkpoint[key] != contour_checkpoint[key]:
            raise ValueError(f"classification/contour checkpoint mismatch: {key}")
    arrays = compose(classification, contour)
    atomic_npz(args.output, arrays)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "classification": str(args.classification.resolve()),
        "classification_sha256": sha256(args.classification),
        "classification_checkpoint": classification_checkpoint,
        "contour": str(args.contour.resolve()),
        "contour_sha256": sha256(args.contour),
        "contour_checkpoint": contour_checkpoint,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "patches": len(arrays["patch_id"]),
        "positive_patches": len(arrays["positive_patch_id"]),
        "labels_read": False,
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
