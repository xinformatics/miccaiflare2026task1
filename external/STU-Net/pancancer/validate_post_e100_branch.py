#!/usr/bin/env python3
"""Validate or initialize a hash-bound post-epoch-100 STU-Net branch."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


COMPONENT_DIFFERENCE = "union of expanded 26-connected component boxes"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path) -> dict:
    allowed = [
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.float32)),
        type(np.dtype(np.int64)),
        type(np.dtype(np.bool_)),
    ]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    return state


def validate_state(path: Path, *, exact_epoch: int | None = None) -> int:
    state = load_state(path)
    epoch = int(state.get("current_epoch", -1))
    if exact_epoch is not None:
        valid_epoch = epoch == exact_epoch
    else:
        valid_epoch = 100 <= epoch < 300
    if (
        not valid_epoch
        or not state.get("network_weights")
        or not state.get("optimizer_state")
    ):
        raise ValueError(f"invalid optimizer checkpoint epoch {epoch}: {path}")
    return epoch


def validate_branch(
    mode: str,
    source: Path,
    latest: Path,
    trainer_source: Path | None = None,
    manifest: Path | None = None,
) -> dict:
    source_epoch = validate_state(source, exact_epoch=100)
    latest_epoch = validate_state(latest)
    source_sha = sha256_file(source)
    latest_sha = sha256_file(latest)
    result = {
        "status": "PASS",
        "mode": mode,
        "source_sha256": source_sha,
        "source_epoch": source_epoch,
        "latest_sha256": latest_sha,
        "latest_epoch": latest_epoch,
        "public_read": False,
        "hidden_read": False,
    }
    if mode == "base":
        return result
    if mode != "component" or trainer_source is None or manifest is None:
        raise ValueError("component validation requires trainer source and manifest")

    trainer_sha = sha256_file(trainer_source)
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("source_checkpoint", {}).get("sha256") != source_sha
            or payload.get("branch_checkpoint", {}).get("sha256") != source_sha
            or payload.get("trainer", {}).get("sha256") != trainer_sha
            or payload.get("only_intended_training_difference")
            != COMPONENT_DIFFERENCE
        ):
            raise ValueError("component source manifest mismatch")
        result["manifest_created"] = False
        result["manifest_sha256"] = sha256_file(manifest)
        return result

    if latest_epoch != 100 or latest_sha != source_sha:
        raise ValueError("component branch was not initialized from exact epoch-100 bytes")
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "source_checkpoint": {
            "path": str(source.resolve()),
            "sha256": source_sha,
            "embedded_epoch": 100,
        },
        "branch_checkpoint": {
            "path": str(latest.resolve()),
            "sha256": latest_sha,
        },
        "trainer": {
            "path": str(trainer_source.resolve()),
            "sha256": trainer_sha,
        },
        "only_intended_training_difference": COMPONENT_DIFFERENCE,
        "public_read": False,
        "hidden_read": False,
    }
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    result["manifest_created"] = True
    result["manifest_sha256"] = sha256_file(manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("base", "component"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--latest", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = validate_branch(
        args.mode, args.source, args.latest, args.trainer_source, args.manifest
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
