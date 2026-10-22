#!/usr/bin/env python3
"""Resolve and verify the trained base parent for the partial-band branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


ALLOWED_PARENT_EPOCHS = (100, 200, 300, 400, 500)
BASE_TRAINER = "nnUNetTrainerResEncS6PanCancerPartialBox"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_parent(selection_path: Path, expected_fold: Path) -> dict[str, object]:
    selection_path = selection_path.resolve()
    expected_fold = expected_fold.resolve()
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "COMPLETE"
        or payload.get("public_read") is not True
        or payload.get("hidden_read")
        or payload.get("submission_created")
        or payload.get("uploaded")
    ):
        raise ValueError("base fixed-public selection is not safely complete")
    selected = payload.get("selected")
    diagnostic = payload.get("diagnostic_best")
    row = selected or diagnostic
    source = "qualified_selected" if selected else "diagnostic_best"
    if not isinstance(row, dict):
        raise ValueError("base selection has no usable parent row")
    epoch = int(row.get("epoch", -1))
    if epoch not in ALLOWED_PARENT_EPOCHS:
        raise ValueError(f"base parent epoch is not allowed: {epoch}")
    checkpoint_record = row.get("checkpoint") or {}
    checkpoint = Path(checkpoint_record.get("path", "")).resolve()
    if checkpoint.parent != expected_fold or not checkpoint.is_file():
        raise ValueError("base checkpoint is outside the expected fold or absent")
    observed_sha = sha256(checkpoint)
    if checkpoint_record.get("sha256") != observed_sha:
        raise ValueError("base checkpoint SHA256 differs from public selection")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        state.get("trainer_name") != BASE_TRAINER
        or int(state.get("current_epoch", -1)) != epoch
        or not isinstance(state.get("network_weights"), dict)
        or not state["network_weights"]
    ):
        raise ValueError("base checkpoint trainer, epoch, or weights are invalid")
    metrics = {
        "Lesion_DSC": float(row["Lesion_DSC"]),
        "Lesion_NSD": float(row["Lesion_NSD"]),
        "deployment": str(row["deployment"]),
        "qualified": bool(row.get("qualified")),
    }
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "source": source,
        "epoch": epoch,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": observed_sha,
        },
        "fixed_public_metrics": metrics,
        "selection": {
            "path": str(selection_path),
            "sha256": sha256(selection_path),
        },
        "public_selection_read": True,
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--expected-fold", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = select_parent(args.selection, args.expected_fold)
    if args.output:
        atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
