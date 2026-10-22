#!/usr/bin/env python3
"""Write a terminal audit for the five-fold FLARE23 component refiner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--patch-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for fold in range(args.folds):
        directory = args.runs / f"fold{fold}"
        checkpoint = directory / "checkpoint_last.pt"
        history_path = directory / "history.json"
        if not checkpoint.is_file() or not history_path.is_file():
            raise FileNotFoundError(f"incomplete fold {fold}: {directory}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("fold", -1)) != fold or int(payload.get("epoch", -1)) != 30:
            raise RuntimeError(f"invalid terminal checkpoint for fold {fold}")
        rows.append(
            {
                "fold": fold,
                "epoch": int(payload["epoch"]),
                "best_segmentation": float(payload.get("best_segmentation", -1.0)),
                "best_cohort_segmentation": float(
                    payload.get("best_cohort_segmentation", -1.0)
                ),
                "best_classification": float(payload.get("best_classification", -1.0)),
                "checkpoint_last": {
                    "path": str(checkpoint.resolve()),
                    "sha256": sha256(checkpoint),
                },
                "history": {
                    "path": str(history_path.resolve()),
                    "sha256": sha256(history_path),
                },
            }
        )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "model": "base32 prompt-masked 3D component classifier/refiner",
        "folds": rows,
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest)},
        "patch_summary": {
            "path": str(args.patch_summary.resolve()),
            "sha256": sha256(args.patch_summary),
        },
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
