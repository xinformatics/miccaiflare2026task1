#!/usr/bin/env python3
"""Select and materialize one source-group contour checkpoint reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_report(record: dict, source_group: str) -> dict:
    groups = record.get("source_segmentation_groups", {})
    sources = record.get("source_segmentation", {})
    if source_group in groups:
        return groups[source_group]
    if source_group in sources:
        return sources[source_group]
    raise ValueError(f"source/group absent from audit: {source_group}")


def choose(records: list[dict], source_group: str) -> dict:
    if not records:
        raise ValueError("no scored-epoch source audits supplied")
    identities = {
        (int(record["fold"]), int(record["folds"]), int(record["seed"]), record["index_sha256"])
        for record in records
    }
    if len(identities) != 1:
        raise ValueError("source audits mix fold/index identities")
    candidates = []
    for record in records:
        report = source_report(record, source_group)
        candidates.append(
            (
                float(report["dice"]),
                -abs(float(report["threshold"]) - 0.5),
                -int(record["epoch"]),
                record,
            )
        )
    return max(candidates, key=lambda item: item[:3])[3]


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--source-group", required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    audit_inventory = []
    for path in args.audit:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = Path(payload["checkpoint"])
        prediction = Path(payload["prediction"])
        if sha256(checkpoint) != payload["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        if sha256(prediction) != payload["prediction_sha256"]:
            raise ValueError(f"prediction hash mismatch: {prediction}")
        records.append(payload)
        audit_inventory.append(
            {"path": str(path.resolve()), "sha256": sha256(path), "epoch": payload["epoch"]}
        )
    selected = choose(records, args.source_group)
    checkpoint = Path(selected["checkpoint"])
    prediction = Path(selected["prediction"])
    atomic_copy(checkpoint, args.checkpoint_output)
    atomic_copy(prediction, args.prediction_output)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "internal case-grouped held-out source-group contour Dice",
        "source_group": args.source_group,
        "selected_epoch": int(selected["epoch"]),
        "selected_dice": float(
            source_report(selected, args.source_group)["dice"]
        ),
        "selected_threshold": float(
            source_report(selected, args.source_group)["threshold"]
        ),
        "checkpoint_source": str(checkpoint.resolve()),
        "checkpoint_output": str(args.checkpoint_output.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint_output),
        "prediction_source": str(prediction.resolve()),
        "prediction_output": str(args.prediction_output.resolve()),
        "prediction_sha256": sha256(args.prediction_output),
        "audits": sorted(audit_inventory, key=lambda value: int(value["epoch"])),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
