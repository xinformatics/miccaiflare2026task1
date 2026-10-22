#!/usr/bin/env python3
"""Backfill hashes only when an independent prewarm audit proves identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "COMPLETE"
        or audit.get("public_read") is not False
        or audit.get("hidden_read") is not False
        or audit.get("submission_created") is not False
    ):
        raise ValueError("invalid inference prewarm audit")
    records = audit.get("folds", [])
    if [int(record["fold"]) for record in records] != list(range(len(records))):
        raise ValueError("non-contiguous prewarm fold coverage")
    for record in records:
        checkpoint = Path(record["checkpoint"])
        marker = Path(record["marker"])
        expected_digest = str(record["checkpoint_sha256"])
        if digest(checkpoint) != expected_digest:
            raise ValueError(f"checkpoint changed after prewarm: {checkpoint}")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("checkpoints") != [str(checkpoint.resolve())]:
            raise ValueError(f"marker checkpoint mismatch: {marker}")
        if payload.get("contour_checkpoints", []) != []:
            raise ValueError(f"unexpected contour checkpoint: {marker}")
        payload["checkpoint_sha256"] = [expected_digest]
        payload["contour_checkpoint_sha256"] = []
        payload["hash_identity_backfilled_from"] = str(args.audit.resolve())
        temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, marker)
    print(json.dumps({"updated_markers": len(records), "audit": str(args.audit.resolve())}))


if __name__ == "__main__":
    main()
