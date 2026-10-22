#!/usr/bin/env python3
"""Verify inference checkpoint provenance and publish a GPU-release marker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.work) != len(args.checkpoint) or not args.work:
        raise ValueError("work/checkpoint counts must match and be nonzero")
    records = []
    for work, checkpoint in zip(args.work, args.checkpoint):
        marker = work / "INFERENCE_COMPLETE.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        checkpoint = checkpoint.resolve(strict=True)
        digest = sha256_file(checkpoint)
        valid = (
            payload.get("checkpoints") == [str(checkpoint)]
            and payload.get("checkpoint_sha256") == [digest]
            and payload.get("contour_checkpoints", []) == []
            and payload.get("contour_checkpoint_sha256", []) == []
        )
        if not valid:
            raise ValueError(f"inference provenance mismatch: {marker}")
        records.append(
            {
                "work": str(work.resolve(strict=True)),
                "inference_marker": str(marker.resolve(strict=True)),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
            }
        )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpu_inference_complete": True,
        "folds": len(records),
        "records": records,
        "cache_payload_audits_pending_or_independent": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
