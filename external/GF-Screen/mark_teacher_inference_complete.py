#!/usr/bin/env python3
"""Write a hash-bound scheduling terminal after nested teacher inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from audit_nested_teacher_masks import discover_images


THRESHOLDS = ("0.85", "0.90", "0.95", "0.97")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def inventory(directory: Path) -> tuple[set[str], str]:
    members = {
        path.name: path.stat().st_size
        for path in directory.glob("*.nii.gz")
        if path.is_file() and not path.name.endswith("_trans.nii.gz")
    }
    if any(size < 1 for size in members.values()):
        raise ValueError(f"empty teacher prediction in {directory}")
    digest = hashlib.sha256(
        "".join(f"{name}\t{members[name]}\n" for name in sorted(members)).encode()
    ).hexdigest()
    return set(members), digest


def mark_complete(
    images: Path,
    teacher_root: Path,
    checkpoint: Path,
    source_audit: Path,
    output: Path,
    expected_count: int = 2856,
) -> dict[str, object]:
    expected = set(discover_images(images))
    if len(expected) != expected_count:
        raise ValueError(
            f"source inventory mismatch: {len(expected)} != {expected_count}"
        )
    audit = json.loads(source_audit.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "PASS_WITH_EXCLUSIONS"
        or int(audit.get("unlabeled_files", -1)) != expected_count
        or int(audit.get("usable_unique_files", -1)) != expected_count - 6
    ):
        raise ValueError("Train-Unlabel source audit mismatch")
    inventories = {}
    for threshold in THRESHOLDS:
        directory = teacher_root / f"lesion_{threshold}"
        names, digest = inventory(directory)
        if names != expected:
            raise ValueError(
                f"threshold {threshold} set mismatch: "
                f"missing={len(expected - names)} extra={len(names - expected)}"
            )
        inventories[threshold] = {
            "directory": str(directory.resolve()),
            "members": len(names),
            "member_name_size_sha256": digest,
        }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": "TEACHER_INFERENCE_COMPLETE_GPU_RELEASE_PENDING",
        "source_images": str(images.resolve()),
        "source_inventory": expected_count,
        "usable_after_exact_overlap_exclusion": expected_count - 6,
        "teacher_checkpoint": asset(checkpoint),
        "source_audit": asset(source_audit),
        "threshold_inventories": inventories,
        "meaning": (
            "all teacher files are complete; GPU consumers must still apply "
            "their own memory-admission check"
        ),
        "pseudo_materialization_complete": False,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2856)
    args = parser.parse_args()
    payload = mark_complete(
        args.images, args.teacher_root, args.checkpoint, args.source_audit,
        args.output, args.expected_count,
    )
    print(json.dumps({
        "status": payload["status"],
        "decision": payload["decision"],
        "source_inventory": payload["source_inventory"],
        "threshold_inventories": payload["threshold_inventories"],
    }, indent=2))


if __name__ == "__main__":
    main()
