#!/usr/bin/env python3
"""Fail-closed audit for exact multi-source training proposal caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def parse_sources(values: list[str]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"source must use NAME=DIR: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in seen:
            raise ValueError(f"invalid or duplicate source name: {name!r}")
        path = Path(raw_path).resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"source is not a directory: {path}")
        sources.append((name, path))
        seen.add(name)
    if len(sources) < 2:
        raise ValueError("at least two named proposal sources are required")
    return sources


def _audit_case(task: tuple[str, str, tuple[tuple[str, str], ...]]) -> dict:
    name, image_text, source_values = task
    reference = nib.load(image_text)
    case_records: dict[str, dict[str, object]] = {}
    for source_name, directory_text in source_values:
        path = Path(directory_text) / name
        proposal = nib.load(str(path))
        if proposal.shape != reference.shape or not np.allclose(
            proposal.affine, reference.affine, rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"geometry mismatch: {source_name}/{name}")
        data = np.asanyarray(proposal.dataobj)
        if not np.isfinite(data).all() or not np.isin(data, (0, 1)).all():
            raise ValueError(f"non-finite or non-binary mask: {source_name}/{name}")
        case_records[source_name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "foreground_voxels": int(np.count_nonzero(data)),
        }
    return {"name": name, "sources": case_records}


def _inventory(results: list[dict], source_name: str) -> dict[str, object]:
    digest = hashlib.sha256()
    total_bytes = 0
    total_voxels = 0
    nonempty_cases = 0
    for result in sorted(results, key=lambda row: str(row["name"])):
        name = str(result["name"])
        record = result["sources"][source_name]
        size = int(record["bytes"])
        voxels = int(record["foreground_voxels"])
        total_bytes += size
        total_voxels += voxels
        nonempty_cases += int(voxels > 0)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return {
        "files": len(results),
        "bytes": total_bytes,
        "foreground_voxels": total_voxels,
        "nonempty_cases": nonempty_cases,
        "sha256_inventory": digest.hexdigest(),
    }


def audit_sources(
    manifest_path: Path,
    sources: list[tuple[str, Path]],
    workers: int,
    expected_count: int,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != expected_count:
        raise ValueError(
            f"manifest case count mismatch: {0 if not isinstance(items, list) else len(items)}/{expected_count}"
        )
    names = [str(item["prediction_name"]) for item in items]
    if len(names) != len(set(names)):
        raise ValueError("manifest prediction names are not unique")
    if any(item.get("complete_label") is not True for item in items):
        raise ValueError("manifest contains a case without a verified complete label")
    expected = set(names)
    for source_name, directory in sources:
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected:
            raise ValueError(
                f"coverage mismatch for {source_name}: observed={len(observed)} "
                f"missing={sorted(expected-observed)[:5]} extra={sorted(observed-expected)[:5]}"
            )
    source_values = tuple((name, str(path)) for name, path in sources)
    tasks = [
        (str(item["prediction_name"]), str(Path(str(item["image"])).resolve(strict=True)), source_values)
        for item in sorted(items, key=lambda row: str(row["prediction_name"]))
    ]
    if workers == 1:
        results = [_audit_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_audit_case, tasks))
    inventories = {
        name: {"directory": str(path), **_inventory(results, name)}
        for name, path in sources
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "cases": len(results),
        "source_names": [name for name, _ in sources],
        "exact_file_coverage": True,
        "complete_labels_verified_from_manifest": True,
        "all_masks_finite_binary": True,
        "all_geometries_match_source_images": True,
        "labels_read": False,
        "inventories": inventories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = audit_sources(
        args.manifest.resolve(strict=True),
        parse_sources(args.source),
        args.workers,
        args.expected_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
