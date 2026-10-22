#!/usr/bin/env python3
"""Create a validated flat symlink view of the nested Train-Unlabel images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    sources = sorted(args.source.rglob("*.nii.gz"), key=lambda path: path.name)
    if len(sources) != args.expected:
        raise RuntimeError(f"expected {args.expected} images, found {len(sources)}")
    source_names = [path.name for path in sources]
    if len(source_names) != len(set(source_names)):
        duplicates = sorted(
            {name for name in source_names if source_names.count(name) > 1}
        )
        raise RuntimeError(f"duplicate source basenames: {duplicates[:20]}")
    names = [
        path.name
        if path.name.endswith("_0000.nii.gz")
        else path.name.removesuffix(".nii.gz") + "_0000.nii.gz"
        for path in sources
    ]
    if len(names) != len(set(names)):
        raise RuntimeError("canonical modality-suffixed basenames are not unique")

    args.output.mkdir(parents=True, exist_ok=True)
    expected_names = set(names)
    unexpected = {path.name for path in args.output.iterdir()} - expected_names
    if unexpected:
        raise RuntimeError(f"unexpected staged files: {sorted(unexpected)[:20]}")
    rows = []
    for source, staged_name in zip(sources, names):
        destination = args.output / staged_name
        resolved = source.resolve()
        if destination.is_symlink():
            if destination.resolve() != resolved:
                raise RuntimeError(f"stale symlink: {destination}")
        elif destination.exists():
            raise RuntimeError(f"non-symlink staged member: {destination}")
        else:
            destination.symlink_to(resolved)
        stat = resolved.stat()
        rows.append(f"{staged_name}\t{source.name}\t{stat.st_size}\t{resolved}")
    digest = hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "cases": len(rows),
        "unique_basenames": len(set(names)),
        "source_name_size_path_sha256": digest,
        "public_read": False,
        "hidden_read": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.manifest, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
