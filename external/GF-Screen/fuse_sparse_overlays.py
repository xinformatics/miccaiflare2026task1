#!/usr/bin/env python3
"""Union sparse, label-blind proposal overlays into a complete mask stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all() or not np.isin(data, (0, 1)).all():
        raise ValueError(f"not finite binary: {path}")
    return image, data.astype(bool, copy=False)


def fuse_case(
    name: str,
    base: Path,
    overlays: tuple[Path, ...],
    output: Path,
    empty_base: bool = False,
) -> dict:
    present = [directory / name for directory in overlays if (directory / name).is_file()]
    source = base / name
    if not present and not empty_base:
        shutil.copyfile(source, output / name)
    else:
        if empty_base:
            reference = nib.load(str(source))
            fused = np.zeros(reference.shape, dtype=bool)
        else:
            reference, mask = load_binary(source)
            fused = mask.copy()
        for path in present:
            image, overlay = load_binary(path)
            if image.shape != reference.shape or not np.allclose(
                image.affine, reference.affine, rtol=0.0, atol=1e-5
            ):
                raise ValueError(f"geometry mismatch for {name}: {path}")
            fused |= overlay
        header = reference.header.copy(); header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(fused.astype(np.uint8), reference.affine, header), output / name)
    destination = output / name
    return {
        "name": name,
        "overlays": len(present),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def fuse_sparse(
    manifest: Path,
    base: Path,
    overlays: tuple[Path, ...],
    output: Path,
    workers: int,
    empty_base: bool = False,
) -> dict:
    if not overlays or workers < 1:
        raise ValueError("at least one overlay and a positive worker count are required")
    items = json.loads(manifest.read_text(encoding="utf-8")).get("items", [])
    names = [str(item["prediction_name"]) for item in items]
    if not names or len(names) != len(set(names)):
        raise ValueError("manifest prediction names are empty or duplicated")
    expected = set(names)
    observed_base = {path.name for path in base.glob("*.nii.gz")}
    if observed_base != expected:
        raise ValueError("base does not have exact manifest coverage")
    overlay_sets = []
    for directory in overlays:
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if not observed or not observed <= expected:
            raise ValueError(f"overlay is empty or contains unknown cases: {directory}")
        overlay_sets.append(observed)
    output.mkdir(parents=True, exist_ok=True)
    operation = partial(
        fuse_case,
        base=base,
        overlays=overlays,
        output=output,
        empty_base=empty_base,
    )
    if workers == 1:
        records = [operation(name) for name in sorted(names)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(operation, sorted(names)))
    observed_output = {path.name for path in output.glob("*.nii.gz")}
    if observed_output != expected:
        raise ValueError("output does not have exact manifest coverage")
    digest = hashlib.sha256()
    total_bytes = 0
    for record in records:
        total_bytes += int(record["bytes"])
        digest.update(record["name"].encode()); digest.update(b"\0")
        digest.update(str(record["bytes"]).encode()); digest.update(b"\0")
        digest.update(record["sha256"].encode()); digest.update(b"\n")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": (
            "empty_or_sparse_overlays" if empty_base else "base_or_sparse_overlays"
        ),
        "manifest": str(manifest.resolve()),
        "base": str(base.resolve()),
        "overlays": [
            {"directory": str(directory.resolve()), "files": len(observed)}
            for directory, observed in zip(overlays, overlay_sets)
        ],
        "cases": len(records),
        "cases_with_overlay": sum(record["overlays"] > 0 for record in records),
        "exact_file_coverage": True,
        "all_masks_finite_binary": True,
        "all_overlay_geometries_match": True,
        "empty_base": bool(empty_base),
        "labels_read": False,
        "bytes": total_bytes,
        "sha256_inventory": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--empty-base",
        action="store_true",
        help="use the base only as geometry and emit zeros where no overlay exists",
    )
    args = parser.parse_args()
    report = fuse_sparse(
        args.manifest,
        args.base,
        tuple(args.overlay),
        args.output,
        args.workers,
        args.empty_base,
    )
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.marker.with_name(f".{args.marker.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.marker)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
