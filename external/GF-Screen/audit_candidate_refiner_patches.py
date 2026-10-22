#!/usr/bin/env python3
"""Fail-closed integrity audit for candidate-refiner training patches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def audit_row(task: tuple[dict[str, str], tuple[int, int, int]]) -> dict[str, int]:
    row, shape = task
    path = Path(row["path"])
    with np.load(path, allow_pickle=False) as archive:
        required = {"image", "target", "label", "target_affine", "source_affine"}
        if set(archive.files) != required:
            raise ValueError(f"unexpected archive keys: {path} {archive.files}")
        image = archive["image"]
        target = archive["target"]
        label = int(archive["label"])
        if image.shape != (3, *shape) or target.shape != (1, *shape):
            raise ValueError(f"shape mismatch: {path} {image.shape} {target.shape}")
        if not np.isfinite(image).all() or float(image.min()) < 0 or float(image.max()) > 1:
            raise ValueError(f"invalid image values: {path}")
        if not np.isin(target, (0, 1)).all():
            raise ValueError(f"non-binary target: {path}")
        if label != int(row["label"]):
            raise ValueError(f"label mismatch: {path}")
        target_voxels = int(target.sum())
        seed_voxels = int(np.count_nonzero(image[2] > 1e-6))
        if seed_voxels == 0:
            raise ValueError(f"empty candidate prompt: {path}")
        if bool(target_voxels) != bool(label):
            raise ValueError(f"target/label inconsistency: {path}")
    return {
        "label": label,
        "target_voxels": target_voxels,
        "seed_voxels": seed_voxels,
        "fallback": int(row.get("seed_prompt_fallback", "0")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    allowed_versions = {
        "candidate-refiner-v4-iso15-preserved-seed-target",
        "candidate-refiner-v5-bbox24-tiled-all-volume",
    }
    if summary.get("version") not in allowed_versions:
        raise ValueError(f"unexpected patch version: {summary.get('version')}")
    if summary.get("version") == "candidate-refiner-v5-bbox24-tiled-all-volume":
        if summary.get("bbox_margin_mm") != 24.0 or summary.get("tile_stride_mm") != 72.0:
            raise ValueError("tiled training cache is not the required bbox24/stride72 geometry")
    if summary.get("shape") != [64, 64, 64] or summary.get("spacing") != [1.5, 1.5, 1.5]:
        raise ValueError("training cache is not the required isotropic geometry")
    digest = hashlib.sha256(args.index.read_bytes()).hexdigest()
    if summary.get("index", {}).get("sha256") != digest:
        raise ValueError("index digest does not match summary")
    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    patch_ids = [row["patch_id"] for row in rows]
    if not rows or len(patch_ids) != len(set(patch_ids)):
        raise ValueError("empty or duplicate patch index")
    indexed_paths = {Path(row["path"]).resolve() for row in rows}
    patch_directory = args.index.parent / "patches"
    disk_paths = {path.resolve() for path in patch_directory.glob("*.npz")}
    if indexed_paths != disk_paths:
        raise ValueError(
            f"patch coverage mismatch missing={len(indexed_paths-disk_paths)} "
            f"extra={len(disk_paths-indexed_paths)}"
        )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(audit_row, ((row, (64, 64, 64)) for row in rows)))
    report = {
        "version": summary["version"],
        "index": str(args.index.resolve()),
        "index_sha256": digest,
        "patches": len(rows),
        "positive": sum(result["label"] for result in results),
        "negative": sum(1 - result["label"] for result in results),
        "seed_prompt_fallbacks": sum(result["fallback"] for result in results),
        "target_resample_fallbacks": sum(
            int(row.get("target_resample_fallback", "0")) for row in rows
        ),
        "minimum_seed_voxels": min(result["seed_voxels"] for result in results),
        "minimum_positive_target_voxels": min(
            result["target_voxels"] for result in results if result["label"]
        ),
        "exact_file_coverage": True,
        "all_prompts_nonempty": True,
        "all_values_finite": True,
        "all_targets_binary_and_label_consistent": True,
        "shape": [3, 64, 64, 64],
        "spacing": [1.5, 1.5, 1.5],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
