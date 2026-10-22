#!/usr/bin/env python3
"""Fail-closed provenance and voxel audit for a label-blind mask union."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all() or not np.isin(data, (0, 1)).all():
        raise ValueError(f"mask is not finite binary data: {path}")
    return image, data.astype(bool, copy=False)


def audit_case(task: tuple[str, str, str, str, str]) -> dict:
    name, image_text, source_a_text, source_b_text, union_text = task
    image_path = Path(image_text)
    paths = {
        "source_a": Path(source_a_text),
        "source_b": Path(source_b_text),
        "union": Path(union_text),
    }
    reference = nib.load(str(image_path))
    loaded = {key: _load_binary(path) for key, path in paths.items()}
    for key, (image, _) in loaded.items():
        if image.shape != reference.shape or not np.allclose(
            image.affine, reference.affine, rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"geometry mismatch for {name}: {key}")
    mask_a = loaded["source_a"][1]
    mask_b = loaded["source_b"][1]
    union = loaded["union"][1]
    expected = np.logical_or(mask_a, mask_b)
    if not np.array_equal(union, expected):
        differing = int(np.count_nonzero(union != expected))
        raise ValueError(f"union mismatch for {name}: {differing} voxels")
    return {
        "name": name,
        "voxels": {
            "source_a": int(mask_a.sum()),
            "source_b": int(mask_b.sum()),
            "union": int(union.sum()),
            "overlap": int(np.logical_and(mask_a, mask_b).sum()),
        },
        "files": {
            key: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
    }


def inventory_digest(results: list[dict], key: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    for result in sorted(results, key=lambda item: item["name"]):
        record = result["files"][key]
        total_bytes += int(record["bytes"])
        digest.update(result["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), total_bytes


def exact_names(directory: Path, expected: set[str], label: str) -> None:
    observed = {path.name for path in directory.glob("*.nii.gz")}
    if observed != expected:
        raise ValueError(
            f"{label} coverage mismatch: expected={len(expected)} "
            f"observed={len(observed)} missing={sorted(expected-observed)[:5]} "
            f"extra={sorted(observed-expected)[:5]}"
        )


def audit_union(
    manifest_path: Path,
    source_a: Path,
    source_b: Path,
    union: Path,
    workers: int,
    require_complete_labels: bool = True,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("manifest has no items")
    names = [str(item["prediction_name"]) for item in items]
    if len(names) != len(set(names)):
        raise ValueError("manifest prediction names are not unique")
    if require_complete_labels and any(
        item.get("complete_label") is not True for item in items
    ):
        raise ValueError("manifest contains a case without verified complete labels")
    expected = set(names)
    exact_names(source_a, expected, "source_a")
    exact_names(source_b, expected, "source_b")
    exact_names(union, expected, "union")
    tasks = [
        (
            str(item["prediction_name"]),
            str(Path(str(item["image"]))),
            str(source_a / str(item["prediction_name"])),
            str(source_b / str(item["prediction_name"])),
            str(union / str(item["prediction_name"])),
        )
        for item in sorted(items, key=lambda entry: str(entry["prediction_name"]))
    ]
    if workers == 1:
        results = [audit_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(audit_case, tasks))
    inventories = {}
    for key, directory in (
        ("source_a", source_a), ("source_b", source_b), ("union", union)
    ):
        digest, total_bytes = inventory_digest(results, key)
        inventories[key] = {
            "directory": str(directory.resolve()),
            "files": len(results),
            "bytes": total_bytes,
            "sha256_inventory": digest,
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": "binary_union",
        "minimum_component_mm3": 0.0,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "cases": len(results),
        "exact_file_coverage": True,
        "all_masks_finite_binary": True,
        "all_geometries_match_source_images": True,
        "all_unions_exact_voxelwise": True,
        "labels_read": False,
        "complete_labels_required": require_complete_labels,
        "inventories": inventories,
        "voxel_totals": {
            key: sum(result["voxels"][key] for result in results)
            for key in ("source_a", "source_b", "union", "overlap")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-a", type=Path, required=True)
    parser.add_argument("--source-b", type=Path, required=True)
    parser.add_argument("--union", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="audit inference cases without requiring complete-label metadata",
    )
    args = parser.parse_args()
    report = audit_union(
        args.manifest, args.source_a, args.source_b, args.union, args.workers,
        require_complete_labels=not args.allow_unlabeled,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
