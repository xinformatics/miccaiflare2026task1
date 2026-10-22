#!/usr/bin/env python3
"""Generate deterministic lung-coverage grid seeds without reading labels."""

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
from scipy import ndimage as ndi

from generate_lung_blob_proposals import approximate_lung_region, paint_sphere


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def lung_grid_centers(
    ct: np.ndarray,
    spacing: np.ndarray,
    target_spacing_mm: float,
    grid_spacing_mm: float,
) -> list[np.ndarray]:
    zoom = spacing / target_spacing_mm
    sampled = ndi.zoom(
        ct.astype(np.float32, copy=False), zoom, order=1, mode="nearest", prefilter=False
    )
    lung = approximate_lung_region(
        sampled, max(1, round(10.0 / target_spacing_mm))
    )
    coordinates = np.argwhere(lung)
    if coordinates.size == 0:
        return []
    physical = coordinates * target_spacing_mm
    cells = np.unique(np.floor(physical / grid_spacing_mm).astype(np.int32), axis=0)
    centers_sampled = (cells.astype(np.float64) + 0.5) * (
        grid_spacing_mm / target_spacing_mm
    )
    maximum = np.asarray(sampled.shape, dtype=np.float64) - 1.0
    centers_sampled = np.clip(centers_sampled, 0.0, maximum)
    return [center / zoom for center in centers_sampled]


def _run_case(task: dict[str, object]) -> dict[str, object]:
    image_path = Path(str(task["image"]))
    destination = Path(str(task["destination"]))
    image = nib.load(str(image_path))
    ct = np.asanyarray(image.dataobj, dtype=np.float32)
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    centers = lung_grid_centers(
        ct,
        spacing,
        float(task["target_spacing_mm"]),
        float(task["grid_spacing_mm"]),
    )
    mask = np.zeros(image.shape, dtype=bool)
    for center in centers:
        paint_sphere(mask, center, spacing, float(task["sphere_radius_mm"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), image.affine, header), str(temporary))
    os.replace(temporary, destination)
    return {
        "case": destination.name,
        "grid_seeds": len(centers),
        "foreground_voxels": int(mask.sum()),
        "image_sha256": sha256_file(image_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-spacing-mm", type=float, default=2.5)
    parser.add_argument("--grid-spacing-mm", type=float, default=48.0)
    parser.add_argument("--sphere-radius-mm", type=float, default=3.0)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if min(args.target_spacing_mm, args.grid_spacing_mm, args.sphere_radius_mm) <= 0:
        raise ValueError("spacings and radius must be positive")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = [item for item in payload["items"] if str(item.get("cohort")) == args.cohort]
    if len(items) != args.expected_count or any(item.get("complete_label") is not True for item in items):
        raise ValueError(f"cohort coverage/completeness mismatch: {len(items)}/{args.expected_count}")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "image": item["image"],
            "destination": str(args.output / str(item["prediction_name"])),
            "target_spacing_mm": args.target_spacing_mm,
            "grid_spacing_mm": args.grid_spacing_mm,
            "sphere_radius_mm": args.sphere_radius_mm,
        }
        for item in items
    ]
    if args.workers == 1:
        rows = [_run_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_run_case, tasks))
    expected = {str(item["prediction_name"]) for item in items}
    observed = {path.name for path in args.output.glob("*.nii.gz")}
    if observed != expected:
        raise ValueError(f"output coverage mismatch: {len(observed)}/{len(expected)}")
    counts = [int(row["grid_seeds"]) for row in rows]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "occupied_48mm_lung_cell_centers",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "cohort": args.cohort,
        "cases": len(rows),
        "parameters": {
            "target_spacing_mm": args.target_spacing_mm,
            "grid_spacing_mm": args.grid_spacing_mm,
            "sphere_radius_mm": args.sphere_radius_mm,
        },
        "grid_seed_count": {
            "minimum": min(counts), "mean": float(np.mean(counts)), "maximum": max(counts)
        },
        "labels_read": False,
        "public_or_hidden_data_read": False,
        "rows": rows,
    }
    marker = args.output / "PROPOSALS_COMPLETE.json"
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
