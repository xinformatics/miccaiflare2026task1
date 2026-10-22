#!/usr/bin/env python3
"""Validate geometry and monotonicity of nested teacher-mask exports.

This is safe to run while sharded inference is active: it snapshots only names
that are already present in every requested threshold directory, then audits
that immutable set. Temporary writer files are ignored because their names do
not match the corresponding source prediction name.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from inference_naming import prediction_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--threshold-dir",
        action="append",
        required=True,
        metavar="THRESHOLD=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=64)
    return parser.parse_args()


def parse_threshold_dirs(values: list[str]) -> list[tuple[float, Path]]:
    parsed: list[tuple[float, Path]] = []
    for value in values:
        threshold_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"invalid threshold directory: {value}")
        threshold = float(threshold_text)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold outside [0,1]: {threshold}")
        parsed.append((threshold, Path(path_text)))
    parsed.sort()
    if len({threshold for threshold, _ in parsed}) != len(parsed):
        raise ValueError("duplicate thresholds")
    if len(parsed) < 2:
        raise ValueError("at least two thresholds are required")
    return parsed


def discover_images(root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for path in sorted(root.rglob("*.nii.gz")):
        if not path.is_file():
            continue
        key = prediction_name(path.name)
        if key in discovered:
            raise ValueError(f"duplicate prediction name {key}: {path}")
        discovered[key] = path
    return discovered


def evenly_spaced(names: list[str], maximum: int) -> list[str]:
    if maximum < 1:
        raise ValueError("max-cases must be positive")
    if len(names) <= maximum:
        return names
    indices = np.linspace(0, len(names) - 1, num=maximum, dtype=int)
    return [names[int(index)] for index in np.unique(indices)]


def audit(
    image_root: Path,
    threshold_dirs: list[tuple[float, Path]],
    max_cases: int,
    slab_depth: int = 32,
) -> dict[str, object]:
    if slab_depth < 1:
        raise ValueError("slab_depth must be positive")
    images = discover_images(image_root)
    inventories = {
        threshold: {
            path.name: path
            for path in directory.glob("*.nii.gz")
            if path.is_file() and path.name in images
        }
        for threshold, directory in threshold_dirs
    }
    common = sorted(set.intersection(*(set(value) for value in inventories.values())))
    if not common:
        raise RuntimeError("no completed names are common to all threshold directories")
    selected = evenly_spaced(common, max_cases)

    threshold_stats = {
        str(threshold): {"nonempty_cases": 0, "foreground_voxels": 0}
        for threshold, _ in threshold_dirs
    }
    errors: list[str] = []
    cases: list[dict[str, object]] = []
    for name in selected:
        source = nib.load(str(images[name]))
        threshold_images: list[nib.spatialimages.SpatialImage | None] = []
        foreground_counts = [0 for _ in threshold_dirs]
        nonbinary = [False for _ in threshold_dirs]
        nesting_violation = [False for _ in range(len(threshold_dirs) - 1)]
        for threshold, _ in threshold_dirs:
            path = inventories[threshold][name]
            image = nib.load(str(path))
            if image.shape != source.shape:
                errors.append(
                    f"{name}: shape mismatch at {threshold}: "
                    f"{image.shape} != {source.shape}"
                )
                threshold_images.append(None)
                continue
            if not np.allclose(image.affine, source.affine, atol=1e-4):
                errors.append(f"{name}: affine mismatch at {threshold}")
            threshold_images.append(image)

        # Large CT masks can contain hundreds of millions of voxels. Keeping
        # all threshold arrays resident at once made this read-only validator
        # compete with live inference for tens of GiB. Stream along the final
        # axis and retain only one small slab per threshold.
        for start in range(0, source.shape[-1], slab_depth):
            stop = min(source.shape[-1], start + slab_depth)
            slab_slice = (slice(None),) * (len(source.shape) - 1) + (
                slice(start, stop),
            )
            masks: list[np.ndarray | None] = []
            for index, image in enumerate(threshold_images):
                if image is None:
                    masks.append(None)
                    continue
                data = np.asanyarray(image.dataobj[slab_slice])
                binary = data > 0
                nonbinary[index] |= bool(np.any((data != 0) & (data != 1)))
                foreground_counts[index] += int(np.count_nonzero(binary))
                masks.append(binary)
                del data
            for index in range(1, len(masks)):
                lower, higher = masks[index - 1], masks[index]
                if lower is not None and higher is not None:
                    nesting_violation[index - 1] |= bool(np.any(higher & ~lower))

        foreground: dict[str, int] = {}
        for index, (threshold, _) in enumerate(threshold_dirs):
            if nonbinary[index]:
                errors.append(f"{name}: non-binary values at {threshold}")
            count = foreground_counts[index]
            threshold_stats[str(threshold)]["foreground_voxels"] += count
            threshold_stats[str(threshold)]["nonempty_cases"] += int(count > 0)
            foreground[str(threshold)] = count
        for index, violated in enumerate(nesting_violation, start=1):
            if violated:
                errors.append(
                    f"{name}: threshold {threshold_dirs[index][0]} is not "
                    f"nested inside {threshold_dirs[index - 1][0]}"
                )
        cases.append({"name": name, "foreground_voxels": foreground})

    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "live_snapshot": True,
        "image_inventory": len(images),
        "threshold_inventories": {
            str(threshold): len(inventories[threshold])
            for threshold, _ in threshold_dirs
        },
        "completed_common_inventory": len(common),
        "audited_cases": len(selected),
        "selection": "evenly_spaced_over_sorted_completed_common_inventory",
        "threshold_stats": threshold_stats,
        "errors": errors,
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    threshold_dirs = parse_threshold_dirs(args.threshold_dir)
    payload = audit(args.images, threshold_dirs, args.max_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: payload[key] for key in (
        "status", "image_inventory", "threshold_inventories",
        "completed_common_inventory", "audited_cases", "threshold_stats",
        "errors",
    )}, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
