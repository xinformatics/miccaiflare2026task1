#!/usr/bin/env python3
"""Read-only spot audit of the deterministic GF-Screen preprocessing cache."""

import argparse
import gc
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

from utils.data_utils import get_abdomen_ds, get_chest_ds, get_nolesion_ds


def dataset_args(cache_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=".",
        abdomen_json="jsons/local/abdomen.json",
        chest_json="jsons/local/chest.json",
        healthy_json="jsons/local/healthy.json",
        healthy_repeat=1,
        use_persistent_dataset=True,
        cache_dir=str(cache_dir),
        space_x=1.0,
        space_y=1.0,
        space_z=3.0,
        b_min=0.0,
        b_max=1.0,
        roi_x=96,
        roi_y=96,
        roi_z=64,
        sw_batch_size=2,
        RandFlipd_prob=0.2,
        RandRotate90d_prob=0.2,
        RandShiftIntensityd_prob=0.1,
    )


def cache_key(dataset, item) -> str:
    return dataset.hash_func(item).decode("utf-8") + dataset.transform_hash


def build_index(region: str, datasets, cache_dir: Path):
    indexed = {}
    for split, dataset in datasets:
        for item in dataset.data:
            key = cache_key(dataset, item)
            path = cache_dir / region / f"{key}.pt"
            if path.is_file() and path.stat().st_size > 0:
                indexed.setdefault(path, (split, dataset, item))
    return indexed


def choose_samples(indexed, count: int):
    # Avoid files that may still be in the live producer's working set. Cache
    # publication is atomic, but older files also avoid contending with writes.
    cutoff = time.time() - 300
    stable = sorted(
        (path for path in indexed if path.stat().st_mtime < cutoff),
        key=lambda path: path.stat().st_size,
    )
    if len(stable) < count:
        raise RuntimeError(f"only {len(stable)} stable cache files available")
    # Deterministic size-stratified sampling rather than auditing only tiny files.
    fractions = np.linspace(0.25, 0.75, count)
    return [stable[min(int(fraction * len(stable)), len(stable) - 1)] for fraction in fractions]


def tensor(value):
    return value.as_tensor() if hasattr(value, "as_tensor") else value


def audit_one(region: str, path: Path, indexed):
    split, dataset, item = indexed[path]
    cached = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    fresh = dataset._pre_transform(deepcopy(item))

    image = tensor(cached["image"])
    label = tensor(cached["label"])
    fresh_image = tensor(fresh["image"])
    fresh_label = tensor(fresh["label"])
    label_values = torch.unique(label).cpu().tolist()

    affine = np.asarray(fresh["image"].affine.cpu(), dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    orientation = "".join(nib.orientations.aff2axcodes(affine))

    checks = {
        "readable": True,
        "name_matches": cached.get("name") == item.get("name"),
        "shape_aligned": tuple(image.shape) == tuple(label.shape),
        "channel_first": image.ndim == 4 and image.shape[0] == 1,
        "minimum_roi": all(size >= minimum for size, minimum in zip(image.shape[1:], (96, 96, 64))),
        "finite_image": bool(torch.isfinite(image).all()),
        "intensity_0_1": float(image.min()) >= -1e-6 and float(image.max()) <= 1.0 + 1e-6,
        "binary_label": set(label_values).issubset({0.0, 1.0}),
        "spacing_1_1_3": bool(np.allclose(spacing, (1.0, 1.0, 3.0), atol=1e-4)),
        "orientation_ras": orientation == "RAS",
        "cache_matches_fresh": bool(torch.equal(image, fresh_image) and torch.equal(label, fresh_label)),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"{path}: failed checks: {', '.join(failed)}")

    result = {
        "region": region,
        "split": split,
        "case": item.get("name"),
        "cache": path.name,
        "size_mb": round(path.stat().st_size / (1024**2), 1),
        "shape": list(image.shape),
        "spacing": [round(value, 4) for value in spacing.tolist()],
        "orientation": orientation,
        "image_range": [round(float(image.min()), 6), round(float(image.max()), 6)],
        "label_values": label_values,
        "foreground_voxels": int(torch.count_nonzero(label)),
        "checks": "PASS",
    }
    del cached, fresh, image, label, fresh_image, fresh_label
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--per-region", type=int, default=2)
    args = parser.parse_args()

    ds_args = dataset_args(args.cache_dir.resolve())
    abdomen = get_abdomen_ds(ds_args)
    chest = get_chest_ds(ds_args)
    healthy = get_nolesion_ds(ds_args)
    groups = {
        "abdomen": [("train", abdomen[0]), ("validation", abdomen[1])],
        "chest": [("train", chest[0]), ("validation", chest[1])],
        "healthy": [("train", healthy[0]), ("validation", healthy[1])],
    }

    results = []
    for region, datasets in groups.items():
        indexed = build_index(region, datasets, args.cache_dir.resolve())
        for path in choose_samples(indexed, args.per_region):
            results.append(audit_one(region, path, indexed))

    for result in results:
        print(result)
    print(f"AUDIT PASS: {len(results)} cache files across {len(groups)} regions")


if __name__ == "__main__":
    main()
