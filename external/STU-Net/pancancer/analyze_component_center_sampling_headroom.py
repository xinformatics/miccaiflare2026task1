#!/usr/bin/env python3
"""Quantify lesion-size sampling bias in the real preprocessed training masks.

This is an analysis-only, deterministic sample. It compares nnU-Net's current
foreground-voxel center distribution with a lesion-uniform (component first,
then voxel) alternative. It never changes preprocessed data or reads challenge
validation/hidden data.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import blosc2
import numpy as np
from scipy import ndimage


PROXY_BOUNDS = (100, 500, 1_000, 3_000, 9_999)
LESION_MM3_BOUNDS = (100.0, 500.0, 1_000.0, 3_000.0)
VOXEL_MM3 = 2.4 * 1.6 * 1.6


def bin_index(value: float, bounds) -> int:
    for index, upper in enumerate(bounds):
        if value < upper:
            return index
    return len(bounds)


def foreground_proxy_count(properties: dict) -> int:
    for key, value in properties.get("class_locations", {}).items():
        if str(key) in {"1", "(1,)"}:
            return int(len(value))
    return 0


def component_sizes(segmentation: np.ndarray) -> np.ndarray:
    foreground = np.asarray(segmentation[0] == 1)
    coordinates = np.where(foreground)
    if coordinates[0].size == 0:
        return np.empty(0, dtype=np.int64)
    region = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates
    )
    labeled, count = ndimage.label(
        foreground[region], structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    return np.bincount(labeled.ravel(), minlength=count + 1)[1:].astype(np.int64)


def selection_mass(sizes_voxels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    voxel_mass = np.zeros(len(LESION_MM3_BOUNDS) + 1, dtype=np.float64)
    component_mass = np.zeros_like(voxel_mass)
    if sizes_voxels.size == 0:
        return voxel_mass, component_mass
    sizes_mm3 = sizes_voxels.astype(np.float64) * VOXEL_MM3
    bins = np.asarray([bin_index(value, LESION_MM3_BOUNDS) for value in sizes_mm3])
    for index in range(voxel_mass.size):
        mask = bins == index
        voxel_mass[index] = sizes_voxels[mask].sum() / sizes_voxels.sum()
        component_mass[index] = mask.sum() / sizes_voxels.size
    return voxel_mass, component_mass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-proxy-bin", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    folder = args.preprocessed.resolve()
    identifiers = sorted(
        path.name.removesuffix(".pkl")
        for path in folder.glob("*.pkl")
        if (folder / f"{path.name.removesuffix('.pkl')}_seg.b2nd").is_file()
    )
    if len(identifiers) != 17_575:
        raise SystemExit(f"expected 17575 cases, found {len(identifiers)}")
    proxy_bins = {index: [] for index in range(len(PROXY_BOUNDS) + 1)}
    for identifier in identifiers:
        with (folder / f"{identifier}.pkl").open("rb") as handle:
            count = foreground_proxy_count(pickle.load(handle))
        proxy_bins[bin_index(count, PROXY_BOUNDS)].append(identifier)

    rng = np.random.default_rng(args.seed)
    sampled = []
    for index, values in proxy_bins.items():
        if len(values) < args.per_proxy_bin:
            raise SystemExit(f"proxy bin {index} has only {len(values)} cases")
        chosen = rng.choice(values, size=args.per_proxy_bin, replace=False)
        sampled.extend((index, str(value)) for value in chosen)
    sampled.sort(key=lambda item: (item[0], item[1]))

    voxel_masses = []
    component_masses = []
    multi_voxel_masses = []
    multi_component_masses = []
    component_sizes_all = []
    cases = []
    started = time.monotonic()
    for number, (proxy_bin, identifier) in enumerate(sampled, start=1):
        segmentation = blosc2.open(
            urlpath=str(folder / f"{identifier}_seg.b2nd"),
            mode="r",
            dparams={"nthreads": 1},
            mmap_mode="r",
        )
        sizes = component_sizes(np.asarray(segmentation))
        voxel_mass, component_mass = selection_mass(sizes)
        voxel_masses.append(voxel_mass)
        component_masses.append(component_mass)
        if sizes.size > 1:
            multi_voxel_masses.append(voxel_mass)
            multi_component_masses.append(component_mass)
        component_sizes_all.extend((sizes.astype(np.float64) * VOXEL_MM3).tolist())
        cases.append(
            {
                "identifier": identifier,
                "proxy_bin": proxy_bin,
                "component_count": int(sizes.size),
                "foreground_voxels": int(sizes.sum()),
            }
        )
        print(f"component-center audit {number}/{len(sampled)}: {identifier}", flush=True)

    labels = ["lt100", "100to500", "500to1000", "1000to3000", "ge3000"]
    mean = lambda values: np.asarray(values).mean(axis=0).tolist()
    sizes_array = np.asarray(component_sizes_all, dtype=np.float64)
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": "ANALYSIS_ONLY_NO_TRAINING_CHANGE",
        "case_count": len(sampled),
        "per_proxy_bin": args.per_proxy_bin,
        "seed": args.seed,
        "identifier_inventory_sha256": hashlib.sha256(
            ("\n".join(identifiers) + "\n").encode()
        ).hexdigest(),
        "sample_sha256": hashlib.sha256(
            ("\n".join(identifier for _, identifier in sampled) + "\n").encode()
        ).hexdigest(),
        "component_count": int(sizes_array.size),
        "multi_component_cases": int(sum(case["component_count"] > 1 for case in cases)),
        "lesion_size_bins_mm3": labels,
        "mean_case_selection_mass": {
            "foreground_voxel_uniform": dict(zip(labels, mean(voxel_masses), strict=True)),
            "component_uniform": dict(zip(labels, mean(component_masses), strict=True)),
        },
        "mean_multi_component_case_selection_mass": {
            "foreground_voxel_uniform": dict(zip(labels, mean(multi_voxel_masses), strict=True)),
            "component_uniform": dict(zip(labels, mean(multi_component_masses), strict=True)),
        },
        "component_size_mm3": {
            "median": float(np.median(sizes_array)),
            "p10": float(np.percentile(sizes_array, 10)),
            "p90": float(np.percentile(sizes_array, 90)),
        },
        "runtime_seconds": time.monotonic() - started,
        "cases": cases,
        "training_data_read": True,
        "training_data_modified": False,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
