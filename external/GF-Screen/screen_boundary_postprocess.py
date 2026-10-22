#!/usr/bin/env python3
"""Bounded physical-space morphology screen for deployed lesion masks."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "handoff_bundle" / "code"))
from evaluate import score_case  # noqa: E402


CONFIGS = (
    "identity",
    "fill_holes_3d",
    "closing_1mm",
    "closing_2mm",
    "fill_then_closing_1mm",
    "opening_1mm",
    "erosion_1mm",
)


def physical_ball(spacing: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    spacing_array = np.asarray(spacing, dtype=np.float64)
    radii = np.ceil(float(radius_mm) / spacing_array).astype(int)
    offsets = np.ogrid[tuple(slice(-radius, radius + 1) for radius in radii)]
    squared = sum(
        (offset * axis_spacing) ** 2
        for offset, axis_spacing in zip(offsets, spacing_array)
    )
    return squared <= float(radius_mm) ** 2 + 1e-6


def componentwise_transform(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    config: str,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if config == "identity":
        return mask.copy()
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    output = np.zeros_like(mask)
    objects = ndimage.find_objects(labels)
    radius = 0.0
    if config.endswith("1mm"):
        radius = 1.0
    elif config.endswith("2mm"):
        radius = 2.0
    footprint = physical_ball(spacing, radius) if radius else None
    padding = tuple((size // 2, size // 2) for size in footprint.shape) if footprint is not None else None
    for component_id, component_slice in enumerate(objects, 1):
        if component_slice is None:
            continue
        component = labels[component_slice] == component_id
        if padding is not None:
            working = np.pad(component, padding, mode="constant")
        else:
            working = component
        if config == "fill_holes_3d":
            transformed = ndimage.binary_fill_holes(working)
        elif config == "closing_1mm" or config == "closing_2mm":
            transformed = ndimage.binary_closing(working, structure=footprint)
        elif config == "fill_then_closing_1mm":
            transformed = ndimage.binary_closing(
                ndimage.binary_fill_holes(working), structure=footprint
            )
        elif config == "opening_1mm":
            transformed = ndimage.binary_opening(working, structure=footprint)
        elif config == "erosion_1mm":
            transformed = ndimage.binary_erosion(working, structure=footprint)
        else:
            raise ValueError(f"unknown configuration: {config}")
        if padding is not None:
            transformed = transformed[
                tuple(slice(before, transformed.shape[axis] - after) for axis, (before, after) in enumerate(padding))
            ]
        output[component_slice] |= transformed
    return output


def score_one(item: tuple[str, str, str]) -> tuple[str, list[tuple[str, float, float, int]]]:
    name, prediction_path, target_path = item
    prediction_image = nib.load(prediction_path)
    target_image = nib.load(target_path)
    if prediction_image.shape != target_image.shape or not np.allclose(
        prediction_image.affine, target_image.affine, atol=1e-3
    ):
        raise ValueError(f"geometry mismatch: {name}")
    prediction = np.asanyarray(prediction_image.dataobj).astype(bool)
    target = np.asanyarray(target_image.dataobj).astype(bool)
    spacing = tuple(float(value) for value in target_image.header.get_zooms()[:3])
    values = []
    for config in CONFIGS:
        transformed = componentwise_transform(prediction, spacing, config)
        dsc, nsd = score_case(
            target.astype(np.uint8), transformed.astype(np.uint8), spacing
        )
        values.append((config, float(dsc), float(nsd), int(transformed.sum())))
    return name.removesuffix(".nii.gz"), values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    names = sorted(path.name for path in args.labels.glob("*.nii.gz"))
    if not names or args.workers < 1:
        parser.error("labels must be nonempty and workers must be positive")
    missing = [name for name in names if not (args.predictions / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing predictions; first={missing[0]}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {config: [] for config in CONFIGS}
    items = [
        (name, str(args.predictions / name), str(args.labels / name)) for name in names
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (case, values) in enumerate(executor.map(score_one, items), 1):
            for config, dsc, nsd, foreground in values:
                rows[config].append((case, dsc, nsd, foreground))
            print(f"[{index:03d}/{len(names):03d}] {case}", flush=True)
    summary = []
    for config in CONFIGS:
        path = args.output / f"{config}.metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD", "pred_fg_voxels"))
            for row in rows[config]:
                writer.writerow((row[0], round(row[1], 4), round(row[2], 4), row[3]))
        dsc = float(np.mean([row[1] for row in rows[config]]))
        nsd = float(np.mean([row[2] for row in rows[config]]))
        summary.append((config, dsc, nsd, (dsc + nsd) / 2, str(path)))
    summary.sort(key=lambda row: (-row[3], -row[1], row[0]))
    with (args.output / "summary.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("config", "DSC", "NSD", "combined", "metrics"))
        writer.writerows(summary)
    for config, dsc, nsd, _, _ in summary:
        print(f"{config}\tDSC={dsc:.6f}\tNSD={nsd:.6f}")


if __name__ == "__main__":
    main()
