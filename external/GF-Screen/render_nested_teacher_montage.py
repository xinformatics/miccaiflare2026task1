#!/usr/bin/env python3
"""Render orthogonal CT overlays for nested teacher-mask quality control."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def selected_slices(mask: np.ndarray, axis: int, count: int = 4) -> list[int]:
    coordinates = np.nonzero(mask)[axis]
    if coordinates.size == 0:
        return [mask.shape[axis] // 2]
    quantiles = np.linspace(0.15, 0.85, count)
    return sorted({int(round(value)) for value in np.quantile(coordinates, quantiles)})


def display_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(volume, index, axis=axis))


def _weighted_slice_indices(counts: np.ndarray, count: int = 4) -> list[int]:
    """Choose foreground quantiles from a one-dimensional occupancy profile."""
    total = int(counts.sum())
    if total == 0:
        return [len(counts) // 2]
    cumulative = np.cumsum(counts, dtype=np.int64)
    targets = np.linspace(0.15, 0.85, count) * max(total - 1, 0)
    return sorted(
        {
            int(np.searchsorted(cumulative, target + 1, side="left"))
            for target in targets
        }
    )


def _stream_mask_indices(
    support_proxy,
    seed_proxy,
    shape: tuple[int, ...],
    count: int = 4,
    slab_depth: int = 16,
) -> dict[int, list[int]]:
    """Validate nested binary masks and select slices without full-volume loads."""
    occupancy = [np.zeros(length, dtype=np.int64) for length in shape]
    support_voxels = 0
    for start in range(0, shape[2], slab_depth):
        stop = min(shape[2], start + slab_depth)
        support = np.asanyarray(support_proxy[:, :, start:stop])
        seed = np.asanyarray(seed_proxy[:, :, start:stop])
        if np.any((support != 0) & (support != 1)):
            raise ValueError("support mask is not binary")
        if np.any((seed != 0) & (seed != 1)):
            raise ValueError("seed mask is not binary")
        support = support > 0
        seed = seed > 0
        if np.any(seed & ~support):
            raise ValueError("seed mask is not nested inside support")
        occupancy[0] += support.sum(axis=(1, 2), dtype=np.int64)
        occupancy[1] += support.sum(axis=(0, 2), dtype=np.int64)
        occupancy[2][start:stop] = support.sum(axis=(0, 1), dtype=np.int64)
        support_voxels += int(support.sum())
    if support_voxels == 0:
        raise ValueError("support mask is empty")
    return {
        axis: _weighted_slice_indices(axis_counts, count=count)
        for axis, axis_counts in enumerate(occupancy)
    }


def _proxy_slice(proxy, axis: int, index: int) -> np.ndarray:
    slices = [slice(None)] * 3
    slices[axis] = index
    return np.rot90(np.asanyarray(proxy[tuple(slices)]))


def render(
    image_path: Path,
    support_path: Path,
    seed_path: Path,
    output_path: Path,
    window_min: float = -175.0,
    window_max: float = 250.0,
) -> None:
    source_image = nib.load(str(image_path))
    support_image = nib.load(str(support_path))
    seed_image = nib.load(str(seed_path))
    for label, candidate in (("support", support_image), ("seed", seed_image)):
        if candidate.shape != source_image.shape:
            raise ValueError(f"{label}/source shape mismatch")
        if not np.allclose(candidate.affine, source_image.affine, atol=1e-4):
            raise ValueError(f"{label}/source affine mismatch")
    # Challenge CTs can exceed the process memory limit when the image and two
    # masks are materialized together. Validate masks in bounded z slabs, then
    # read only the twelve displayed planes from nibabel's array proxies.
    indices = _stream_mask_indices(
        support_image.dataobj, seed_image.dataobj, source_image.shape
    )
    columns = max(len(values) for values in indices.values())
    figure, axes = plt.subplots(3, columns, figsize=(4 * columns, 12), squeeze=False)
    axis_names = ("sagittal", "coronal", "axial")
    for axis, name in enumerate(axis_names):
        for column in range(columns):
            panel = axes[axis, column]
            panel.axis("off")
            if column >= len(indices[axis]):
                continue
            index = indices[axis][column]
            ct_slice = _proxy_slice(source_image.dataobj, axis, index).astype(
                np.float32, copy=False
            )
            support_slice = _proxy_slice(support_image.dataobj, axis, index) > 0
            seed_slice = _proxy_slice(seed_image.dataobj, axis, index) > 0
            panel.imshow(ct_slice, cmap="gray", vmin=window_min, vmax=window_max)
            overlay = np.zeros((*ct_slice.shape, 4), dtype=np.float32)
            overlay[support_slice] = (1.0, 0.0, 0.0, 0.35)
            overlay[seed_slice] = (1.0, 1.0, 0.0, 0.70)
            panel.imshow(overlay, interpolation="nearest")
            panel.set_title(f"{name} index {index}")
    figure.suptitle(
        f"{image_path.name}: support red, high-confidence seed yellow",
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-min", type=float, default=-175.0)
    parser.add_argument("--window-max", type=float, default=250.0)
    args = parser.parse_args()
    if args.window_min >= args.window_max:
        parser.error("window-min must be smaller than window-max")
    render(
        args.image,
        args.support,
        args.seed,
        args.output,
        args.window_min,
        args.window_max,
    )


if __name__ == "__main__":
    main()
