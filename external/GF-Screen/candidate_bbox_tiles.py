"""Deterministic physical tiling for proposal-component refinement."""

from __future__ import annotations

import itertools
import math

import nibabel as nib
import numpy as np


def component_world_bounds(
    coordinates: np.ndarray, affine: np.ndarray, margin_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return an axis-aligned world box around voxel extents plus margin."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not len(coordinates):
        raise ValueError("coordinates must be a nonempty Nx3 array")
    if margin_mm < 0:
        raise ValueError("margin must be nonnegative")
    low = coordinates.min(axis=0) - 0.5
    high = coordinates.max(axis=0) + 0.5
    corners = np.asarray(
        list(itertools.product(*zip(low, high))), dtype=np.float64
    )
    world = nib.affines.apply_affine(affine, corners)
    return world.min(axis=0) - margin_mm, world.max(axis=0) + margin_mm


def axis_centers(low: float, high: float, field_mm: float, stride_mm: float) -> np.ndarray:
    """Centers whose fixed field covers [low, high] with gaps <= stride."""
    if high < low or field_mm <= 0 or stride_mm <= 0 or stride_mm > field_mm:
        raise ValueError("invalid bounds, field, or stride")
    span = high - low
    if span <= field_mm:
        return np.asarray([(low + high) / 2.0], dtype=np.float64)
    first = low + field_mm / 2.0
    last = high - field_mm / 2.0
    count = int(math.ceil((last - first) / stride_mm)) + 1
    return np.linspace(first, last, count, dtype=np.float64)


def tile_centers_world(
    low_world: np.ndarray,
    high_world: np.ndarray,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    stride_mm: float,
) -> list[np.ndarray]:
    """Enumerate lexicographically stable centers covering a world box."""
    low_world = np.asarray(low_world, dtype=np.float64)
    high_world = np.asarray(high_world, dtype=np.float64)
    if low_world.shape != (3,) or high_world.shape != (3,):
        raise ValueError("world bounds must be three-vectors")
    if len(shape) != 3 or len(spacing) != 3:
        raise ValueError("shape and spacing must have three values")
    field = np.asarray(shape, dtype=np.float64) * np.asarray(spacing, dtype=np.float64)
    axes = [
        axis_centers(float(low), float(high), float(size), stride_mm)
        for low, high, size in zip(low_world, high_world, field)
    ]
    return [np.asarray(center, dtype=np.float64) for center in itertools.product(*axes)]


def in_crop_target(
    target: np.ndarray,
    source_affine: np.ndarray,
    target_affine: np.ndarray,
    shape: tuple[int, int, int],
    resampled: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Preserve a sub-voxel target only when it physically intersects a tile."""
    result = np.asarray(resampled, dtype=bool).copy()
    if result.any() or not np.any(target):
        return result, False
    source_coordinates = np.argwhere(target)
    world = nib.affines.apply_affine(source_affine, source_coordinates)
    tile_coordinates = nib.affines.apply_affine(np.linalg.inv(target_affine), world)
    shape_array = np.asarray(shape, dtype=np.float64)
    inside = np.all((tile_coordinates >= -0.5) & (tile_coordinates < shape_array - 0.5), axis=1)
    if not np.any(inside):
        return result, False
    coordinate = np.rint(tile_coordinates[inside].mean(axis=0)).astype(int)
    coordinate = np.clip(coordinate, 0, np.asarray(shape) - 1)
    result[tuple(coordinate)] = True
    return result, True
