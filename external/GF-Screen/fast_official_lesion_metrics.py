#!/usr/bin/env python3
"""Exact DSC/2-mm NSD decision ladder without unnecessary full EDTs.

The organizer implementation computes complete sorted surface-distance arrays,
although the challenge consumes only whether each distance is <=2 mm. A
spacing-aware binary dilation answers that same threshold query exactly while
preserving the organizer's 2x2x2 surfel construction and area weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import scipy.ndimage
from scipy.spatial import cKDTree

import cc3d


OFFICIAL_CODE = Path(__file__).resolve().parents[2] / "handoff_bundle" / "code"
sys.path.insert(0, str(OFFICIAL_CODE))
from SurfaceDice import compute_dice_coefficient, neighbour_code_to_normals  # noqa: E402


TUMOR_TOLERANCE_MM = 2.0
NEIGHBOUR_KERNEL = np.asarray(
    [[[128, 64], [32, 16]], [[8, 4], [2, 1]]], dtype=np.uint8
)


@dataclass(frozen=True)
class SurfaceReference:
    physical_coordinates: np.ndarray
    areas: np.ndarray
    tree: cKDTree


@lru_cache(maxsize=64)
def surface_area_lookup(spacing: tuple[float, float, float]) -> np.ndarray:
    result = np.zeros(256, dtype=np.float64)
    for code, entries in enumerate(neighbour_code_to_normals):
        normals = np.asarray(entries, dtype=np.float64)
        scaled = normals * np.asarray(
            [spacing[1] * spacing[2], spacing[0] * spacing[2], spacing[0] * spacing[1]],
            dtype=np.float64,
        )
        result[code] = np.linalg.norm(scaled, axis=1).sum()
    return result


@lru_cache(maxsize=64)
def tolerance_structure(
    spacing: tuple[float, float, float], tolerance_mm: float
) -> np.ndarray:
    radii = np.ceil(tolerance_mm / np.asarray(spacing)).astype(int)
    coordinates = np.meshgrid(
        *(np.arange(-radius, radius + 1) for radius in radii), indexing="ij"
    )
    squared_distance = sum(
        (coordinate.astype(np.float64) * spacing[axis]) ** 2
        for axis, coordinate in enumerate(coordinates)
    )
    return squared_distance <= tolerance_mm**2 + 1e-12


def crop_union(mask_gt: np.ndarray, mask_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    locations = np.nonzero(mask_gt | mask_pred)
    if not locations[0].size:
        empty = np.zeros((0, 0, 0), dtype=np.uint8)
        return empty, empty
    low = np.asarray([axis.min() for axis in locations], dtype=np.int64)
    high = np.asarray([axis.max() for axis in locations], dtype=np.int64)
    shape = tuple((high - low + 2).tolist())
    cropped_gt = np.zeros(shape, dtype=np.uint8)
    cropped_pred = np.zeros(shape, dtype=np.uint8)
    source = tuple(slice(int(a), int(b + 1)) for a, b in zip(low, high))
    destination = tuple(slice(0, int(b - a + 1)) for a, b in zip(low, high))
    cropped_gt[destination] = mask_gt[source]
    cropped_pred[destination] = mask_pred[source]
    return cropped_gt, cropped_pred


def dense_surface_dice_at_tolerance(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float = TUMOR_TOLERANCE_MM,
) -> float:
    spacing = tuple(float(value) for value in spacing)
    cropped_gt, cropped_pred = crop_union(mask_gt.astype(bool), mask_pred.astype(bool))
    if not cropped_gt.size:
        return float("nan")
    codes_gt = scipy.ndimage.correlate(
        cropped_gt, NEIGHBOUR_KERNEL, mode="constant", cval=0
    )
    codes_pred = scipy.ndimage.correlate(
        cropped_pred, NEIGHBOUR_KERNEL, mode="constant", cval=0
    )
    borders_gt = (codes_gt != 0) & (codes_gt != 255)
    borders_pred = (codes_pred != 0) & (codes_pred != 255)
    structure = tolerance_structure(spacing, float(tolerance_mm))
    close_to_pred = scipy.ndimage.binary_dilation(borders_pred, structure=structure)
    close_to_gt = scipy.ndimage.binary_dilation(borders_gt, structure=structure)
    lookup = surface_area_lookup(spacing)
    areas_gt = lookup[codes_gt[borders_gt]]
    areas_pred = lookup[codes_pred[borders_pred]]
    overlap_gt = areas_gt[close_to_pred[borders_gt]].sum()
    overlap_pred = areas_pred[close_to_gt[borders_pred]].sum()
    return float((overlap_gt + overlap_pred) / (areas_gt.sum() + areas_pred.sum()))


def component_surfels(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract organizer-identical surfels one union component at a time."""

    union = mask_gt.astype(bool) | mask_pred.astype(bool)
    component_map = cc3d.connected_components(union.astype(np.uint8), connectivity=26)
    statistics = cc3d.statistics(component_map)
    lookup = surface_area_lookup(spacing)
    gt_coordinates: list[np.ndarray] = []
    pred_coordinates: list[np.ndarray] = []
    gt_areas: list[np.ndarray] = []
    pred_areas: list[np.ndarray] = []
    for component_id in range(1, int(component_map.max()) + 1):
        bbox = tuple(statistics["bounding_boxes"][component_id])
        component = component_map[bbox] == component_id
        local_shape = tuple(int(value) + 1 for value in component.shape)
        cropped_gt = np.zeros(local_shape, dtype=np.uint8)
        cropped_pred = np.zeros(local_shape, dtype=np.uint8)
        destination = tuple(slice(0, int(value)) for value in component.shape)
        cropped_gt[destination] = mask_gt[bbox].astype(bool) & component
        cropped_pred[destination] = mask_pred[bbox].astype(bool) & component
        codes_gt = scipy.ndimage.correlate(
            cropped_gt, NEIGHBOUR_KERNEL, mode="constant", cval=0
        )
        codes_pred = scipy.ndimage.correlate(
            cropped_pred, NEIGHBOUR_KERNEL, mode="constant", cval=0
        )
        borders_gt = (codes_gt != 0) & (codes_gt != 255)
        borders_pred = (codes_pred != 0) & (codes_pred != 255)
        offset = np.asarray([int(axis.start) for axis in bbox], dtype=np.int64)
        if borders_gt.any():
            gt_coordinates.append(np.argwhere(borders_gt) + offset)
            gt_areas.append(lookup[codes_gt[borders_gt]])
        if borders_pred.any():
            pred_coordinates.append(np.argwhere(borders_pred) + offset)
            pred_areas.append(lookup[codes_pred[borders_pred]])
    return (
        np.concatenate(gt_coordinates, axis=0),
        np.concatenate(pred_coordinates, axis=0),
        np.concatenate(gt_areas),
        np.concatenate(pred_areas),
    )


def mask_surfels(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a mask's organizer-identical surfels independently of its peer."""

    component_map = cc3d.connected_components(
        mask.astype(np.uint8), connectivity=26
    )
    count = int(component_map.max())
    if count == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64)
    statistics = cc3d.statistics(component_map)
    lookup = surface_area_lookup(spacing)
    coordinates: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    for component_id in range(1, count + 1):
        bbox = tuple(statistics["bounding_boxes"][component_id])
        component = component_map[bbox] == component_id
        local_shape = tuple(int(value) + 1 for value in component.shape)
        cropped = np.zeros(local_shape, dtype=np.uint8)
        destination = tuple(slice(0, int(value)) for value in component.shape)
        cropped[destination] = component
        codes = scipy.ndimage.correlate(
            cropped, NEIGHBOUR_KERNEL, mode="constant", cval=0
        )
        borders = (codes != 0) & (codes != 255)
        if borders.any():
            offset = np.asarray([int(axis.start) for axis in bbox], dtype=np.int64)
            coordinates.append(np.argwhere(borders) + offset)
            areas.append(lookup[codes[borders]])
    return np.concatenate(coordinates, axis=0), np.concatenate(areas)


def prepare_surface_reference(
    mask_gt: np.ndarray,
    spacing: tuple[float, float, float],
) -> SurfaceReference:
    """Prepare the immutable GT surface once for many predictions of one case."""

    spacing = tuple(float(value) for value in spacing)
    coordinates, areas = mask_surfels(mask_gt.astype(bool), spacing)
    if not coordinates.size:
        raise ValueError("surface reference requires nonempty ground truth")
    physical = coordinates.astype(np.float64) * np.asarray(spacing)
    return SurfaceReference(physical, areas, cKDTree(physical))


def surface_dice_with_reference(
    reference: SurfaceReference,
    mask_pred: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float = TUMOR_TOLERANCE_MM,
) -> float:
    """Exact NSD while reusing an already prepared ground-truth surface."""

    spacing = tuple(float(value) for value in spacing)
    coordinates_pred, areas_pred = mask_surfels(mask_pred.astype(bool), spacing)
    if not coordinates_pred.size:
        raise ValueError("surface comparison requires nonempty prediction")
    physical_pred = coordinates_pred.astype(np.float64) * np.asarray(spacing)
    pred_tree = cKDTree(physical_pred)
    distances_gt = pred_tree.query(
        reference.physical_coordinates, k=1, workers=1
    )[0]
    distances_pred = reference.tree.query(physical_pred, k=1, workers=1)[0]
    overlap_gt = reference.areas[distances_gt <= tolerance_mm].sum()
    overlap_pred = areas_pred[distances_pred <= tolerance_mm].sum()
    return float(
        (overlap_gt + overlap_pred) / (reference.areas.sum() + areas_pred.sum())
    )


def surface_dice_at_tolerance(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float = TUMOR_TOLERANCE_MM,
) -> float:
    """Exact tolerance-only NSD via component-local surfels and KD trees."""

    spacing = tuple(float(value) for value in spacing)
    coordinates_gt, coordinates_pred, areas_gt, areas_pred = component_surfels(
        mask_gt, mask_pred, spacing
    )
    physical_gt = coordinates_gt.astype(np.float64) * np.asarray(spacing)
    physical_pred = coordinates_pred.astype(np.float64) * np.asarray(spacing)
    pred_tree = cKDTree(physical_pred)
    gt_tree = cKDTree(physical_gt)
    # Do not use cKDTree.distance_upper_bound here: SciPy treats points exactly
    # on that boundary as missing on some versions, whereas the organizer uses
    # an inclusive <= tolerance comparison.
    distances_gt = pred_tree.query(physical_gt, k=1, workers=1)[0]
    distances_pred = gt_tree.query(physical_pred, k=1, workers=1)[0]
    overlap_gt = areas_gt[distances_gt <= tolerance_mm].sum()
    overlap_pred = areas_pred[distances_pred <= tolerance_mm].sum()
    return float((overlap_gt + overlap_pred) / (areas_gt.sum() + areas_pred.sum()))


def score_case(
    gt_data: np.ndarray, seg_data: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[float, float]:
    """Match the organizer's binary/empty/DSC<0.2 decision ladder exactly."""

    if np.max(seg_data) > 1:
        return 0.0, 0.0
    ground_truth_sum = np.sum(gt_data)
    segmentation_sum = np.sum(seg_data)
    if ground_truth_sum == 0 and segmentation_sum == 0:
        return 1.0, 1.0
    if ground_truth_sum == 0 or segmentation_sum == 0:
        return 0.0, 0.0
    dsc = float(compute_dice_coefficient(gt_data, seg_data))
    if dsc < 0.2:
        return dsc, 0.0
    return dsc, surface_dice_at_tolerance(gt_data, seg_data, spacing)


def score_case_with_reference(
    gt_data: np.ndarray,
    seg_data: np.ndarray,
    spacing: tuple[float, float, float],
    reference: SurfaceReference | None = None,
) -> tuple[float, float]:
    """Organizer decision ladder with reusable GT surfels for ensemble screens."""

    if np.max(seg_data) > 1:
        return 0.0, 0.0
    ground_truth_sum = np.sum(gt_data)
    segmentation_sum = np.sum(seg_data)
    if ground_truth_sum == 0 and segmentation_sum == 0:
        return 1.0, 1.0
    if ground_truth_sum == 0 or segmentation_sum == 0:
        return 0.0, 0.0
    dsc = float(compute_dice_coefficient(gt_data, seg_data))
    if dsc < 0.2:
        return dsc, 0.0
    prepared = reference or prepare_surface_reference(gt_data, spacing)
    return dsc, surface_dice_with_reference(prepared, seg_data, spacing)
