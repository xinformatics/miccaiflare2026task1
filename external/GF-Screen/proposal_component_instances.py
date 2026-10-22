"""Deterministic component inventories for multi-source proposal masks.

The deployed candidate-refiner historically connected the voxelwise union of
all proposal sources.  That is retained by callers as the default policy.
This module implements the opt-in alternative: connect each source
independently, retain overlapping but non-identical alternatives, and remove
only voxel-identical duplicates.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import cc3d
import numpy as np


def _validate_masks(source_masks: Sequence[np.ndarray]) -> tuple[int, int, int]:
    if not source_masks:
        raise ValueError("at least one proposal source is required")
    shape = tuple(int(value) for value in np.asarray(source_masks[0]).shape)
    if len(shape) != 3:
        raise ValueError(f"proposal masks must be 3-D, got {shape}")
    for index, mask in enumerate(source_masks):
        if tuple(np.asarray(mask).shape) != shape:
            raise ValueError(
                f"proposal source {index} shape mismatch: "
                f"{tuple(np.asarray(mask).shape)} != {shape}"
            )
    return shape


def _component_digest(local: np.ndarray, bbox: tuple[slice, slice, slice]) -> bytes:
    """Fingerprint both placement and exact voxels of one local component."""
    digest = hashlib.sha256()
    bounds = np.asarray(
        [(int(axis.start), int(axis.stop)) for axis in bbox], dtype=np.int64
    )
    digest.update(bounds.tobytes())
    digest.update(np.asarray(local.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(local.reshape(-1), bitorder="little").tobytes())
    return digest.digest()


def source_separated_instances(
    source_masks: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Label each source separately and return a stable, exact-deduplicated list.

    Global ``component_id`` values are consecutive in source order and then in
    cc3d component order.  Exact duplicates are represented once, with all
    contributing source/component pairs recorded in ``equivalent_components``.
    Partially overlapping or nested components are deliberately retained.
    """
    _validate_masks(source_masks)
    component_maps: list[np.ndarray] = []
    instances: list[dict[str, object]] = []
    fingerprints: dict[bytes, list[int]] = {}

    for source_index, source_mask in enumerate(source_masks):
        component_map, count = cc3d.connected_components(
            np.asarray(source_mask, dtype=np.uint8),
            connectivity=26,
            return_N=True,
        )
        component_maps.append(component_map)
        statistics = cc3d.statistics(component_map)
        for source_component_id in range(1, count + 1):
            bbox = tuple(statistics["bounding_boxes"][source_component_id])
            local = component_map[bbox] == source_component_id
            fingerprint = _component_digest(local, bbox)
            duplicate_index = None
            for candidate_index in fingerprints.get(fingerprint, []):
                candidate = instances[candidate_index]
                candidate_bbox = tuple(candidate["bbox"])
                if candidate_bbox != bbox:
                    continue
                candidate_local = (
                    component_maps[int(candidate["source_index"])][candidate_bbox]
                    == int(candidate["source_component_id"])
                )
                if np.array_equal(candidate_local, local):
                    duplicate_index = candidate_index
                    break
            identity = {
                "source_index": source_index,
                "source_component_id": source_component_id,
            }
            if duplicate_index is not None:
                instances[duplicate_index]["equivalent_components"].append(identity)
                continue

            instance = {
                "component_id": len(instances) + 1,
                "source_index": source_index,
                "source_component_id": source_component_id,
                "equivalent_components": [identity],
                "bbox": bbox,
                "centroid_voxel": np.asarray(
                    statistics["centroids"][source_component_id], dtype=np.float64
                ),
                "voxels": int(statistics["voxel_counts"][source_component_id]),
            }
            instances.append(instance)
            fingerprints.setdefault(fingerprint, []).append(len(instances) - 1)

    return component_maps, instances


def instance_mask(
    component_maps: Sequence[np.ndarray], instance: dict[str, object]
) -> np.ndarray:
    """Reconstruct the full-volume mask for an inventory entry."""
    source_index = int(instance["source_index"])
    source_component_id = int(instance["source_component_id"])
    if not 0 <= source_index < len(component_maps):
        raise ValueError(f"invalid component source index: {source_index}")
    return component_maps[source_index] == source_component_id


def instance_local_mask(
    component_maps: Sequence[np.ndarray], instance: dict[str, object]
) -> np.ndarray:
    """Reconstruct an entry only inside its stored tight bounding box."""
    bbox = tuple(instance["bbox"])
    source_index = int(instance["source_index"])
    source_component_id = int(instance["source_component_id"])
    return component_maps[source_index][bbox] == source_component_id
