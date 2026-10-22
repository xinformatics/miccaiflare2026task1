#!/usr/bin/env python3
"""Evaluate two GF Glance policies with one shared Focus crop cache.

The bbox and CBG deployment checkpoints have bit-identical Focus tensors. This
module keeps their Glance decisions separate, evaluates Focus once for every
crop selected by either route, and accumulates independent output masks.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class SharedFocusStats:
    grid_crops: int
    route_a_crops: int
    route_b_crops: int
    union_crops: int
    reused_crops: int


def crop_coordinates(
    shape: Sequence[int],
    crop: Sequence[int],
    overlaps: Sequence[float],
) -> list[tuple[int, int, int]]:
    if len(shape) != 3 or len(crop) != 3 or len(overlaps) != 3:
        raise ValueError("shape, crop, and overlaps must be three-dimensional")
    if any(size < window for size, window in zip(shape, crop)):
        raise ValueError("volume must be padded to at least the crop shape")
    if any(not 0.0 <= value < 1.0 for value in overlaps):
        raise ValueError("overlap ratios must be in [0, 1)")
    overlap_voxels = tuple(int(window * value) for window, value in zip(crop, overlaps))
    strides = tuple(window - overlap for window, overlap in zip(crop, overlap_voxels))
    counts = tuple(
        (size - overlap + stride - 1) // stride
        for size, overlap, stride in zip(shape, overlap_voxels, strides)
    )
    coordinates: list[tuple[int, int, int]] = []
    for i in range(counts[0]):
        for j in range(counts[1]):
            for k in range(counts[2]):
                start = [i * strides[0], j * strides[1], k * strides[2]]
                for axis in range(3):
                    if start[axis] + crop[axis] > shape[axis]:
                        start[axis] = shape[axis] - crop[axis]
                point = tuple(start)
                if not coordinates or point != coordinates[-1]:
                    coordinates.append(point)
    return coordinates


def crop_batch(
    volume: torch.Tensor,
    coordinates: Sequence[tuple[int, int, int]],
    crop: Sequence[int],
) -> torch.Tensor:
    if volume.ndim != 4:
        raise ValueError("volume must have shape [channels, x, y, z]")
    cx, cy, cz = map(int, crop)
    return torch.stack(
        [volume[:, x : x + cx, y : y + cy, z : z + cz] for x, y, z in coordinates],
        dim=0,
    )


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def _glance_selection(
    glance_model,
    volume: torch.Tensor,
    coordinates: Sequence[tuple[int, int, int]],
    crop: Sequence[int],
    threshold: float,
    batch_size: int,
    device: torch.device,
    autocast: bool,
) -> np.ndarray:
    selected = np.zeros(len(coordinates), dtype=bool)
    for start in range(0, len(coordinates), batch_size):
        current = coordinates[start : start + batch_size]
        patches = crop_batch(volume, current, crop).to(device, non_blocking=True)
        with _autocast(device, autocast):
            actions = glance_model.get_actions(patches)
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise RuntimeError("Glance logits must have shape [batch, 2]")
        selected[start : start + len(current)] = (
            actions.softmax(1)[:, 1] > threshold
        ).detach().cpu().numpy()
        del patches, actions
    return selected


def dual_glance_shared_focus_predict(
    glance_a,
    glance_b,
    focus_model,
    volume: torch.Tensor,
    *,
    glance_threshold_a: float = 0.5,
    glance_threshold_b: float = 0.2,
    lesion_threshold: float = 0.5,
    crop: Sequence[int] = (96, 96, 64),
    overlaps: Sequence[float] = (0.75, 0.75, 0.75),
    glance_batch_size: int = 8,
    focus_batch_size: int = 4,
    device: torch.device | str | None = None,
    autocast: bool = True,
) -> tuple[np.ndarray, np.ndarray, SharedFocusStats]:
    """Return independent bbox/CBG masks with shared Focus evaluations."""

    if not 0.0 <= glance_threshold_a <= 1.0:
        raise ValueError("glance_threshold_a must be in [0, 1]")
    if not 0.0 <= glance_threshold_b <= 1.0:
        raise ValueError("glance_threshold_b must be in [0, 1]")
    if not 0.0 <= lesion_threshold <= 1.0:
        raise ValueError("lesion_threshold must be in [0, 1]")
    if glance_batch_size < 1 or focus_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    target = torch.device(device if device is not None else volume.device)
    shape = tuple(int(value) for value in volume.shape[1:])
    crop = tuple(int(value) for value in crop)
    coordinates = crop_coordinates(shape, crop, overlaps)

    with torch.inference_mode():
        selected_a = _glance_selection(
            glance_a, volume, coordinates, crop, glance_threshold_a,
            glance_batch_size, target, autocast,
        )
        selected_b = _glance_selection(
            glance_b, volume, coordinates, crop, glance_threshold_b,
            glance_batch_size, target, autocast,
        )
        selected_union = selected_a | selected_b
        union_indices = np.flatnonzero(selected_union).tolist()
        probability_a = np.zeros(shape, dtype=np.float32)
        probability_b = np.zeros(shape, dtype=np.float32)
        coverage_a = np.zeros(shape, dtype=np.uint16)
        coverage_b = np.zeros(shape, dtype=np.uint16)
        cx, cy, cz = crop
        for start in range(0, len(union_indices), focus_batch_size):
            indices = union_indices[start : start + focus_batch_size]
            current = [coordinates[index] for index in indices]
            patches = crop_batch(volume, current, crop).to(target, non_blocking=True)
            with _autocast(target, autocast):
                logits = focus_model(patches)
            if logits.ndim != 5 or logits.shape[1] != 2:
                raise RuntimeError("Focus logits must have shape [batch, 2, x, y, z]")
            foreground = logits.softmax(1)[:, 1].float().detach().cpu().numpy()
            for probability, index, (x, y, z) in zip(foreground, indices, current):
                if selected_a[index]:
                    probability_a[x : x + cx, y : y + cy, z : z + cz] += probability
                    coverage_a[x : x + cx, y : y + cy, z : z + cz] += 1
                if selected_b[index]:
                    probability_b[x : x + cx, y : y + cy, z : z + cz] += probability
                    coverage_b[x : x + cx, y : y + cy, z : z + cz] += 1
            del patches, logits, foreground

    mask_a = (coverage_a > 0) & (probability_a >= lesion_threshold * coverage_a)
    mask_b = (coverage_b > 0) & (probability_b >= lesion_threshold * coverage_b)
    count_a = int(selected_a.sum())
    count_b = int(selected_b.sum())
    count_union = int(selected_union.sum())
    return mask_a, mask_b, SharedFocusStats(
        grid_crops=len(coordinates),
        route_a_crops=count_a,
        route_b_crops=count_b,
        union_crops=count_union,
        reused_crops=count_a + count_b - count_union,
    )

