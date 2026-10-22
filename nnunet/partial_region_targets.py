"""Partial-label target primitives isolated from active production trainers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy import ndimage


IGNORE_LABEL = 2


def scaled_physical_margins(
    full_margin_voxels: Sequence[int],
    spatial_shape: Sequence[int],
    full_shape: Sequence[int],
) -> tuple[int, int, int]:
    if not (len(full_margin_voxels) == len(spatial_shape) == len(full_shape) == 3):
        raise ValueError("margin and shape arguments must contain three axes")
    if any(int(value) <= 0 for value in full_margin_voxels):
        raise ValueError("full-resolution margins must be positive")
    if any(int(value) <= 0 for value in (*spatial_shape, *full_shape)):
        raise ValueError("spatial shapes must be positive")
    return tuple(
        max(1, int(np.ceil(int(margin) * int(size) / int(full))))
        for margin, size, full in zip(
            full_margin_voxels, spatial_shape, full_shape, strict=True
        )
    )


def shape_linf_band_target(
    target: torch.Tensor,
    full_shape: tuple[int, int, int],
    full_margin_voxels: tuple[int, int, int],
    ignore_label: int = IGNORE_LABEL,
) -> torch.Tensor:
    """Trust foreground and a shape-following anisotropic L-infinity band.

    The band is generated from the already augmented target, so its geometry
    matches the input patch.  Work is restricted to the foreground bounding
    region plus the requested margin.  Existing non-binary/ignore values remain
    ignored, and a foreground-empty patch becomes entirely ignored.
    """

    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError("target must have shape [B,1,D,H,W]")
    spatial_shape = tuple(int(value) for value in target.shape[2:])
    margins = scaled_physical_margins(
        full_margin_voxels, spatial_shape, full_shape
    )
    kernel = tuple(2 * margin + 1 for margin in margins)
    result = torch.full_like(target, int(ignore_label))

    for batch_index in range(target.shape[0]):
        source = target[batch_index, 0]
        foreground = source == 1
        coordinates = torch.where(foreground)
        if coordinates[0].numel() == 0:
            continue
        lower = [
            max(0, int(axis.min().item()) - margin)
            for axis, margin in zip(coordinates, margins, strict=True)
        ]
        upper = [
            min(size, int(axis.max().item()) + margin + 1)
            for axis, margin, size in zip(
                coordinates, margins, spatial_shape, strict=True
            )
        ]
        region = tuple(
            slice(start, stop) for start, stop in zip(lower, upper, strict=True)
        )
        foreground_crop = (
            foreground[region].detach().to(device="cpu").numpy().astype(bool, copy=False)
        )
        trusted_numpy = ndimage.maximum_filter(
            foreground_crop,
            size=kernel,
            mode="constant",
            cval=0,
        ).astype(bool, copy=False)
        trusted = torch.from_numpy(np.ascontiguousarray(trusted_numpy)).to(
            device=source.device
        )
        values = source[region]
        known_binary = (values == 0) | (values == 1)
        result[batch_index, 0][region] = torch.where(
            trusted & known_binary,
            values,
            torch.as_tensor(ignore_label, dtype=values.dtype, device=values.device),
        )
    return result
