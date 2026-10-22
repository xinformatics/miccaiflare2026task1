#!/usr/bin/env python3
"""Exact CPU mask composition used by the verified epoch-160 winner.

This module intentionally contains no model-loading code.  It turns the six
raw model products for one CT into the final binary prediction while preserving
the historical connectivity, physical-volume, and strict-threshold semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import cc3d
import numpy as np


@dataclass(frozen=True)
class WinnerRecipeStats:
    bbox_raw_voxels: int
    bbox_filtered_voxels: int
    prompt_confirmed_voxels: int
    guided_voxels: int
    cbg_confirmed_voxels: int
    touching_cbg_voxels: int
    final_voxels: int


def _binary(name: str, value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must be three-dimensional")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape mismatch: {array.shape} != {shape}")
    if array.dtype != np.bool_ and not np.isin(np.unique(array), (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return array.astype(bool, copy=False)


def _spacing(spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in spacing)
    if len(values) != 3 or not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("spacing must contain three positive finite values")
    return values


def filter_components_physical_6(
    mask: np.ndarray, spacing: tuple[float, float, float], minimum_mm3: float = 1000.0
) -> np.ndarray:
    """Reproduce the historical GF 1,000-mm3 filter (6-connectivity)."""

    mask = _binary("bbox_raw", mask)
    spacing = _spacing(spacing)
    if minimum_mm3 < 0:
        raise ValueError("minimum_mm3 must be non-negative")
    if minimum_mm3 <= 0 or not np.any(mask):
        return mask.copy()
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=6)
    volume_mm3 = np.bincount(labels.ravel()) * float(np.prod(spacing))
    keep = volume_mm3 >= float(minimum_mm3)
    keep[0] = False
    return keep[labels]


def confirm_whole_components_26(mask: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Keep a complete 26-connected component iff support overlaps it."""

    mask = _binary("mask", mask)
    support = _binary("support", support, mask.shape)
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    keep = np.zeros(int(labels.max()) + 1, dtype=bool)
    keep[np.unique(labels[mask & support])] = True
    keep[0] = False
    return keep[labels]


def add_guided_components(
    base: np.ndarray,
    guided: np.ndarray,
    prompt_support: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    maximum_mm3: float = 2000.0,
    minimum_support_fraction: float = 0.25,
    minimum_support_mm3: float = 100.0,
) -> np.ndarray:
    """Apply the frozen ``new2k_s25`` disconnected-component rule.

    Eligible guided components are 26-connected, completely disjoint from the
    immutable base, no larger than 2,000 mm3, and independently prompt-supported
    over at least 25% and 100 mm3.  Once accepted, the complete guided component
    is imported; support clips neither its contour nor its extent.
    """

    base = _binary("base", base)
    guided = _binary("guided", guided, base.shape)
    prompt_support = _binary("prompt_support", prompt_support, base.shape)
    spacing = _spacing(spacing)
    if maximum_mm3 < 0 or minimum_support_mm3 < 0:
        raise ValueError("physical-volume thresholds must be non-negative")
    if not 0 <= minimum_support_fraction <= 1:
        raise ValueError("minimum_support_fraction must be in [0, 1]")

    labels = cc3d.connected_components(guided.astype(np.uint8), connectivity=26)
    counts = np.bincount(labels.ravel())
    base_overlap = np.bincount(labels[base], minlength=len(counts))
    support_counts = np.bincount(labels[prompt_support], minlength=len(counts))
    voxel_mm3 = float(np.prod(spacing))
    volumes = counts * voxel_mm3
    support_volumes = support_counts * voxel_mm3
    support_fraction = np.divide(
        support_counts,
        counts,
        out=np.zeros_like(volumes, dtype=np.float64),
        where=counts > 0,
    )
    keep = base_overlap == 0
    keep &= volumes <= float(maximum_mm3)
    keep &= support_fraction >= float(minimum_support_fraction)
    keep &= support_volumes >= float(minimum_support_mm3)
    keep[0] = False
    return base | keep[labels]


def union_touching_support_components_26(
    anchor: np.ndarray, support: np.ndarray
) -> np.ndarray:
    """Import complete support components that already touch the anchor."""

    anchor = _binary("anchor", anchor)
    support = _binary("support", support, anchor.shape)
    labels = cc3d.connected_components(support.astype(np.uint8), connectivity=26)
    keep = np.zeros(int(labels.max()) + 1, dtype=bool)
    keep[np.unique(labels[anchor])] = True
    keep[0] = False
    return anchor | keep[labels]


def confirm_with_freetumor_or_stunet_p70(
    anchor: np.ndarray,
    freetumor: np.ndarray,
    stunet_probability_uint16: np.ndarray,
) -> np.ndarray:
    """Apply the final exact FreeTumor OR quantized-STU p70 component gate."""

    anchor = _binary("anchor", anchor)
    freetumor = _binary("freetumor", freetumor, anchor.shape)
    probability = np.asarray(stunet_probability_uint16)
    if probability.shape != anchor.shape or probability.dtype != np.uint16:
        raise ValueError("STU probability must be an anchor-shaped uint16 array")
    # Historical materialization dequantized by /65535 then used strict > 0.70.
    stunet = probability.astype(np.float32) / 65535.0 > 0.70
    return confirm_whole_components_26(anchor, freetumor | stunet)


def compose_winner_mask(
    *,
    bbox_raw: np.ndarray,
    prompt_whole_body_lesion: np.ndarray,
    prompt_tumor: np.ndarray,
    guided_gf: np.ndarray,
    cbg_gf: np.ndarray,
    freetumor: np.ndarray,
    stunet_probability_uint16: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, WinnerRecipeStats]:
    """Compose one final winner mask from raw model outputs."""

    bbox_raw = _binary("bbox_raw", bbox_raw)
    prompt_whole_body_lesion = _binary(
        "prompt_whole_body_lesion", prompt_whole_body_lesion, bbox_raw.shape
    )
    prompt_tumor = _binary("prompt_tumor", prompt_tumor, bbox_raw.shape)
    guided_gf = _binary("guided_gf", guided_gf, bbox_raw.shape)
    cbg_gf = _binary("cbg_gf", cbg_gf, bbox_raw.shape)
    freetumor = _binary("freetumor", freetumor, bbox_raw.shape)

    bbox_filtered = filter_components_physical_6(bbox_raw, spacing, 1000.0)
    prompt_support = prompt_whole_body_lesion | prompt_tumor
    prompt_confirmed = confirm_whole_components_26(bbox_filtered, prompt_support)
    guided = add_guided_components(prompt_confirmed, guided_gf, prompt_support, spacing)
    cbg_confirmed = confirm_whole_components_26(guided, cbg_gf)
    touching_cbg = union_touching_support_components_26(cbg_confirmed, cbg_gf)
    final = confirm_with_freetumor_or_stunet_p70(
        touching_cbg, freetumor, stunet_probability_uint16
    )
    return final, WinnerRecipeStats(
        bbox_raw_voxels=int(bbox_raw.sum()),
        bbox_filtered_voxels=int(bbox_filtered.sum()),
        prompt_confirmed_voxels=int(prompt_confirmed.sum()),
        guided_voxels=int(guided.sum()),
        cbg_confirmed_voxels=int(cbg_confirmed.sum()),
        touching_cbg_voxels=int(touching_cbg.sum()),
        final_voxels=int(final.sum()),
    )
