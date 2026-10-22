#!/usr/bin/env python3
"""Runtime-oriented final confirmation without a FreeTumor forward pass."""

from __future__ import annotations

from dataclasses import dataclass

import cc3d
import numpy as np

from winner_mask_recipe import (
    add_guided_components,
    confirm_whole_components_26,
    filter_components_physical_6,
    union_touching_support_components_26,
)


@dataclass(frozen=True)
class RuntimeAnchorStats:
    bbox_raw_voxels: int
    bbox_filtered_voxels: int
    prompt_confirmed_voxels: int
    guided_voxels: int
    cbg_confirmed_voxels: int
    final_anchor_voxels: int


@dataclass(frozen=True)
class ProxyConfirmationStats:
    anchor_components: int
    stunet_confirmed_components: int
    proxy_confirmed_components: int
    rejected_components: int


def compose_runtime_anchor(
    *,
    bbox_raw: np.ndarray,
    prompt_whole_body_lesion: np.ndarray,
    prompt_tumor: np.ndarray,
    guided_gf: np.ndarray,
    cbg_gf: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, RuntimeAnchorStats]:
    """Compose the exact pre-STU winner anchor from runtime model products."""

    bbox_raw = _binary("bbox_raw", bbox_raw)
    whole = _binary("prompt_whole_body_lesion", prompt_whole_body_lesion, bbox_raw.shape)
    tumor = _binary("prompt_tumor", prompt_tumor, bbox_raw.shape)
    guided_gf = _binary("guided_gf", guided_gf, bbox_raw.shape)
    cbg_gf = _binary("cbg_gf", cbg_gf, bbox_raw.shape)
    bbox_filtered = filter_components_physical_6(bbox_raw, spacing, 1000.0)
    prompt_support = whole | tumor
    prompt_confirmed = confirm_whole_components_26(bbox_filtered, prompt_support)
    guided = add_guided_components(
        prompt_confirmed,
        guided_gf,
        prompt_support,
        spacing,
        maximum_mm3=2000.0,
        minimum_support_fraction=0.25,
        minimum_support_mm3=100.0,
    )
    cbg_confirmed = confirm_whole_components_26(guided, cbg_gf)
    anchor = union_touching_support_components_26(cbg_confirmed, cbg_gf)
    return anchor, RuntimeAnchorStats(
        bbox_raw_voxels=int(bbox_raw.sum()),
        bbox_filtered_voxels=int(bbox_filtered.sum()),
        prompt_confirmed_voxels=int(prompt_confirmed.sum()),
        guided_voxels=int(guided.sum()),
        cbg_confirmed_voxels=int(cbg_confirmed.sum()),
        final_anchor_voxels=int(anchor.sum()),
    )


def _binary(name: str, value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or (shape is not None and array.shape != shape):
        raise ValueError(f"{name} shape mismatch")
    if array.dtype != np.bool_ and not np.isin(np.unique(array), (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return array.astype(bool, copy=False)


def existing_support_proxy_mask(
    *,
    anchor: np.ndarray,
    prompt_whole_body_lesion: np.ndarray,
    prompt_tumor: np.ndarray,
    bbox_raw: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Return complete anchor components selected by the frozen cheap proxy.

    The proxy uses masks already produced earlier in the winner graph:

    * Tumor-prompt overlap fraction >= 0.40; or
    * whole-body-lesion overlap >= 0.40 and raw-bbox overlap < 0.50; or
    * whole-body-lesion overlap >= 0.75 and component volume >= 800 mm3.
    """

    anchor = _binary("anchor", anchor)
    whole = _binary("prompt_whole_body_lesion", prompt_whole_body_lesion, anchor.shape)
    tumor = _binary("prompt_tumor", prompt_tumor, anchor.shape)
    bbox = _binary("bbox_raw", bbox_raw, anchor.shape)
    spacing = tuple(float(value) for value in spacing)
    if (
        len(spacing) != 3
        or not np.isfinite(spacing).all()
        or min(spacing) <= 0
    ):
        raise ValueError("invalid spacing")
    labels = cc3d.connected_components(anchor.astype(np.uint8), connectivity=26)
    component_count = int(labels.max())
    counts = np.bincount(labels.ravel(), minlength=component_count + 1)

    def overlaps(mask: np.ndarray) -> np.ndarray:
        return np.bincount(labels[mask], minlength=component_count + 1)

    whole_counts = overlaps(whole)
    tumor_counts = overlaps(tumor)
    bbox_counts = overlaps(bbox)
    fractions = lambda values: np.divide(
        values,
        counts,
        out=np.zeros(component_count + 1, dtype=np.float64),
        where=counts > 0,
    )
    whole_fraction = fractions(whole_counts)
    tumor_fraction = fractions(tumor_counts)
    bbox_fraction = fractions(bbox_counts)
    volumes_mm3 = counts * float(np.prod(spacing))
    proxy_keep = (
        (tumor_fraction >= 0.40)
        | ((whole_fraction >= 0.40) & (bbox_fraction < 0.50))
        | ((whole_fraction >= 0.75) & (volumes_mm3 >= 800.0))
    )
    proxy_keep[0] = False
    return proxy_keep[labels]


def confirm_with_stunet_p70_or_existing_support_proxy(
    *,
    anchor: np.ndarray,
    prompt_whole_body_lesion: np.ndarray,
    prompt_tumor: np.ndarray,
    bbox_raw: np.ndarray,
    stunet_probability_uint16: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, ProxyConfirmationStats]:
    """Keep whole anchor components using STU or the frozen cheap proxy."""

    anchor = _binary("anchor", anchor)
    probability = np.asarray(stunet_probability_uint16)
    if probability.shape != anchor.shape or probability.dtype != np.uint16:
        raise ValueError("invalid STU probability")
    proxy_mask = existing_support_proxy_mask(
        anchor=anchor,
        prompt_whole_body_lesion=prompt_whole_body_lesion,
        prompt_tumor=prompt_tumor,
        bbox_raw=bbox_raw,
        spacing=spacing,
    )
    stunet = probability.astype(np.float32) / 65535.0 > 0.70
    labels = cc3d.connected_components(anchor.astype(np.uint8), connectivity=26)
    component_count = int(labels.max())
    stunet_counts = np.bincount(labels[stunet], minlength=component_count + 1)
    proxy_counts = np.bincount(labels[proxy_mask], minlength=component_count + 1)
    stunet_keep = stunet_counts > 0
    proxy_keep = (proxy_counts > 0) & ~stunet_keep
    keep = stunet_keep | proxy_keep
    keep[0] = False
    result = keep[labels]
    return result, ProxyConfirmationStats(
        anchor_components=component_count,
        stunet_confirmed_components=int(stunet_keep[1:].sum()),
        proxy_confirmed_components=int(proxy_keep[1:].sum()),
        rejected_components=int(component_count - keep[1:].sum()),
    )
