"""Auditable physical-location reweighting for a future Glance fallback.

This module is deliberately not wired into an active run. It defines a
bounded, group-mass-preserving sampling primitive that can be predeclared and
tested before a ResEnc endpoint determines whether the fallback is needed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _location_bins(locations: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.ndim != 1 or len(edges) < 3 or edges[0] != 0 or edges[-1] != 1:
        raise ValueError("location edges must span [0,1] with at least two bins")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("location edges must be strictly increasing")
    if np.any(~np.isfinite(locations)) or np.any((locations < 0) | (locations > 1)):
        raise ValueError("physical locations must be finite values in [0,1]")
    # Values exactly at 1 belong to the final bin.
    return np.minimum(np.searchsorted(edges, locations, side="right") - 1, len(edges) - 2)


def group_preserving_location_weights(
    locations: Sequence[float],
    groups: Sequence[str],
    base_weights: Sequence[float] | None = None,
    *,
    power: float = 0.5,
    max_factor: float = 5.0,
    edges: Sequence[float] = DEFAULT_EDGES,
) -> np.ndarray:
    """Return bounded location-balanced weights while preserving group mass.

    Quartile factors are computed globally as ``(largest_bin/count)^power``
    and capped. They are then normalized independently inside every supplied
    source/pseudo group, so multiplying existing cohort weights cannot change
    any group's total exposure.
    """

    locations = np.asarray(locations, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    if locations.ndim != 1 or groups.shape != locations.shape or not len(locations):
        raise ValueError("locations and groups must be aligned nonempty vectors")
    if not 0 <= float(power) <= 1:
        raise ValueError("location-balance power must be in [0,1]")
    if float(max_factor) < 1:
        raise ValueError("maximum location factor must be at least 1")
    if any(not str(group) for group in groups):
        raise ValueError("every location sample must have a nonempty group")
    base = (
        np.ones(len(locations), dtype=np.float64)
        if base_weights is None
        else np.asarray(base_weights, dtype=np.float64)
    )
    if base.shape != locations.shape or np.any(~np.isfinite(base)) or np.any(base <= 0):
        raise ValueError("base weights must be aligned, finite, and positive")

    resolved_edges = np.asarray(edges, dtype=np.float64)
    bins = _location_bins(locations, resolved_edges)
    counts = np.bincount(bins, minlength=len(resolved_edges) - 1)
    if np.any(counts == 0):
        raise ValueError("every predeclared physical-location bin must be represented")
    factors_by_bin = np.minimum(
        (counts.max() / counts.astype(np.float64)) ** float(power),
        float(max_factor),
    )
    factors = factors_by_bin[bins]
    result = base * factors
    for group in sorted(set(str(value) for value in groups)):
        mask = np.asarray([str(value) == group for value in groups], dtype=bool)
        before = base[mask].sum()
        after = result[mask].sum()
        if before <= 0 or after <= 0:
            raise ValueError(f"invalid sampling mass for group {group!r}")
        result[mask] *= before / after
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise AssertionError("location balancing produced invalid weights")
    return result


def location_sampling_audit(
    locations: Sequence[float],
    groups: Sequence[str],
    base_weights: Sequence[float] | None = None,
    **kwargs,
) -> dict[str, object]:
    locations_array = np.asarray(locations, dtype=np.float64)
    groups_array = np.asarray(groups, dtype=object)
    base = (
        np.ones(len(locations_array), dtype=np.float64)
        if base_weights is None
        else np.asarray(base_weights, dtype=np.float64)
    )
    weights = group_preserving_location_weights(
        locations_array, groups_array, base, **kwargs
    )
    edges = np.asarray(kwargs.get("edges", DEFAULT_EDGES), dtype=np.float64)
    bins = _location_bins(locations_array, edges)
    raw_mass = np.bincount(bins, weights=base, minlength=len(edges) - 1)
    balanced_mass = np.bincount(bins, weights=weights, minlength=len(edges) - 1)
    group_mass = {}
    for group in sorted(set(str(value) for value in groups_array)):
        mask = np.asarray([str(value) == group for value in groups_array], dtype=bool)
        group_mass[group] = {
            "before": float(base[mask].sum()),
            "after": float(weights[mask].sum()),
        }
    return {
        "case_count": len(locations_array),
        "edges": edges.tolist(),
        "power": float(kwargs.get("power", 0.5)),
        "max_factor": float(kwargs.get("max_factor", 5.0)),
        "raw_bin_fraction": (raw_mass / raw_mass.sum()).tolist(),
        "balanced_bin_fraction": (balanced_mass / balanced_mass.sum()).tolist(),
        "minimum_weight": float(weights.min()),
        "maximum_weight": float(weights.max()),
        "groups": group_mass,
    }
