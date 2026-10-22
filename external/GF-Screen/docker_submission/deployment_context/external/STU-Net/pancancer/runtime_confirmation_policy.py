#!/usr/bin/env python3
"""Label-blind early exit for the GF/FreeTumor/STU confirmation ensemble."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def confirmation_state(anchor: np.ndarray, freetumor: np.ndarray) -> dict:
    """Return whether STU-Net can change the fixed ``either`` deployment.

    The deployed rule keeps a whole GF anchor component when either FreeTumor
    or STU-Net overlaps it. If FreeTumor already overlaps every anchor
    component, STU-Net cannot change the final mask and may be skipped exactly.
    No labels or image-derived heuristic are used.
    """

    anchor = np.asarray(anchor, dtype=bool)
    freetumor = np.asarray(freetumor, dtype=bool)
    if anchor.shape != freetumor.shape:
        raise ValueError("anchor and FreeTumor masks must have identical shapes")
    labels, component_count = ndimage.label(anchor, structure=CONNECTIVITY_26)
    confirmed = np.zeros(component_count + 1, dtype=bool)
    if component_count:
        confirmed[np.unique(labels[anchor & freetumor])] = True
    confirmed[0] = False
    confirmed_count = int(confirmed.sum())
    return {
        "anchor_component_count": int(component_count),
        "freetumor_confirmed_component_count": confirmed_count,
        "requires_stunet": confirmed_count != int(component_count),
    }


def exact_early_exit_mask(
    anchor: np.ndarray, freetumor: np.ndarray
) -> np.ndarray | None:
    """Return the exact final mask when STU-Net is provably irrelevant."""

    state = confirmation_state(anchor, freetumor)
    if state["requires_stunet"]:
        return None
    return np.asarray(anchor, dtype=bool).copy()

