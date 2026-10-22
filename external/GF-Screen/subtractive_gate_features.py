#!/usr/bin/env python3
"""Inference-visible component features for the qualified subtractive gate."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


FEATURES = (
    "log_mm3",
    "hu_mean",
    "hu_std",
    "hu_p10",
    "hu_p50",
    "hu_p90",
    "shell_hu_p50",
    "contrast",
    "ext_x",
    "ext_y",
    "ext_z",
    "elongation",
    "fill_frac",
    "surface_ratio",
    "z_centroid_norm",
    "case_n_comps",
    "case_pred_mm3",
    "vol_rank",
    "vol_frac_of_case",
)


def extract_component_features(
    prediction: np.ndarray,
    ct: np.ndarray,
    affine: np.ndarray,
    minimum_component_mm3: float = 1000.0,
) -> tuple[list[dict[str, float | int]], np.ndarray]:
    """Reproduce the audited 707-case builder's inference features exactly."""
    prediction = np.asarray(prediction, dtype=bool)
    ct = np.asarray(ct)
    if prediction.ndim != 3 or ct.shape != prediction.shape:
        raise ValueError("prediction and CT must be same-shape 3D arrays")
    if not np.isfinite(affine).all() or minimum_component_mm3 <= 0:
        raise ValueError("invalid affine or component threshold")
    voxel_mm3 = float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))
    if not math.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError("invalid voxel volume")

    component_map, count = ndimage.label(
        prediction, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    if count == 0:
        return [], component_map
    sizes = np.asarray(
        ndimage.sum(
            prediction, component_map, index=np.arange(1, count + 1)
        ),
        dtype=np.float64,
    )
    keep = [
        value
        for value in range(1, count + 1)
        if sizes[value - 1] * voxel_mm3 >= minimum_component_mm3
    ]
    if not keep:
        return [], component_map
    deployed = np.isin(component_map, keep)
    predicted_voxels = int(deployed.sum())
    objects = ndimage.find_objects(component_map)
    z_dimension = prediction.shape[2]
    ordered_volumes = sorted((sizes[value - 1] for value in keep), reverse=True)

    rows: list[dict[str, float | int]] = []
    for value in sorted(keep):
        component = component_map == value
        voxels = int(sizes[value - 1])
        region = objects[value - 1]
        if region is None:
            raise RuntimeError(f"missing component region: {value}")
        extents = [axis.stop - axis.start for axis in region]
        pad = 6
        expanded = tuple(
            slice(max(0, axis.start - pad), min(dimension, axis.stop + pad))
            for axis, dimension in zip(region, prediction.shape)
        )
        local_component = component[expanded]
        local_ct = np.asarray(ct[expanded]).astype(np.float32)
        shell = ndimage.binary_dilation(local_component, iterations=3) & ~local_component
        inside_values = local_ct[local_component]
        shell_values = local_ct[shell] if shell.any() else np.asarray([np.nan], np.float32)
        shell_median = float(np.nanmedian(shell_values))
        centroid_z = float(ndimage.center_of_mass(component)[2]) / max(1, z_dimension - 1)
        rows.append(
            {
                "comp": value,
                "mm3": voxels * voxel_mm3,
                "log_mm3": float(np.log10(voxels * voxel_mm3)),
                "hu_mean": float(inside_values.mean()),
                "hu_std": float(inside_values.std()),
                "hu_p10": float(np.percentile(inside_values, 10)),
                "hu_p50": float(np.median(inside_values)),
                "hu_p90": float(np.percentile(inside_values, 90)),
                "shell_hu_p50": shell_median,
                "contrast": float(np.median(inside_values) - shell_median),
                "ext_x": extents[0],
                "ext_y": extents[1],
                "ext_z": extents[2],
                "elongation": float(max(extents) / max(1, min(extents))),
                "fill_frac": float(
                    voxels / max(1, extents[0] * extents[1] * extents[2])
                ),
                "surface_ratio": float(
                    ndimage.binary_dilation(local_component).sum()
                    - local_component.sum()
                )
                / max(1, voxels),
                "z_centroid_norm": centroid_z,
                "case_n_comps": len(keep),
                "case_pred_mm3": predicted_voxels * voxel_mm3,
                "vol_rank": ordered_volumes.index(sizes[value - 1]) + 1,
                "vol_frac_of_case": float(voxels / max(1, predicted_voxels)),
            }
        )
    return rows, component_map


def feature_matrix(rows: list[dict[str, float | int]]) -> np.ndarray:
    matrix = np.asarray(
        [[float(row[name]) for name in FEATURES] for row in rows], dtype=np.float64
    )
    if matrix.shape != (len(rows), len(FEATURES)):
        raise ValueError("invalid subtractive-gate feature matrix")
    return np.nan_to_num(matrix, nan=-2048.0, posinf=4096.0, neginf=-4096.0)
