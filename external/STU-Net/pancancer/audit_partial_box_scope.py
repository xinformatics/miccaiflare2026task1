#!/usr/bin/env python3
"""Measure excess trusted background from one box around multiple lesions.

The audit is training-only and deterministic. It samples cases at evenly spaced
indices from the sorted preprocessed set, then extracts two foreground-centred
patches per case using the stored nnU-Net class locations. No validation images
or labels are read.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
from pathlib import Path

import blosc2
import numpy as np
from scipy import ndimage


PATCH_SHAPE = np.asarray((128, 160, 160), dtype=np.int64)
MARGIN_VOXELS = np.asarray((7, 10, 10), dtype=np.int64)
QUANTILES = (0.25, 0.75)


def box_scope(mask: np.ndarray) -> dict[str, int | float]:
    """Return exact global/component-wise trusted-box volumes for one patch."""

    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError("mask must be a three-dimensional boolean array")
    coordinates = np.where(mask)
    if coordinates[0].size == 0:
        return {
            "components": 0,
            "foreground_voxels": 0,
            "global_box_voxels": 0,
            "component_box_union_voxels": 0,
            "global_over_component_union": 0.0,
        }

    lower = np.asarray([axis.min() for axis in coordinates], dtype=np.int64)
    upper = np.asarray([axis.max() + 1 for axis in coordinates], dtype=np.int64)
    foreground_crop = mask[
        tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    ]
    labeled, component_count = ndimage.label(
        foreground_crop, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )

    global_lower = np.maximum(0, lower - MARGIN_VOXELS)
    global_upper = np.minimum(np.asarray(mask.shape), upper + MARGIN_VOXELS)
    global_volume = int(np.prod(global_upper - global_lower))
    component_union = np.zeros(mask.shape, dtype=np.bool_)
    for component_slice in ndimage.find_objects(labeled, component_count):
        if component_slice is None:
            continue
        component_lower = lower + np.asarray(
            [axis.start for axis in component_slice], dtype=np.int64
        )
        component_upper = lower + np.asarray(
            [axis.stop for axis in component_slice], dtype=np.int64
        )
        expanded_lower = np.maximum(0, component_lower - MARGIN_VOXELS)
        expanded_upper = np.minimum(
            np.asarray(mask.shape), component_upper + MARGIN_VOXELS
        )
        component_union[
            tuple(
                slice(int(start), int(stop))
                for start, stop in zip(expanded_lower, expanded_upper)
            )
        ] = True
    component_volume = int(component_union.sum())
    return {
        "components": int(component_count),
        "foreground_voxels": int(mask.sum()),
        "global_box_voxels": global_volume,
        "component_box_union_voxels": component_volume,
        "global_over_component_union": global_volume / max(1, component_volume),
    }


def audit(preprocessed: Path, case_count: int) -> dict:
    segmentations = sorted(preprocessed.glob("*_seg.b2nd"))
    if len(segmentations) < case_count:
        raise ValueError(
            f"requested {case_count} cases but found only {len(segmentations)}"
        )
    indices = np.linspace(0, len(segmentations) - 1, case_count, dtype=np.int64)
    rows = []
    for segmentation_index in indices:
        segmentation_path = segmentations[int(segmentation_index)]
        case = segmentation_path.name.removesuffix("_seg.b2nd")
        properties_path = preprocessed / f"{case}.pkl"
        with properties_path.open("rb") as handle:
            properties = pickle.load(handle)
        locations = np.asarray(properties["class_locations"][1])
        if locations.ndim != 2 or locations.shape[1] != 4 or len(locations) == 0:
            raise ValueError(f"invalid foreground class locations: {properties_path}")
        compressed = blosc2.open(str(segmentation_path))
        spatial_shape = np.asarray(compressed.shape[1:], dtype=np.int64)
        for quantile in QUANTILES:
            location_index = min(len(locations) - 1, int(quantile * len(locations)))
            center = locations[location_index, 1:].astype(np.int64)
            patch_lower = np.minimum(
                np.maximum(0, center - PATCH_SHAPE // 2),
                np.maximum(0, spatial_shape - PATCH_SHAPE),
            )
            patch_upper = np.minimum(spatial_shape, patch_lower + PATCH_SHAPE)
            patch_slice = (slice(0, 1),) + tuple(
                slice(int(start), int(stop))
                for start, stop in zip(patch_lower, patch_upper)
            )
            mask = compressed[patch_slice][0] == 1
            row = box_scope(mask)
            row.update(
                {
                    "case": case,
                    "foreground_location_quantile": quantile,
                    "patch_lower_zyx": patch_lower.tolist(),
                    "patch_shape_zyx": list(mask.shape),
                }
            )
            rows.append(row)

    components = np.asarray([row["components"] for row in rows], dtype=np.int64)
    ratios = np.asarray(
        [row["global_over_component_union"] for row in rows], dtype=np.float64
    )
    multi = components > 1
    multi_ratios = ratios[multi]
    summary = {
        "population_cases": len(segmentations),
        "sampled_cases": case_count,
        "sampled_patches": len(rows),
        "multi_component_patches": int(multi.sum()),
        "multi_component_fraction": float(multi.mean()),
        "component_count_median": float(np.median(components)),
        "component_count_p90": float(np.quantile(components, 0.9)),
        "component_count_max": int(components.max()),
        "multi_global_over_component_union_median": float(
            np.median(multi_ratios)
        ),
        "multi_global_over_component_union_p90": float(
            np.quantile(multi_ratios, 0.9)
        ),
        "multi_global_over_component_union_max": float(multi_ratios.max()),
        "multi_ratio_gt_2": int((multi & (ratios > 2.0)).sum()),
        "multi_ratio_gt_5": int((multi & (ratios > 5.0)).sum()),
    }
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "dataset_role": "Train-Labeled partial masks only",
        "validation_read": False,
        "selection_use": False,
        "method": {
            "case_sampling": "evenly spaced indices in sorted segmentation paths",
            "foreground_location_quantiles": list(QUANTILES),
            "patch_shape_zyx": PATCH_SHAPE.tolist(),
            "spacing_zyx_mm": [2.4, 1.6, 1.6],
            "context_margin_zyx_voxels": MARGIN_VOXELS.tolist(),
            "component_connectivity": 26,
        },
        "summary": summary,
        "patches": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cases < 1:
        raise ValueError("--cases must be positive")
    payload = audit(args.preprocessed, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
