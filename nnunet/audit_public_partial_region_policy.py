"""Compare component boxes with shape-following trusted supervision bands.

This public-only diagnostic simulates one observed primary lesion per complete
mask.  It compares the current expanded component bounding box against an
anisotropic L-infinity band around the actual lesion shape.  The latter is the
union of physical-axis-aligned neighborhoods around foreground voxels and can
be implemented efficiently with 3-D max pooling after augmentation.

No image, label, checkpoint, prediction, or submission is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nnunet.audit_public_partial_box_collision import (
    CONNECTIVITY_26,
    scenario_metrics_from_slice,
    sha256,
    summarize,
)


def expanded_slice(
    component_slice: tuple[slice, slice, slice],
    shape: Sequence[int],
    spacing_zyx_mm: Sequence[float],
    margin_mm: float,
) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError("spacing must contain three positive finite values")
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin must be finite and nonnegative")
    margins = np.ceil(float(margin_mm) / spacing).astype(np.int64)
    region = tuple(
        slice(
            max(0, int(axis.start) - int(margin)),
            min(int(size), int(axis.stop) + int(margin)),
        )
        for axis, size, margin in zip(component_slice, shape, margins, strict=True)
    )
    return region, margins


def shape_band_metrics_from_slice(
    labels: np.ndarray,
    component_id: int,
    component_slice: tuple[slice, slice, slice],
    component_size: int,
    total_foreground: int,
    spacing_zyx_mm: Sequence[float],
    margin_mm: float,
) -> dict[str, object]:
    """Measure a physical-axis L-infinity band around one lesion shape."""

    region, margins = expanded_slice(
        component_slice, labels.shape, spacing_zyx_mm, margin_mm
    )
    values = labels[region]
    primary = values == int(component_id)
    if int(primary.sum()) != int(component_size):
        raise ValueError("expanded region does not contain the complete component")
    kernel = tuple(int(2 * margin + 1) for margin in margins)
    trusted = ndimage.maximum_filter(
        primary, size=kernel, mode="constant", cval=0
    ).astype(bool, copy=False)
    omitted = (values > 0) & ~primary
    collided = omitted & trusted
    collided_ids = np.unique(values[collided])
    omitted_voxels = int(total_foreground - component_size)
    omitted_inside = int(collided.sum())
    true_background = int(((values == 0) & trusted).sum())
    return {
        "primary_voxels": int(component_size),
        "omitted_voxels": omitted_voxels,
        "omitted_voxels_inside_box": omitted_inside,
        "omitted_components_inside_box": int(len(collided_ids)),
        "omitted_component_ids_inside_box": [
            int(value) for value in collided_ids.tolist()
        ],
        "any_collision": bool(omitted_inside > 0),
        "omitted_voxel_collision_fraction": (
            float(omitted_inside / omitted_voxels) if omitted_voxels else 0.0
        ),
        "trusted_true_background_voxels": true_background,
        "trusted_background_to_primary_ratio": float(
            true_background / component_size
        ),
    }


def enrich_record(
    record: dict[str, object],
    *,
    case: str,
    component_id: int,
    component_volumes_mm3: np.ndarray,
) -> None:
    omitted_ids = [
        candidate
        for candidate in range(1, len(component_volumes_mm3) + 1)
        if candidate != component_id
    ]
    collided_ids = [int(value) for value in record["omitted_component_ids_inside_box"]]
    record.update(
        {
            "case": case,
            "primary_component": component_id,
            "primary_volume_mm3": float(component_volumes_mm3[component_id - 1]),
        }
    )
    for threshold in (1_000, 3_000):
        record[f"omitted_components_ge_{threshold}mm3"] = int(
            sum(component_volumes_mm3[value - 1] >= threshold for value in omitted_ids)
        )
        record[f"collided_components_ge_{threshold}mm3"] = int(
            sum(component_volumes_mm3[value - 1] >= threshold for value in collided_ids)
        )


def choose_policy(results: dict[str, object]) -> dict[str, object]:
    """Apply the documented post-descriptive-audit nomination rule."""

    band12 = results.get("shape_linf_band", {}).get("12", {}).get(
        "largest_component_as_primary"
    )
    box8 = results.get("component_box", {}).get("8", {}).get(
        "largest_component_as_primary"
    )
    box16 = results.get("component_box", {}).get("16", {}).get(
        "largest_component_as_primary"
    )
    if not (band12 and box8 and box16):
        return {
            "status": "NOT_EVALUATED",
            "reason": "12-mm band plus 8-mm and 16-mm boxes are required",
            "current_training_unchanged": True,
        }
    band12_large = band12["physical_component_bands"]["ge_3000mm3"]
    box8_large = box8["physical_component_bands"]["ge_3000mm3"]
    box16_large = box16["physical_component_bands"]["ge_3000mm3"]
    criteria = {
        "clean_background_not_less_than_8mm_box": bool(
            band12["trusted_background_to_primary_ratio"]["median"]
            >= box8["trusted_background_to_primary_ratio"]["median"]
        ),
        "omitted_voxel_collision_less_than_8mm_box": bool(
            band12["pooled_omitted_voxel_collision_fraction"]
            < box8["pooled_omitted_voxel_collision_fraction"]
        ),
        "large_collision_count_not_greater_than_8mm_box": bool(
            band12_large["collided_component_scenarios"]
            <= box8_large["collided_component_scenarios"]
        ),
        "omitted_voxel_collision_less_than_half_current_16mm_box": bool(
            band12["pooled_omitted_voxel_collision_fraction"]
            < 0.5 * box16["pooled_omitted_voxel_collision_fraction"]
        ),
        "large_collision_count_less_than_current_16mm_box": bool(
            band12_large["collided_component_scenarios"]
            < box16_large["collided_component_scenarios"]
        ),
    }
    passed = all(criteria.values())
    return {
        "status": "PASS" if passed else "REJECT",
        "recommended_future_policy": "shape_linf_band_12mm" if passed else None,
        "criteria": criteria,
        "decision_rule_timing": (
            "formalized after the descriptive geometry audit; this nominates a "
            "future trained experiment and does not prove performance improvement"
        ),
        "activation": (
            "prospective continuation only after a trained parent endpoint; "
            "no change to the active 16-mm run or frozen queue"
        ),
        "current_training_unchanged": True,
    }


def audit(labels_dir: Path, margins_mm: Sequence[float]) -> dict[str, object]:
    paths = sorted(labels_dir.glob("*.nii.gz"))
    if len(paths) != 50:
        raise ValueError(f"fixed-public labels must contain 50 cases, found {len(paths)}")
    margins = tuple(float(value) for value in margins_mm)
    if not margins or len(set(margins)) != len(margins):
        raise ValueError("margins must be nonempty and unique")
    policies = ("component_box", "shape_linf_band")
    records = {
        policy: {margin: {"all": [], "largest": []} for margin in margins}
        for policy in policies
    }
    inventory = []
    component_total = 0

    for path in paths:
        image = sitk.ReadImage(str(path))
        mask = sitk.GetArrayFromImage(image) > 0
        labels, count = ndimage.label(mask, structure=CONNECTIVITY_26)
        if count < 1:
            raise ValueError(f"empty public lesion mask: {path}")
        objects = ndimage.find_objects(labels, count)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        spacing_zyx = tuple(reversed(tuple(float(x) for x in image.GetSpacing())))
        volumes = sizes.astype(np.float64) * float(np.prod(spacing_zyx))
        foreground = int(mask.sum())
        largest = int(np.argmax(sizes))
        for margin in margins:
            per_policy = {policy: [] for policy in policies}
            for component_id, component_slice in enumerate(objects, start=1):
                if component_slice is None:
                    raise ValueError("connected-component slice is unexpectedly absent")
                box = scenario_metrics_from_slice(
                    labels,
                    component_id,
                    component_slice,
                    int(sizes[component_id - 1]),
                    foreground,
                    spacing_zyx,
                    margin,
                )
                band = shape_band_metrics_from_slice(
                    labels,
                    component_id,
                    component_slice,
                    int(sizes[component_id - 1]),
                    foreground,
                    spacing_zyx,
                    margin,
                )
                for policy, row in (("component_box", box), ("shape_linf_band", band)):
                    enrich_record(
                        row,
                        case=path.name,
                        component_id=component_id,
                        component_volumes_mm3=volumes,
                    )
                    per_policy[policy].append(row)
            for policy in policies:
                records[policy][margin]["all"].extend(per_policy[policy])
                records[policy][margin]["largest"].append(per_policy[policy][largest])
        component_total += count
        inventory.append(
            {
                "case": path.name,
                "components": count,
                "spacing_zyx_mm": list(spacing_zyx),
                "sha256": sha256(path),
            }
        )
        del labels, mask

    results = {}
    for policy in policies:
        results[policy] = {}
        for margin in margins:
            results[policy][f"{margin:g}"] = {
                "all_components_as_possible_primary": summarize(
                    records[policy][margin]["all"]
                ),
                "largest_component_as_primary": summarize(
                    records[policy][margin]["largest"]
                ),
            }
    decision = choose_policy(results)
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": (
            "Compare current component boxes with shape-following trusted bands "
            "under simulated primary-only lesion annotation"
        ),
        "dataset_role": "fixed-public diagnostic only",
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
        "connectivity": 26,
        "shape_band_geometry": (
            "anisotropic L-infinity physical-axis dilation; implementable by 3-D max pooling"
        ),
        "cases": len(paths),
        "lesion_components": component_total,
        "margins_mm": list(margins),
        "results": results,
        "decision": decision,
        "inventory": inventory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("validation/Validation-Public-Labels"),
    )
    parser.add_argument(
        "--margins-mm", nargs="+", type=float, default=(4.0, 8.0, 12.0, 16.0)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nnunet/analysis/public_partial_region_policy_20260827/summary.json"),
    )
    args = parser.parse_args()
    payload = audit(args.labels.resolve(), args.margins_mm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": payload["results"]}, indent=2))


if __name__ == "__main__":
    main()
