"""Audit hard-negative collisions induced by partial-label trusted boxes.

The FLARE Task-1 training masks may contain only one primary lesion.  This
diagnostic uses the complete fixed-public masks to simulate that contract: one
connected component is retained as the observed primary and every other
component is treated as omitted.  It then measures how often an axis-aligned
box around the observed component would accidentally trust voxels belonging to
an omitted lesion as background.

This is a diagnostic only.  It never writes labels, predictions, checkpoints,
or submission archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expanded_component_box(
    component: np.ndarray,
    spacing_zyx_mm: Sequence[float],
    margin_mm: float,
) -> tuple[slice, slice, slice]:
    """Return a clipped component bounding box expanded in physical units."""

    component = np.asarray(component, dtype=bool)
    if component.ndim != 3 or not component.any():
        raise ValueError("component must be a nonempty 3-D mask")
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx_mm must contain three positive finite values")
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin_mm must be finite and nonnegative")

    coordinates = np.where(component)
    margins = np.ceil(float(margin_mm) / spacing).astype(np.int64)
    lower = np.asarray([int(axis.min()) for axis in coordinates], dtype=np.int64)
    upper = np.asarray([int(axis.max()) + 1 for axis in coordinates], dtype=np.int64)
    lower = np.maximum(0, lower - margins)
    upper = np.minimum(np.asarray(component.shape, dtype=np.int64), upper + margins)
    return tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper, strict=True))


def scenario_metrics(
    labels: np.ndarray,
    component_id: int,
    spacing_zyx_mm: Sequence[float],
    margin_mm: float,
) -> dict[str, float | int | bool]:
    """Measure one hypothetical observed-primary scenario."""

    labels = np.asarray(labels)
    primary = labels == int(component_id)
    if not primary.any():
        raise ValueError(f"component {component_id} is absent")
    omitted = (labels > 0) & ~primary
    box = expanded_component_box(primary, spacing_zyx_mm, margin_mm)
    primary_voxels = int(primary.sum())
    omitted_voxels = int(omitted.sum())
    omitted_inside = int(omitted[box].sum())
    true_background_inside = int((labels[box] == 0).sum())
    omitted_ids_inside = np.unique(labels[box][omitted[box]])
    omitted_ids_inside = omitted_ids_inside[omitted_ids_inside > 0]
    return {
        "primary_voxels": primary_voxels,
        "omitted_voxels": omitted_voxels,
        "omitted_voxels_inside_box": omitted_inside,
        "omitted_components_inside_box": int(len(omitted_ids_inside)),
        "omitted_component_ids_inside_box": [
            int(value) for value in omitted_ids_inside.tolist()
        ],
        "any_collision": bool(omitted_inside > 0),
        "omitted_voxel_collision_fraction": (
            float(omitted_inside / omitted_voxels) if omitted_voxels else 0.0
        ),
        "trusted_true_background_voxels": true_background_inside,
        "trusted_background_to_primary_ratio": float(
            true_background_inside / primary_voxels
        ),
    }


def scenario_metrics_from_slice(
    labels: np.ndarray,
    component_id: int,
    component_slice: tuple[slice, slice, slice],
    component_size: int,
    total_foreground: int,
    spacing_zyx_mm: Sequence[float],
    margin_mm: float,
) -> dict[str, float | int | bool]:
    """Memory-efficient equivalent of :func:`scenario_metrics` for volumes."""

    spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx_mm must contain three positive finite values")
    if component_size <= 0 or total_foreground < component_size:
        raise ValueError("component and foreground sizes are inconsistent")
    margins = np.ceil(float(margin_mm) / spacing).astype(np.int64)
    box = tuple(
        slice(
            max(0, int(axis.start) - int(margin)),
            min(int(size), int(axis.stop) + int(margin)),
        )
        for axis, margin, size in zip(
            component_slice, margins, labels.shape, strict=True
        )
    )
    values = labels[box]
    foreground_inside = values > 0
    primary_inside = values == int(component_id)
    omitted_inside_mask = foreground_inside & ~primary_inside
    omitted_inside = int(omitted_inside_mask.sum())
    omitted_voxels = int(total_foreground - component_size)
    omitted_ids_inside = np.unique(values[omitted_inside_mask])
    return {
        "primary_voxels": int(component_size),
        "omitted_voxels": omitted_voxels,
        "omitted_voxels_inside_box": omitted_inside,
        "omitted_components_inside_box": int(len(omitted_ids_inside)),
        "omitted_component_ids_inside_box": [
            int(value) for value in omitted_ids_inside.tolist()
        ],
        "any_collision": bool(omitted_inside > 0),
        "omitted_voxel_collision_fraction": (
            float(omitted_inside / omitted_voxels) if omitted_voxels else 0.0
        ),
        "trusted_true_background_voxels": int((values == 0).sum()),
        "trusted_background_to_primary_ratio": float(
            (values == 0).sum() / component_size
        ),
    }


def summarize(records: Sequence[dict[str, float | int | bool]]) -> dict[str, object]:
    if not records:
        raise ValueError("cannot summarize an empty scenario set")
    collisions = np.asarray([bool(row["any_collision"]) for row in records])
    omitted = np.asarray([int(row["omitted_voxels"]) for row in records], dtype=np.int64)
    omitted_inside = np.asarray(
        [int(row["omitted_voxels_inside_box"]) for row in records], dtype=np.int64
    )
    ratios = np.asarray(
        [float(row["trusted_background_to_primary_ratio"]) for row in records],
        dtype=np.float64,
    )
    eligible = omitted > 0
    component_bands = {}
    for threshold in (1_000, 3_000):
        total = sum(
            int(row.get(f"omitted_components_ge_{threshold}mm3", 0))
            for row in records
        )
        collided = sum(
            int(row.get(f"collided_components_ge_{threshold}mm3", 0))
            for row in records
        )
        component_bands[f"ge_{threshold}mm3"] = {
            "omitted_component_scenarios": total,
            "collided_component_scenarios": collided,
            "collision_fraction": float(collided / total) if total else 0.0,
        }
    return {
        "scenarios": len(records),
        "scenarios_with_omitted_lesions": int(eligible.sum()),
        "scenarios_with_collision": int(collisions.sum()),
        "collision_rate_all_scenarios": float(collisions.mean()),
        "collision_rate_when_omitted_lesions_exist": (
            float(collisions[eligible].mean()) if eligible.any() else 0.0
        ),
        "pooled_omitted_voxel_collision_fraction": (
            float(omitted_inside.sum() / omitted.sum()) if omitted.sum() else 0.0
        ),
        "trusted_background_to_primary_ratio": {
            "median": float(np.median(ratios)),
            "p10": float(np.percentile(ratios, 10)),
            "p90": float(np.percentile(ratios, 90)),
        },
        "physical_component_bands": component_bands,
    }


def audit(labels_dir: Path, margins_mm: Iterable[float]) -> dict[str, object]:
    paths = sorted(labels_dir.glob("*.nii.gz"))
    if len(paths) != 50:
        raise ValueError(f"fixed-public label inventory must be 50, found {len(paths)}")
    margins = tuple(float(value) for value in margins_mm)
    if not margins or len(set(margins)) != len(margins):
        raise ValueError("margins must be nonempty and unique")

    records_by_margin = {
        margin: {"all": [], "largest": []}
        for margin in margins
    }
    inventory = []
    total_components = 0
    for path in paths:
        image = sitk.ReadImage(str(path))
        mask = sitk.GetArrayFromImage(image) > 0
        labels, count = ndimage.label(mask, structure=CONNECTIVITY_26)
        if count < 1:
            raise ValueError(f"public lesion mask is empty: {path}")
        sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        objects = ndimage.find_objects(labels, count)
        spacing_zyx = tuple(reversed(tuple(float(x) for x in image.GetSpacing())))
        voxel_volume_mm3 = float(np.prod(spacing_zyx))
        component_volumes_mm3 = sizes.astype(np.float64) * voxel_volume_mm3
        total_foreground_case = int(mask.sum())
        largest_index = int(np.argmax(sizes))
        for margin in margins:
            records = [
                scenario_metrics_from_slice(
                    labels,
                    component_id,
                    component_slice,
                    int(sizes[component_id - 1]),
                    total_foreground_case,
                    spacing_zyx,
                    margin,
                )
                for component_id, component_slice in enumerate(objects, start=1)
                if component_slice is not None
            ]
            for component_id, record in enumerate(records, start=1):
                omitted_ids = [
                    candidate
                    for candidate in range(1, count + 1)
                    if candidate != component_id
                ]
                collided_ids = [
                    int(value)
                    for value in record["omitted_component_ids_inside_box"]
                ]
                record.update(
                    {
                        "case": path.name,
                        "primary_component": component_id,
                        "primary_volume_mm3": float(
                            component_volumes_mm3[component_id - 1]
                        ),
                        "omitted_components_ge_1000mm3": int(
                            sum(
                                component_volumes_mm3[value - 1] >= 1_000
                                for value in omitted_ids
                            )
                        ),
                        "collided_components_ge_1000mm3": int(
                            sum(
                                component_volumes_mm3[value - 1] >= 1_000
                                for value in collided_ids
                            )
                        ),
                        "omitted_components_ge_3000mm3": int(
                            sum(
                                component_volumes_mm3[value - 1] >= 3_000
                                for value in omitted_ids
                            )
                        ),
                        "collided_components_ge_3000mm3": int(
                            sum(
                                component_volumes_mm3[value - 1] >= 3_000
                                for value in collided_ids
                            )
                        ),
                    }
                )
            records_by_margin[margin]["all"].extend(records)
            records_by_margin[margin]["largest"].append(records[largest_index])
        total_components += count
        inventory.append(
            {
                "case": path.name,
                "components": count,
                "foreground_voxels": total_foreground_case,
                "spacing_zyx_mm": list(spacing_zyx),
                "sha256": sha256(path),
            }
        )
        del labels, mask

    by_margin: dict[str, object] = {}
    for margin in margins:
        by_margin[f"{margin:g}"] = {
            "all_components_as_possible_primary": summarize(
                records_by_margin[margin]["all"]
            ),
            "largest_component_as_primary": summarize(
                records_by_margin[margin]["largest"]
            ),
        }

    decision: dict[str, object]
    if "8" in by_margin and "16" in by_margin:
        eight = by_margin["8"]["largest_component_as_primary"]
        sixteen = by_margin["16"]["largest_component_as_primary"]
        eight_background = eight["trusted_background_to_primary_ratio"]["median"]
        eight_collision = eight["pooled_omitted_voxel_collision_fraction"]
        sixteen_collision = sixteen["pooled_omitted_voxel_collision_fraction"]
        eight_large = eight["physical_component_bands"]["ge_3000mm3"]
        sixteen_large = sixteen["physical_component_bands"]["ge_3000mm3"]
        passes = bool(
            eight_background >= 5.0
            and eight_collision <= 0.5 * sixteen_collision
            and eight_large["collided_component_scenarios"]
            < sixteen_large["collided_component_scenarios"]
        )
        decision = {
            "status": "PASS" if passes else "REJECT",
            "recommended_future_margin_mm": 8.0 if passes else None,
            "current_16mm_training_run_unchanged": True,
            "activation": (
                "single predeclared partial-supervision continuation only after "
                "the active base endpoint is evaluated; not an automatic queue change"
            ),
            "decision_rule_timing": (
                "formalized after the descriptive collision audit; this nominates a "
                "future candidate and is not evidence of model-performance improvement"
            ),
            "criteria": {
                "median_clean_background_to_primary_at_8mm_ge_5": bool(
                    eight_background >= 5.0
                ),
                "omitted_voxel_collision_at_8mm_le_half_16mm": bool(
                    eight_collision <= 0.5 * sixteen_collision
                ),
                "fewer_collided_omitted_components_ge_3000mm3_at_8mm": bool(
                    eight_large["collided_component_scenarios"]
                    < sixteen_large["collided_component_scenarios"]
                ),
            },
            "limitation": (
                "The public complete masks simulate, but cannot identify, which lesion "
                "the challenge would designate as primary in each training case."
            ),
        }
    else:
        decision = {
            "status": "NOT_EVALUATED",
            "reason": "both 8-mm and 16-mm margins are required for the decision",
        }

    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": (
            "Estimate whether trusted component boxes convert nearby omitted lesions "
            "into hard-negative supervision under the Train-Labeled primary-only contract"
        ),
        "dataset_role": "fixed-public diagnostic only",
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
        "connectivity": 26,
        "cases": len(paths),
        "lesion_components": total_components,
        "margins_mm": list(margins),
        "results": by_margin,
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
        "--margins-mm",
        nargs="+",
        type=float,
        default=(4.0, 8.0, 12.0, 16.0, 24.0, 32.0),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nnunet/analysis/public_partial_box_collision_20260827/summary.json"),
    )
    args = parser.parse_args()
    payload = audit(args.labels.resolve(), args.margins_mm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": payload["results"]}, indent=2))


if __name__ == "__main__":
    main()
