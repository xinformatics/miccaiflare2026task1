#!/usr/bin/env python3
"""Measure retained pseudo-lesion sizes in a live nested-teacher snapshot.

The audit is label-blind. It applies the same component and high-confidence
seed criteria as ``build_unlabeled_hysteresis_pseudolabels.py`` without writing
pseudo masks, allowing the small-lesion exposure policy to be checked while
teacher inference is still running.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np

from inference_naming import prediction_name
from utils.data_utils import component_candidate_counts


VOLUME_BINS = (
    ("100_to_500", 100.0, 500.0),
    ("500_to_1000", 500.0, 1000.0),
    ("1000_to_3000", 1000.0, 3000.0),
    ("ge_3000", 3000.0, float("inf")),
)


def evenly_spaced(names: list[str], maximum: int) -> list[str]:
    if maximum < 1:
        raise ValueError("max-cases must be positive")
    if len(names) <= maximum:
        return names
    indices = np.linspace(0, len(names) - 1, num=maximum, dtype=int)
    return [names[int(index)] for index in np.unique(indices)]


def summarize_recipe(
    support: np.ndarray,
    seed: np.ndarray,
    voxel_mm3: float,
    min_seed_mm3: float,
    min_component_mm3: float,
) -> dict[str, object]:
    support = np.asarray(support, dtype=bool)
    seed = np.asarray(seed, dtype=bool)
    if support.shape != seed.shape:
        raise ValueError("support/seed shape mismatch")
    if np.any(seed & ~support):
        raise ValueError("seed is not nested inside support")
    locations = np.nonzero(support)
    if not locations[0].size:
        return {"retained_components": 0, "retained_volume_mm3": 0.0,
                "component_volumes_mm3": []}
    bounds = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in locations
    )
    support = support[bounds]
    seed = seed[bounds]
    components, count = cc3d.connected_components(
        support, connectivity=26, return_N=True
    )
    identifiers = np.arange(1, count + 1)
    component_mm3 = (
        np.bincount(components.reshape(-1), minlength=count + 1)[1:].astype(float)
        * voxel_mm3
    )
    seed_mm3 = (
        np.bincount(components[seed], minlength=count + 1)[1:].astype(float)
        * voxel_mm3
    )
    retained = component_mm3[
        (component_mm3 >= min_component_mm3) & (seed_mm3 >= min_seed_mm3)
    ]
    return {
        "retained_components": int(retained.size),
        "retained_volume_mm3": float(retained.sum()),
        "component_volumes_mm3": [float(value) for value in retained],
    }


def summarize_candidate_mass(
    component_volumes_mm3: list[float],
    points_per_component: int,
    size_power: float,
) -> dict[str, object]:
    bins = {name: 0 for name, _, _ in VOLUME_BINS}
    if component_volumes_mm3:
        counts = component_candidate_counts(
            component_volumes_mm3,
            points_per_component=points_per_component,
            size_power=size_power,
        )
        for volume, count in zip(component_volumes_mm3, counts):
            for name, lower, upper in VOLUME_BINS:
                if lower <= volume < upper:
                    bins[name] += int(count)
                    break
    total = sum(bins.values())
    return {
        "candidate_centers": total,
        "candidate_center_volume_bins": bins,
        "small_candidate_fraction_lt_500mm3": (
            bins["100_to_500"] / total if total else 0.0
        ),
    }


def discover(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*.nii.gz")):
        if not path.is_file():
            continue
        name = prediction_name(path.name)
        if name in result:
            raise ValueError(f"duplicate prediction name: {name}")
        result[name] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--exclude-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument("--component-crop-points", type=int, default=64)
    parser.add_argument("--component-crop-size-power", type=float, default=0.5)
    args = parser.parse_args()
    if args.component_crop_points < 1:
        parser.error("component-crop-points must be positive")
    if not 0.0 <= args.component_crop_size_power <= 1.0:
        parser.error("component-crop-size-power must be in [0,1]")

    recipes = {
        "balanced": {
            "support_threshold": "0.85", "seed_threshold": "0.95",
            "min_seed_mm3": 25.0, "min_component_mm3": 100.0,
        },
        "conservative": {
            "support_threshold": "0.90", "seed_threshold": "0.97",
            "min_seed_mm3": 50.0, "min_component_mm3": 100.0,
        },
    }
    images = discover(args.images)
    excluded: set[str] = set()
    if args.exclude_audit:
        payload = json.loads(args.exclude_audit.read_text(encoding="utf-8"))
        excluded = {
            prediction_name(Path(path).name)
            for path in payload.get("excluded_unlabeled_files", [])
        }
    threshold_paths = {
        threshold: {
            path.name: path
            for path in (args.teacher_root / f"lesion_{threshold}").glob("*.nii.gz")
            if path.is_file()
        }
        for threshold in ("0.85", "0.90", "0.95", "0.97")
    }
    common = sorted(
        (set(images) - excluded).intersection(
            *(set(paths) for paths in threshold_paths.values())
        )
    )
    if not common:
        raise SystemExit("no completed non-excluded teacher cases")
    selected = evenly_spaced(common, args.max_cases)

    summaries: dict[str, dict[str, object]] = {}
    for recipe, configuration in recipes.items():
        bins = {name: 0 for name, _, _ in VOLUME_BINS}
        positive_cases = 0
        components = 0
        volume_mm3 = 0.0
        candidate_bins = {name: 0 for name, _, _ in VOLUME_BINS}
        per_case = []
        for name in selected:
            source = nib.load(str(images[name]))
            support_image = nib.load(str(threshold_paths[configuration["support_threshold"]][name]))
            seed_image = nib.load(str(threshold_paths[configuration["seed_threshold"]][name]))
            if support_image.shape != source.shape or seed_image.shape != source.shape:
                raise ValueError(f"{name}: source/teacher shape mismatch")
            if not np.allclose(support_image.affine, source.affine, atol=1e-4):
                raise ValueError(f"{name}: support/source affine mismatch")
            if not np.allclose(seed_image.affine, source.affine, atol=1e-4):
                raise ValueError(f"{name}: seed/source affine mismatch")
            stats = summarize_recipe(
                np.asanyarray(support_image.dataobj) > 0,
                np.asanyarray(seed_image.dataobj) > 0,
                float(abs(np.linalg.det(source.affine[:3, :3]))),
                float(configuration["min_seed_mm3"]),
                float(configuration["min_component_mm3"]),
            )
            positive_cases += int(stats["retained_components"] > 0)
            components += int(stats["retained_components"])
            volume_mm3 += float(stats["retained_volume_mm3"])
            for value in stats["component_volumes_mm3"]:
                for bin_name, lower, upper in VOLUME_BINS:
                    if lower <= value < upper:
                        bins[bin_name] += 1
                        break
            candidate_summary = summarize_candidate_mass(
                stats["component_volumes_mm3"],
                args.component_crop_points,
                args.component_crop_size_power,
            )
            for bin_name, count in candidate_summary[
                "candidate_center_volume_bins"
            ].items():
                candidate_bins[bin_name] += int(count)
            per_case.append({
                "name": name,
                "retained_components": stats["retained_components"],
                "retained_volume_mm3": stats["retained_volume_mm3"],
                "component_volumes_mm3": stats["component_volumes_mm3"],
                **candidate_summary,
            })
        candidate_centers = sum(candidate_bins.values())
        summaries[recipe] = {
            **configuration,
            "positive_cases": positive_cases,
            "positive_case_fraction": positive_cases / len(selected),
            "retained_components": components,
            "retained_volume_mm3": volume_mm3,
            "component_volume_bins": bins,
            "small_component_fraction_lt_500mm3": (
                bins["100_to_500"] / components if components else 0.0
            ),
            "component_crop_points_per_lesion": args.component_crop_points,
            "component_crop_size_power": args.component_crop_size_power,
            "candidate_centers": candidate_centers,
            "candidate_center_volume_bins": candidate_bins,
            "small_candidate_fraction_lt_500mm3": (
                candidate_bins["100_to_500"] / candidate_centers
                if candidate_centers else 0.0
            ),
            "cases": per_case,
        }

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "PASS",
        "label_blind": True,
        "image_inventory": len(images),
        "excluded_images": len(excluded),
        "completed_common_inventory": len(common),
        "audited_cases": len(selected),
        "selection": "evenly_spaced_over_sorted_completed_nonexcluded_inventory",
        "recipes": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": payload["status"],
        "completed_common_inventory": len(common),
        "audited_cases": len(selected),
        "recipes": {
            name: {key: value for key, value in summary.items() if key != "cases"}
            for name, summary in summaries.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
