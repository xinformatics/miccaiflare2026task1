#!/usr/bin/env python3
"""Label-informed upper bound for multi-tile candidate bounding-box refinement.

For every proposal component touching a target component, this diagnostic
reconstructs target voxels inside the proposal bounding box plus a physical
margin. It also estimates the number of fixed 96-mm tiles needed to cover the
same region. Results are analysis-only and must never drive hidden masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean

import cc3d
import nibabel as nib
import numpy as np

from proposal_component_instances import source_separated_instances


DEFAULT_MARGINS_MM = (0.0, 24.0, 48.0, 72.0, 96.0)


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.count_nonzero(prediction & target))
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def distribution(values: list[float]) -> dict[str, float] | None:
    """Compact deterministic geometry distribution for positive components."""

    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("component geometry contains non-finite values")
    return {
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(array.max()),
    }


def analyze_case(
    task: tuple,
) -> dict[str, object]:
    if len(task) == 7:
        (
            name,
            target_path,
            proposal_paths,
            minimum_mm3,
            maximum_mm3,
            tile_stride_mm,
            margins_mm,
        ) = task
        component_policy = "union"
    elif len(task) == 8:
        (
            name,
            target_path,
            proposal_paths,
            minimum_mm3,
            maximum_mm3,
            tile_stride_mm,
            margins_mm,
            component_policy,
        ) = task
    else:
        raise ValueError(f"invalid bbox-oracle task length: {len(task)}")
    if component_policy not in ("union", "source_separate"):
        raise ValueError(f"invalid component policy: {component_policy}")
    target_nii = nib.load(target_path)
    target = np.asanyarray(target_nii.dataobj).astype(bool)
    if isinstance(proposal_paths, str):
        proposal_paths = (proposal_paths,)
    proposal_masks = []
    for proposal_path in proposal_paths:
        proposal_nii = nib.load(proposal_path)
        if target_nii.shape[:3] != proposal_nii.shape[:3] or not np.allclose(
            target_nii.affine, proposal_nii.affine, atol=1e-3, rtol=0
        ):
            raise ValueError(f"geometry mismatch: {name} {proposal_path}")
        proposal_masks.append(np.asanyarray(proposal_nii.dataobj).astype(bool))
    target_components = cc3d.connected_components(target.astype(np.uint8), connectivity=26)
    if component_policy == "union":
        proposal = np.ascontiguousarray(np.logical_or.reduce(proposal_masks))
        proposal_components, count = cc3d.connected_components(
            proposal.astype(np.uint8), connectivity=26, return_N=True
        )
        statistics = cc3d.statistics(proposal_components)
        component_maps = None
        instances = [
            {
                "source_index": -1,
                "source_component_id": component_id,
                "bbox": statistics["bounding_boxes"][component_id],
                "voxels": int(statistics["voxel_counts"][component_id]),
            }
            for component_id in range(1, count + 1)
        ]
    else:
        component_maps, instances = source_separated_instances(proposal_masks)
        proposal_components = None
    spacing = np.linalg.norm(target_nii.affine[:3, :3], axis=0)
    voxel_mm3 = float(np.prod(spacing))
    predictions = {margin: np.zeros_like(target) for margin in margins_mm}
    tile_counts = {margin: 0 for margin in margins_mm}
    positive_candidates = 0
    positive_component_mm3: list[float] = []
    positive_component_seed_overlap: list[float] = []
    for instance in instances:
        component_voxels = int(instance["voxels"])
        component_mm3 = component_voxels * voxel_mm3
        if not minimum_mm3 <= component_mm3 <= maximum_mm3:
            continue
        # cc3d already computed exact component bounds. Restricting both the
        # component mask and target lookup to that box is array-equivalent to
        # scanning the full CT once per component and is dramatically faster
        # for cases containing many proposals.
        component_bbox = tuple(instance["bbox"])
        if component_policy == "union":
            assert proposal_components is not None
            component_local = (
                proposal_components[component_bbox]
                == int(instance["source_component_id"])
            )
        else:
            assert component_maps is not None
            component_local = (
                component_maps[int(instance["source_index"])][component_bbox]
                == int(instance["source_component_id"])
            )
        target_ids = np.unique(target_components[component_bbox][component_local])
        target_ids = target_ids[target_ids > 0]
        if not target_ids.size:
            continue
        positive_candidates += 1
        positive_component_mm3.append(component_mm3)
        positive_component_seed_overlap.append(
            float(np.count_nonzero(component_local & target[component_bbox]))
            / component_voxels
        )
        low_native = np.asarray([value.start for value in component_bbox], dtype=int)
        high_native = np.asarray([value.stop for value in component_bbox], dtype=int)
        for margin in margins_mm:
            margin_voxels = np.ceil(margin / spacing).astype(int)
            low = np.maximum(low_native - margin_voxels, 0)
            high = np.minimum(high_native + margin_voxels, np.asarray(target.shape))
            slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
            predictions[margin][slices] |= np.isin(
                target_components[slices], target_ids
            )
            extent_mm = (high - low) * spacing
            tiles_per_axis = [
                max(1, 1 + math.ceil(max(float(value) - 96.0, 0.0) / tile_stride_mm))
                for value in extent_mm
            ]
            tile_counts[margin] += math.prod(tiles_per_axis)
    row: dict[str, object] = {
        "case": name.removesuffix(".nii.gz"),
        "component_policy": component_policy,
        "positive_candidates": positive_candidates,
        "positive_component_mm3": positive_component_mm3,
        "positive_component_seed_overlap_fraction": positive_component_seed_overlap,
    }
    for margin in margins_mm:
        key = f"margin_{int(margin):03d}"
        row[f"{key}_dsc"] = dice(predictions[margin], target)
        row[f"{key}_tiles"] = tile_counts[margin]
    return row


def cohort_summary(
    rows: list[dict[str, object]],
    cohort_by_case: dict[str, str],
    margins_mm: tuple[float, ...],
) -> dict[str, dict[str, object]]:
    observed = {str(row["case"]) for row in rows}
    if observed != set(cohort_by_case):
        raise ValueError("cohort manifest does not exactly match analyzed cases")
    result: dict[str, dict[str, object]] = {}
    for cohort in sorted(set(cohort_by_case.values())):
        selected = [row for row in rows if cohort_by_case[str(row["case"])] == cohort]
        positive_mm3 = [
            float(value)
            for row in selected
            for value in row["positive_component_mm3"]
        ]
        positive_overlap = [
            float(value)
            for row in selected
            for value in row["positive_component_seed_overlap_fraction"]
        ]
        result[cohort] = {
            "cases": len(selected),
            "positive_candidates": sum(int(row["positive_candidates"]) for row in selected),
            "positive_geometry": {
                "seed_mm3": distribution(positive_mm3),
                "seed_overlap_fraction": distribution(positive_overlap),
            },
            "margins": {
                str(int(margin)): {
                    "macro_dsc": fmean(
                        float(row[f"margin_{int(margin):03d}_dsc"]) for row in selected
                    ),
                    "mean_tiles_per_case": fmean(
                        int(row[f"margin_{int(margin):03d}_tiles"]) for row in selected
                    ),
                }
                for margin in margins_mm
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-mm3", type=float, default=5.0)
    parser.add_argument("--max-mm3", type=float, default=20000.0)
    parser.add_argument("--tile-stride-mm", type=float, default=72.0)
    parser.add_argument(
        "--margins-mm",
        type=float,
        nargs="+",
        default=DEFAULT_MARGINS_MM,
        help="Physical margins to test around each proposal bounding box.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument(
        "--component-policy",
        choices=("union", "source_separate"),
        default="union",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional complete-label manifest used only for cohort summaries.",
    )
    args = parser.parse_args()
    margins_mm = tuple(sorted(set(args.margins_mm)))
    if (
        args.workers < 1
        or args.expected_count < 1
        or args.tile_stride_mm <= 0
        or not margins_mm
        or any(margin < 0 for margin in margins_mm)
    ):
        parser.error("invalid worker or tile stride")
    names = sorted(path.name for path in args.gt.glob("*.nii.gz"))
    if len(names) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} cases, found {len(names)}")
    for directory in args.proposal:
        if {path.name for path in directory.glob("*.nii.gz")} != set(names):
            raise ValueError(f"proposal cases do not exactly match labels: {directory}")
    tasks = [
        (
            name,
            str(args.gt / name),
            tuple(str(directory / name) for directory in args.proposal),
            args.min_mm3,
            args.max_mm3,
            args.tile_stride_mm,
            margins_mm,
            args.component_policy,
        )
        for name in names
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(analyze_case, tasks), 1):
            rows.append(row)
            print(f"[{index:03d}/{len(names):03d}] {row['case']}", flush=True)
    report = {
        "warning": "Label-informed upper bound only; never use for hidden inference or submission.",
        "cases": len(rows),
        "proposals": [str(directory.resolve()) for directory in args.proposal],
        "component_mm3": [args.min_mm3, args.max_mm3],
        "component_policy": args.component_policy,
        "tile_field_mm": 96.0,
        "tile_stride_mm": args.tile_stride_mm,
        "margins": {},
    }
    for margin in margins_mm:
        key = f"margin_{int(margin):03d}"
        report["margins"][str(int(margin))] = {
            "macro_dsc": fmean(float(row[f"{key}_dsc"]) for row in rows),
            "total_tiles": sum(int(row[f"{key}_tiles"]) for row in rows),
            "mean_tiles_per_case": fmean(int(row[f"{key}_tiles"]) for row in rows),
        }
    report["positive_candidates"] = sum(int(row["positive_candidates"]) for row in rows)
    report["positive_geometry"] = {
        "seed_mm3": distribution(
            [float(value) for row in rows for value in row["positive_component_mm3"]]
        ),
        "seed_overlap_fraction": distribution(
            [
                float(value)
                for row in rows
                for value in row["positive_component_seed_overlap_fraction"]
            ]
        ),
    }
    if args.manifest is not None:
        items = json.loads(args.manifest.read_text(encoding="utf-8")).get("items", [])
        if any(item.get("complete_label") is not True for item in items):
            raise ValueError("cohort oracle manifest requires complete labels")
        cohort_by_case = {
            str(item["name"]): str(item["cohort"])
            for item in items
        }
        if len(cohort_by_case) != len(items):
            raise ValueError("duplicate cases in cohort manifest")
        report["cohort_summary"] = cohort_summary(rows, cohort_by_case, margins_mm)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
