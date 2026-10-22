#!/usr/bin/env python3
"""Compare union and source-separated proposal connectivity without labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np

from proposal_component_instances import instance_local_mask, source_separated_instances


def strip_nii(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def process_case(task: dict[str, object]) -> dict[str, object]:
    item = dict(task["item"])
    paths = [Path(value) / str(item["prediction_name"]) for value in task["sources"]]
    images = [nib.load(str(path)) for path in paths]
    reference = images[0]
    for path, image in zip(paths[1:], images[1:]):
        if image.shape[:3] != reference.shape[:3] or not np.allclose(
            image.affine, reference.affine, atol=1e-3, rtol=0
        ):
            raise ValueError(f"proposal geometry mismatch: {item['name']} {path}")
    masks = [np.asanyarray(image.dataobj) > 0 for image in images]
    component_maps, instances = source_separated_instances(masks)
    union_map, union_count = cc3d.connected_components(
        np.logical_or.reduce(masks).astype(np.uint8), connectivity=26, return_N=True
    )
    union_stats = cc3d.statistics(union_map)
    spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
    voxel_mm3 = float(np.prod(spacing))
    minimum = float(task["min_mm3"])
    maximum = float(task["max_mm3"])

    union_volumes = [
        int(union_stats["voxel_counts"][component_id]) * voxel_mm3
        for component_id in range(1, union_count + 1)
    ]
    kept_union = {
        component_id
        for component_id, volume in enumerate(union_volumes, 1)
        if minimum <= volume <= maximum
    }
    kept_instances = []
    separated_volumes = []
    duplicate_source_components = 0
    for instance in instances:
        volume = int(instance["voxels"]) * voxel_mm3
        if not minimum <= volume <= maximum:
            continue
        separated_volumes.append(volume)
        duplicate_source_components += len(instance["equivalent_components"]) - 1
        bbox = tuple(instance["bbox"])
        local = instance_local_mask(component_maps, instance)
        union_ids = np.unique(union_map[bbox][local])
        union_ids = union_ids[union_ids > 0]
        if union_ids.size != 1:
            raise RuntimeError(
                f"source component did not map to one union component: "
                f"{item['name']} {instance['component_id']} {union_ids.tolist()}"
            )
        kept_instances.append((int(instance["component_id"]), int(union_ids[0])))

    by_union: dict[int, list[int]] = defaultdict(list)
    for component_id, union_id in kept_instances:
        if union_id in kept_union:
            by_union[union_id].append(component_id)
    bridge_sizes = [len(values) for values in by_union.values() if len(values) > 1]
    source_raw_counts = [int(component_map.max()) for component_map in component_maps]
    return {
        "case": str(item["name"]),
        "cohort": str(item.get("cohort", "unknown")),
        "source_raw_components": source_raw_counts,
        "source_raw_total": sum(source_raw_counts),
        "source_separated_unique": len(instances),
        "source_separated_kept": len(kept_instances),
        "exact_duplicate_source_components_kept": duplicate_source_components,
        "union_components": union_count,
        "union_kept": len(kept_union),
        "bridged_union_components": len(bridge_sizes),
        "instances_in_bridged_unions": sum(bridge_sizes),
        "largest_bridge_instances": max(bridge_sizes, default=0),
        "source_separated_volumes_mm3": separated_volumes,
        "union_volumes_mm3": [
            volume
            for component_id, volume in enumerate(union_volumes, 1)
            if component_id in kept_union
        ],
    }


def aggregate(rows: list[dict[str, object]], source_names: list[str]) -> dict[str, object]:
    separated_volumes = [
        value for row in rows for value in row["source_separated_volumes_mm3"]
    ]
    union_volumes = [value for row in rows for value in row["union_volumes_mm3"]]
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "source_raw_total",
            "source_separated_unique",
            "source_separated_kept",
            "exact_duplicate_source_components_kept",
            "union_components",
            "union_kept",
            "bridged_union_components",
            "instances_in_bridged_unions",
        )
    }
    totals["cases_with_bridged_unions"] = sum(
        int(row["bridged_union_components"]) > 0 for row in rows
    )
    totals["largest_bridge_instances"] = max(
        (int(row["largest_bridge_instances"]) for row in rows), default=0
    )
    totals["source_raw_components"] = {
        name: sum(int(row["source_raw_components"][index]) for row in rows)
        for index, name in enumerate(source_names)
    }
    totals["source_separated_volume_mm3"] = summarize(separated_volumes)
    totals["union_volume_mm3"] = summarize(union_volumes)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proposal", action="append", required=True)
    parser.add_argument("--min-mm3", type=float, default=5.0)
    parser.add_argument("--max-mm3", type=float, default=1e15)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.min_mm3 < 0 or args.max_mm3 <= args.min_mm3:
        parser.error("invalid worker or component volume limit")
    source_names, sources = [], []
    for value in args.proposal:
        if "=" not in value:
            parser.error(f"proposal must be NAME=DIRECTORY: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not name or name in source_names or not path.is_dir():
            parser.error(f"invalid proposal source: {value}")
        source_names.append(name)
        sources.append(path)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("manifest contains no items")
    if args.case_limit < 0:
        parser.error("case limit must be nonnegative")
    if args.case_limit:
        items = items[: args.case_limit]
    tasks = [
        {
            "item": item,
            "sources": [str(path) for path in sources],
            "min_mm3": args.min_mm3,
            "max_mm3": args.max_mm3,
        }
        for item in items
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(process_case, tasks), 1):
            rows.append(row)
            print(
                f"[{index:03d}/{len(tasks):03d}] {row['case']}: "
                f"separate={row['source_separated_kept']} union={row['union_kept']} "
                f"bridges={row['bridged_union_components']}",
                flush=True,
            )
    cohorts = {
        cohort: aggregate(
            [row for row in rows if row["cohort"] == cohort], source_names
        )
        for cohort in sorted({str(row["cohort"]) for row in rows})
    }
    report = {
        "schema_version": 1,
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "labels_read": False,
        "component_policies": ["union", "source_separate_exact_dedup"],
        "connectivity": 26,
        "component_mm3": [args.min_mm3, args.max_mm3],
        "manifest": str(args.manifest.resolve()),
        "sources": [
            {"name": name, "directory": str(path)}
            for name, path in zip(source_names, sources)
        ],
        "cases": len(rows),
        "overall": aggregate(rows, source_names),
        "by_cohort": cohorts,
        "case_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
