#!/usr/bin/env python3
"""Fast label-informed audit of proposal touch/coverage across complete-label cases."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean

import cc3d
import nibabel as nib
import numpy as np


METRICS = (
    "proposal_voxel_recall",
    "voxel_fp_free_oracle_dice",
    "gt_component_reconstruction_oracle_dice",
    "gt_component_recall",
)


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be NAME=DIRECTORY")
    name, directory = value.split("=", 1)
    if not name or not directory:
        raise argparse.ArgumentTypeError("source must be NAME=DIRECTORY")
    return name, Path(directory).resolve()


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    return 1.0 if denominator == 0 else float(2 * np.count_nonzero(prediction & target) / denominator)


def coverage_metrics(proposal: np.ndarray, target: np.ndarray, target_components: np.ndarray) -> dict[str, float | int]:
    target_voxels = int(np.count_nonzero(target))
    target_count = int(target_components.max())
    covered = proposal & target
    touched = np.unique(target_components[proposal])
    touched = touched[touched > 0]
    reconstructed = np.isin(target_components, touched) if touched.size else np.zeros_like(target)
    return {
        "proposal_voxel_recall": 1.0 if target_voxels == 0 else float(np.count_nonzero(covered) / target_voxels),
        "voxel_fp_free_oracle_dice": dice(covered, target),
        "gt_component_reconstruction_oracle_dice": dice(reconstructed, target),
        "gt_component_recall": 1.0 if target_count == 0 else float(len(touched) / target_count),
        "detected_gt_components": int(len(touched)),
        "target_components": target_count,
    }


def filter_union(union: np.ndarray, voxel_mm3: float, minimum_mm3: float) -> np.ndarray:
    labels = cc3d.connected_components(union.astype(np.uint8), connectivity=26)
    sizes = np.bincount(labels.ravel()) * voxel_mm3
    keep = sizes >= minimum_mm3
    keep[0] = False
    return keep[labels]


def analyze_case(task: tuple[str, str, tuple[tuple[str, str], ...], float]) -> dict[str, object]:
    name, target_path, sources, minimum_mm3 = task
    target_nii = nib.load(target_path)
    target = np.asanyarray(target_nii.dataobj).astype(bool)
    target_components = cc3d.connected_components(target.astype(np.uint8), connectivity=26)
    union = np.zeros_like(target)
    source_metrics: dict[str, dict[str, float | int]] = {}
    for source_name, directory in sources:
        image = nib.load(str(Path(directory) / name))
        if image.shape[:3] != target.shape or not np.allclose(image.affine, target_nii.affine, atol=1e-3, rtol=0):
            raise ValueError(f"geometry mismatch: {source_name}/{name}")
        proposal = np.asanyarray(image.dataobj).astype(bool)
        union |= proposal
        source_metrics[source_name] = coverage_metrics(proposal, target, target_components)
    spacing = np.linalg.norm(target_nii.affine[:3, :3], axis=0)
    union = filter_union(union, float(np.prod(spacing)), minimum_mm3)
    return {
        "case": name.removesuffix(".nii.gz"),
        **coverage_metrics(union, target, target_components),
        "source_metrics": source_metrics,
    }


def aggregate(rows: list[dict[str, object]], sources: tuple[tuple[str, str], ...], cohort_by_case: dict[str, str]) -> dict[str, object]:
    if {str(row["case"]) for row in rows} != set(cohort_by_case):
        raise ValueError("manifest and analyzed cases differ")

    def means(selected: list[dict[str, object]]) -> dict[str, float | int]:
        return {"cases": len(selected), **{metric: fmean(float(row[metric]) for row in selected) for metric in METRICS}}

    result: dict[str, object] = {
        **means(rows),
        "source_attribution": {
            source_name: {
                metric: fmean(float(row["source_metrics"][source_name][metric]) for row in rows)
                for metric in METRICS
            }
            for source_name, _ in sources
        },
        "cohort_summary": {},
    }
    for cohort in sorted(set(cohort_by_case.values())):
        selected = [row for row in rows if cohort_by_case[str(row["case"])] == cohort]
        result["cohort_summary"][cohort] = means(selected)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--min-mm3", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.expected_count < 1 or args.workers < 1 or args.min_mm3 < 0:
        parser.error("invalid count, workers, or minimum volume")
    names = sorted(path.name for path in args.gt.glob("*.nii.gz"))
    if len(names) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} labels, found {len(names)}")
    expected = set(names)
    sources = tuple((name, str(directory)) for name, directory in args.source)
    if len({name for name, _ in sources}) != len(sources):
        raise ValueError("duplicate proposal source")
    for source_name, directory in sources:
        observed = {path.name for path in Path(directory).glob("*.nii.gz")}
        if observed != expected:
            raise ValueError(f"{source_name} coverage mismatch")
    items = json.loads(args.manifest.read_text(encoding="utf-8")).get("items", [])
    if any(item.get("complete_label") is not True for item in items):
        raise ValueError("touch audit requires complete labels")
    cohort_by_case = {str(item["name"]): str(item["cohort"]) for item in items}
    if len(cohort_by_case) != len(items):
        raise ValueError("duplicate manifest case")
    tasks = [(name, str(args.gt / name), sources, args.min_mm3) for name in names]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(analyze_case, tasks), 1):
            rows.append(row)
            print(f"[{index:03d}/{len(names):03d}] {row['case']}", flush=True)
    report = {
        "warning": "Complete-label diagnostic only; never deploy an oracle selection.",
        "minimum_union_component_mm3": args.min_mm3,
        "sources": dict(sources),
        **aggregate(rows, sources, cohort_by_case),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        flat_rows = []
        for row in rows:
            flat_rows.append({**{key: value for key, value in row.items() if key != "source_metrics"}, "source_metrics": json.dumps(row["source_metrics"], separators=(",", ":"))})
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
