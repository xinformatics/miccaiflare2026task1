#!/usr/bin/env python3
"""Compute label-informed component and voxel proposal upper bounds.

The analysis unions components from several independently generated public
prediction directories.  A component is admitted at several *ground-truth
precision* cutoffs, yielding a diagnostic upper bound on what an ideal
component confirmer could recover.  It also reports the voxel-level coverage
oracle of the proposal union.  These masks are never written and must never be
used for hidden inference or a submission.
"""

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


PRECISION_THRESHOLDS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50)


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.count_nonzero(prediction & target))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=DIRECTORY")
    name, directory = value.split("=", 1)
    if not name or not directory:
        raise argparse.ArgumentTypeError("candidate must be NAME=DIRECTORY")
    return name, Path(directory).resolve()


def load_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(bool), image


def analyze_case(item: tuple[str, str, tuple[tuple[str, str], ...]]) -> dict[str, object]:
    name, target_path, candidate_items = item
    target, target_image = load_mask(Path(target_path))
    proposal_union = np.zeros(target.shape, dtype=bool)
    support_count = np.zeros(target.shape, dtype=np.uint8)
    # Six thresholds fit into one uint8 volume. A voxel receives a bit when it
    # belongs to any component satisfying that ground-truth precision cutoff.
    component_bits = np.zeros(target.shape, dtype=np.uint8)
    source_component_counts: dict[str, int] = {}
    source_metrics: dict[str, dict[str, float]] = {}

    target_components = cc3d.connected_components(
        target.astype(np.uint8), connectivity=26
    )
    target_size = int(np.count_nonzero(target))

    for source_name, directory in candidate_items:
        prediction, image = load_mask(Path(directory) / name)
        if prediction.shape != target.shape or not np.allclose(
            image.affine, target_image.affine, atol=1e-3
        ):
            raise ValueError(f"geometry mismatch for {source_name}/{name}")
        proposal_union |= prediction
        support_count += prediction.astype(np.uint8)
        source_covered = prediction & target
        source_target_ids = np.unique(target_components[prediction])
        source_target_ids = source_target_ids[source_target_ids > 0]
        source_reconstructed = (
            np.isin(target_components, source_target_ids)
            if source_target_ids.size
            else np.zeros_like(target, dtype=bool)
        )
        source_metrics[source_name] = {
            "proposal_voxel_recall": (
                1.0
                if target_size == 0
                else float(np.count_nonzero(source_covered)) / target_size
            ),
            "voxel_fp_free_oracle_dice": dice(source_covered, target),
            "gt_component_reconstruction_oracle_dice": dice(
                source_reconstructed, target
            ),
        }
        labels = cc3d.connected_components(prediction.astype(np.uint8), connectivity=26)
        counts = np.bincount(labels.ravel())
        overlaps = np.bincount(labels[target], minlength=len(counts))
        precision = np.divide(
            overlaps,
            counts,
            out=np.zeros(len(counts), dtype=np.float64),
            where=counts > 0,
        )
        component_code = np.zeros(len(counts), dtype=np.uint8)
        for bit, threshold in enumerate(PRECISION_THRESHOLDS):
            keep = overlaps > 0 if threshold == 0 else precision >= threshold
            keep[0] = False
            component_code[keep] |= np.uint8(1 << bit)
        component_bits |= component_code[labels]
        source_component_counts[source_name] = max(0, len(counts) - 1)
        del prediction, labels, counts, overlaps, precision, component_code

    covered = proposal_union & target
    covered_size = int(np.count_nonzero(covered))
    detected_target_ids = np.unique(target_components[proposal_union])
    detected_target_ids = detected_target_ids[detected_target_ids > 0]
    reconstructed = (
        np.isin(target_components, detected_target_ids)
        if detected_target_ids.size
        else np.zeros_like(target, dtype=bool)
    )
    row: dict[str, object] = {
        "case": name.removesuffix(".nii.gz"),
        "target_voxels": target_size,
        "proposal_covered_voxels": covered_size,
        "proposal_voxel_recall": 1.0 if target_size == 0 else covered_size / target_size,
        "voxel_fp_free_oracle_dice": dice(covered, target),
        "detected_gt_components": int(detected_target_ids.size),
        "gt_component_reconstruction_oracle_dice": dice(reconstructed, target),
        "source_component_counts": source_component_counts,
        "source_metrics": source_metrics,
    }
    for source_name, directory in candidate_items:
        prediction, _ = load_mask(Path(directory) / name)
        without_source = (support_count - prediction.astype(np.uint8)) > 0
        without_covered = without_source & target
        without_target_ids = np.unique(target_components[without_source])
        without_target_ids = without_target_ids[without_target_ids > 0]
        without_reconstructed = (
            np.isin(target_components, without_target_ids)
            if without_target_ids.size
            else np.zeros_like(target, dtype=bool)
        )
        source_metrics[source_name].update(
            {
                "leave_one_out_proposal_voxel_recall": (
                    1.0
                    if target_size == 0
                    else float(np.count_nonzero(without_covered)) / target_size
                ),
                "leave_one_out_voxel_fp_free_oracle_dice": dice(
                    without_covered, target
                ),
                "leave_one_out_gt_component_reconstruction_oracle_dice": dice(
                    without_reconstructed, target
                ),
            }
        )
    for bit, threshold in enumerate(PRECISION_THRESHOLDS):
        prediction = (component_bits & np.uint8(1 << bit)) != 0
        row[f"component_precision_{threshold:g}_dice"] = dice(prediction, target)
        row[f"component_precision_{threshold:g}_voxels"] = int(
            np.count_nonzero(prediction)
        )
    return row


def summarize_cohorts(
    rows: list[dict[str, object]], cohort_by_case: dict[str, str]
) -> dict[str, dict[str, float | int]]:
    observed = {str(row["case"]) for row in rows}
    if observed != set(cohort_by_case):
        raise ValueError("cohort manifest does not exactly match analyzed cases")
    metrics = (
        "proposal_voxel_recall",
        "voxel_fp_free_oracle_dice",
        "gt_component_reconstruction_oracle_dice",
    )
    result: dict[str, dict[str, float | int]] = {}
    for cohort in sorted(set(cohort_by_case.values())):
        selected = [row for row in rows if cohort_by_case[str(row["case"])] == cohort]
        result[cohort] = {
            "cases": len(selected),
            **{
                metric: fmean(float(row[metric]) for row in selected)
                for metric in metrics
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    gt = args.gt.resolve()
    candidates = tuple((name, str(directory)) for name, directory in args.candidate)
    names = sorted(path.name for path in gt.glob("*.nii.gz"))
    if args.expected_count < 1 or len(names) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} labels, found {len(names)}")
    expected = set(names)
    for source_name, directory_text in candidates:
        directory = Path(directory_text)
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected:
            raise ValueError(
                f"{source_name} case mismatch: "
                f"missing={sorted(expected-observed)[:3]} "
                f"extra={sorted(observed-expected)[:3]}"
            )

    items = [
        (name, str(gt / name), candidates)
        for name in names
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(analyze_case, items), 1):
            rows.append(row)
            print(f"[{index:03d}/{len(names):03d}] {row['case']}", flush=True)

    summary = {
        "warning": (
            "Ground-truth-informed diagnostic upper bound only. It is not a "
            "deployable selector and must never be applied to hidden data."
        ),
        "cases": len(rows),
        "candidate_directories": dict(candidates),
        "proposal_voxel_recall": fmean(float(row["proposal_voxel_recall"]) for row in rows),
        "voxel_fp_free_oracle_dice": fmean(
            float(row["voxel_fp_free_oracle_dice"]) for row in rows
        ),
        "gt_component_reconstruction_oracle_dice": fmean(
            float(row["gt_component_reconstruction_oracle_dice"]) for row in rows
        ),
        "component_precision_oracle_dice": {
            str(threshold): fmean(
                float(row[f"component_precision_{threshold:g}_dice"])
                for row in rows
            )
            for threshold in PRECISION_THRESHOLDS
        },
        "source_attribution": {
            source_name: {
                metric: fmean(
                    float(row["source_metrics"][source_name][metric])
                    for row in rows
                )
                for metric in (
                    "proposal_voxel_recall",
                    "voxel_fp_free_oracle_dice",
                    "gt_component_reconstruction_oracle_dice",
                    "leave_one_out_proposal_voxel_recall",
                    "leave_one_out_voxel_fp_free_oracle_dice",
                    "leave_one_out_gt_component_reconstruction_oracle_dice",
                )
            }
            for source_name, _ in candidates
        },
    }
    for values in summary["source_attribution"].values():
        values["marginal_voxel_oracle_dice"] = (
            summary["voxel_fp_free_oracle_dice"]
            - values["leave_one_out_voxel_fp_free_oracle_dice"]
        )
    if args.manifest is not None:
        items = json.loads(args.manifest.read_text(encoding="utf-8")).get("items", [])
        if any(item.get("complete_label") is not True for item in items):
            raise ValueError("oracle cohort manifest requires complete labels")
        cohort_by_case = {
            str(item["name"]): str(item["cohort"])
            for item in items
        }
        if len(cohort_by_case) != len(items):
            raise ValueError("duplicate cases in oracle cohort manifest")
        summary["cohort_summary"] = summarize_cohorts(rows, cohort_by_case)
    best_threshold, best_dsc = max(
        summary["component_precision_oracle_dice"].items(),
        key=lambda item: item[1],
    )
    summary["best_component_precision_threshold"] = float(best_threshold)
    summary["best_component_oracle_dice"] = best_dsc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    csv_rows = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"source_component_counts", "source_metrics"}
        }
        flat["source_component_counts"] = json.dumps(
            row["source_component_counts"], separators=(",", ":")
        )
        flat["source_metrics"] = json.dumps(
            row["source_metrics"], separators=(",", ":")
        )
        csv_rows.append(flat)
    with (args.output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
