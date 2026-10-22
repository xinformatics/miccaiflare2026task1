#!/usr/bin/env python3
"""Label-informed upper bound for the exact candidate-refiner deployment path.

This diagnostic must never be used for hidden inference or submission. It
assumes perfect candidate classification and exact target reconstruction only
inside each 96-mm refiner crop, then applies the same disconnected-union and
crop-local connected-replacement operations as deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean

import cc3d
import nibabel as nib
import numpy as np

from apply_candidate_refiner import apply_local_replacement, local_target_grid
from prepare_candidate_refiner_patches import resample_array, same_grid


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.count_nonzero(prediction & target))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def load_manifest(path: Path, disconnected: bool) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("shape") != [64, 64, 64] or manifest.get("spacing") != [1.5, 1.5, 1.5]:
        raise ValueError(f"unexpected refiner geometry: {path}")
    if bool(manifest.get("only_disconnected")) != disconnected:
        raise ValueError(f"unexpected component selection: {path}")
    return manifest


def target_for_record(
    record: dict,
    proposal_components: np.ndarray,
    target_components: np.ndarray,
    reference: nib.Nifti1Image,
) -> tuple[tuple[slice, slice, slice], np.ndarray] | None:
    seed = proposal_components == int(record["component_id"])
    target_ids = np.unique(target_components[seed])
    target_ids = target_ids[target_ids > 0]
    if not target_ids.size:
        return None
    with np.load(record["patch"], allow_pickle=False) as archive:
        patch_affine = archive["patch_affine"]
        patch_shape = tuple(int(value) for value in archive["image"].shape[1:])
    slices, local_affine, local_shape = local_target_grid(
        patch_shape, patch_affine, reference.shape[:3], reference.affine
    )
    footprint = resample_array(
        np.ones(patch_shape, dtype=np.uint8),
        patch_affine,
        local_affine,
        local_shape,
        0,
        0.0,
    ) > 0
    return slices, np.isin(target_components[slices], target_ids) & footprint


def analyze_case(
    case: str,
    target_path: Path,
    base_path: Path,
    proposal_path: Path,
    disconnected_records: list[dict],
    connected_records: list[dict],
    skip_anchored: bool = False,
) -> dict[str, object]:
    target_nii = nib.load(str(target_path))
    base_nii = nib.load(str(base_path))
    proposal_nii = nib.load(str(proposal_path))
    if not same_grid(target_nii, base_nii) or not same_grid(target_nii, proposal_nii):
        raise ValueError(f"geometry mismatch: {case}")
    target = np.asanyarray(target_nii.dataobj).astype(bool)
    base = np.asanyarray(base_nii.dataobj).astype(bool)
    # `load_sources` in deployment sums source arrays before connected-
    # component labeling, yielding C-contiguous storage. Match that layout so
    # component IDs agree exactly with the prepared manifest.
    proposal = np.ascontiguousarray(np.asanyarray(proposal_nii.dataobj).astype(bool))
    target_components = cc3d.connected_components(target.astype(np.uint8), connectivity=26)
    proposal_components = cc3d.connected_components(proposal.astype(np.uint8), connectivity=26)
    base_components = cc3d.connected_components(base.astype(np.uint8), connectivity=26)

    touched_ids = np.unique(target_components[proposal])
    touched_ids = touched_ids[touched_ids > 0]
    reconstructed_full = np.isin(target_components, touched_ids)
    output = base.copy()
    standalone = np.zeros_like(base)
    approved_base = np.zeros_like(base)
    positive_disconnected = 0
    for record in disconnected_records:
        operation = target_for_record(record, proposal_components, target_components, target_nii)
        if operation is None:
            continue
        slices, refined = operation
        output[slices] |= refined
        standalone[slices] |= refined
        approved_base[slices] |= refined
        positive_disconnected += 1

    # Perfect gating may accept only connected replacements which improve the
    # current case. Greedy best-improvement order avoids penalizing the upper
    # bound for interacting candidates while retaining the exact deployable
    # crop-local operation.
    remaining = list(connected_records)
    accepted_connected = 0
    positive_connected = 0
    connected_operations = []
    for record in remaining:
        operation = target_for_record(record, proposal_components, target_components, target_nii)
        if operation is None:
            continue
        slices, refined = operation
        seed = proposal_components == int(record["component_id"])
        base_ids = np.unique(base_components[seed])
        base_ids = base_ids[base_ids > 0]
        if not base_ids.size:
            continue
        base_local = np.isin(base_components[slices], base_ids)
        base_region = np.isin(base_components, base_ids)
        if not skip_anchored:
            connected_operations.append((slices, base_local, refined))
        standalone[slices] |= refined
        approved_base |= base_region
        apply_local_replacement(approved_base, slices, base_local, refined)
        positive_connected += 1
    while connected_operations:
        current_score = dice(output, target)
        best_index = -1
        best_score = current_score
        best_output = None
        for index, (slices, base_local, refined) in enumerate(connected_operations):
            candidate = output.copy()
            apply_local_replacement(candidate, slices, base_local, refined)
            score = dice(candidate, target)
            if score > best_score + 1e-12:
                best_index, best_score, best_output = index, score, candidate
        if best_index < 0:
            break
        output = best_output
        connected_operations.pop(best_index)
        accepted_connected += 1

    return {
        "case": case,
        "base_dsc": dice(base, target),
        "raw_support_oracle_dsc": dice(proposal & target, target),
        "full_component_oracle_dsc": dice(reconstructed_full, target),
        "base_plus_full_component_oracle_dsc": dice(base | reconstructed_full, target),
        "crop_deployment_oracle_dsc": (
            None if skip_anchored else dice(output, target)
        ),
        "crop_standalone_oracle_dsc": dice(standalone, target),
        "approved_base_crop_oracle_dsc": dice(approved_base, target),
        "detected_gt_components": int(touched_ids.size),
        "positive_disconnected_candidates": positive_disconnected,
        "positive_connected_candidates": positive_connected,
        "accepted_connected_replacements": accepted_connected,
    }


def analyze_case_task(task: tuple) -> dict[str, object]:
    return analyze_case(*task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--disconnected-work", type=Path, required=True)
    parser.add_argument("--connected-work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-anchored", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    disconnected = load_manifest(args.disconnected_work, True)
    connected = load_manifest(args.connected_work, False)
    disconnected_by_case: dict[str, list[dict]] = defaultdict(list)
    connected_by_case: dict[str, list[dict]] = defaultdict(list)
    for record in disconnected["records"]:
        disconnected_by_case[str(record["case"])].append(record)
    for record in connected["records"]:
        connected_by_case[str(record["case"])].append(record)
    disconnected_cases = {str(record["case"]) for record in disconnected["case_records"]}
    connected_cases = {str(record["case"]) for record in connected["case_records"]}
    if disconnected_cases != connected_cases or len(disconnected_cases) != 50:
        raise ValueError("public cache case mismatch")
    tasks = []
    for case in sorted(disconnected_cases):
        target_path = args.gt / f"{case}.nii.gz"
        base_path = args.base / f"{case}.nii.gz"
        case_record = next(
            record for record in disconnected["case_records"] if record["case"] == case
        )
        source_paths = case_record["source_proposals"]
        if len(source_paths) != 1:
            raise ValueError("this oracle currently requires one proposal source")
        tasks.append(
            (
                case,
                target_path,
                base_path,
                Path(source_paths[0]),
                disconnected_by_case.get(case, []),
                connected_by_case.get(case, []),
                args.skip_anchored,
            )
        )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(analyze_case_task, tasks), 1):
            rows.append(row)
            anchored = row["crop_deployment_oracle_dsc"]
            anchored_text = "skipped" if anchored is None else f"{anchored:.6f}"
            print(
                f"[{index:02d}/50] {row['case']} anchored={anchored_text} "
                f"standalone={row['crop_standalone_oracle_dsc']:.6f} "
                f"approved={row['approved_base_crop_oracle_dsc']:.6f}", flush=True
            )
    metric_names = [
        "base_dsc",
        "raw_support_oracle_dsc",
        "full_component_oracle_dsc",
        "base_plus_full_component_oracle_dsc",
        "crop_deployment_oracle_dsc",
        "crop_standalone_oracle_dsc",
        "approved_base_crop_oracle_dsc",
    ]
    if args.skip_anchored:
        metric_names.remove("crop_deployment_oracle_dsc")
    report = {
        "warning": "Public-label-informed upper bound only; never use for hidden inference or submission.",
        "cases": len(rows),
        "metrics": {name: fmean(float(row[name]) for row in rows) for name in metric_names},
        "candidate_counts": {
            name: sum(int(row[name]) for row in rows)
            for name in (
                "detected_gt_components",
                "positive_disconnected_candidates",
                "positive_connected_candidates",
                "accepted_connected_replacements",
            )
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
