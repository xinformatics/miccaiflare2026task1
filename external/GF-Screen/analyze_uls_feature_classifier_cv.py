#!/usr/bin/env python3
"""Nested grouped-CV selector using inference-visible ULS+ refinement features.

Public labels define training targets and inner-fold Dice only. Every reported
public prediction is held out by case from fitting and threshold selection.
This script has no hidden-data, packaging, or upload path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import cc3d
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import GroupKFold

from analyze_component_classifier_cv import FEATURES as SOURCE_FEATURES
from analyze_component_classifier_cv import model_grid, probability
from materialize_uls_consensus import boxes_overlap, crop_indices_and_bbox
from run_uls_plus_component_refiner import same_geometry


ULS_FEATURES = (
    "case_base_empty",
    "case_base_log_mm3",
    "case_proposal_log_mm3",
    "case_base_to_proposal_fraction",
    "case_seed40_log_mm3",
    "case_seed40_to_proposal_fraction",
    "case_seed55_log_mm3",
    "case_seed55_to_proposal_fraction",
    "case_richfull_log_mm3",
    "case_richfull_to_proposal_fraction",
    "case_whole_prompt_log_mm3",
    "case_whole_prompt_to_proposal_fraction",
    "case_tumor_prompt_log_mm3",
    "case_tumor_prompt_to_proposal_fraction",
    "case_nnunet_log_mm3",
    "case_nnunet_to_proposal_fraction",
    "uls_log_refined_mm3",
    "uls_log_addition_mm3",
    "uls_seed_coverage",
    "uls_refined_overlap",
    "uls_volume_ratio_log",
    "uls_base_fraction",
    "uls_seed40_fraction",
    "uls_seed55_fraction",
    "uls_richfull_fraction",
    "uls_whole_prompt_fraction",
    "uls_tumor_prompt_fraction",
    "uls_nnunet_fraction",
    "uls_hu_mean",
    "uls_hu_std",
    "uls_bbox_x_fraction",
    "uls_bbox_y_fraction",
    "uls_bbox_z_fraction",
    "uls_max_mutual_dice",
    "uls_agree_count_050_log",
    "uls_agree_count_070_log",
)
FEATURES = tuple(SOURCE_FEATURES) + ULS_FEATURES
SUPPORTS = ("seed40", "seed55", "richfull", "whole_prompt", "tumor_prompt", "nnunet")


@dataclass
class Candidate:
    row: dict[str, object]
    addition_indices: np.ndarray


@dataclass
class CaseStats:
    shape: tuple[int, int, int]
    base_voxels: int
    target_voxels: int
    base_overlap: int
    target_indices: np.ndarray
    reference: sitk.Image


def parse_named_directory(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("support must be NAME=DIRECTORY")
    name, directory = value.split("=", 1)
    return name, Path(directory).resolve()


def load_component_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["case"], int(row["component_id"])): row for row in rows}


def fraction(indices: np.ndarray, mask_flat: np.ndarray) -> float:
    return float(np.count_nonzero(mask_flat[indices])) / max(int(indices.size), 1)


def extract_case(
    case: str,
    records: list[dict],
    component_rows: dict[tuple[str, int], dict[str, str]],
    images: Path,
    labels: Path,
    base_dir: Path,
    supports: dict[str, Path],
    predicted_crops: Path,
) -> tuple[list[Candidate], CaseStats]:
    proposal_img = sitk.ReadImage(records[0]["proposal"])
    proposal = sitk.GetArrayFromImage(proposal_img) > 0
    shape = proposal.shape
    voxel_mm3 = float(np.prod(proposal_img.GetSpacing()))
    target_img = sitk.ReadImage(str(labels / f"{case}.nii.gz"))
    base_img = sitk.ReadImage(str(base_dir / f"{case}.nii.gz"))
    ct_img = sitk.ReadImage(str(images / f"{case}_0000.nii.gz"))
    for name, image in (("target", target_img), ("base", base_img), ("ct", ct_img)):
        if not same_geometry(proposal_img, image):
            raise RuntimeError(f"{name} geometry mismatch for {case}")
    target = sitk.GetArrayFromImage(target_img) > 0
    base = sitk.GetArrayFromImage(base_img) > 0
    ct = np.nan_to_num(
        sitk.GetArrayFromImage(ct_img).astype(np.float32),
        nan=-1024.0,
        posinf=3071.0,
        neginf=-1024.0,
    )
    support_masks = {}
    for name, directory in supports.items():
        image = sitk.ReadImage(str(directory / f"{case}.nii.gz"))
        if not same_geometry(proposal_img, image):
            raise RuntimeError(f"{name} geometry mismatch for {case}")
        support_masks[name] = sitk.GetArrayFromImage(image).astype(bool)
    support_flat = {name: mask.ravel() for name, mask in support_masks.items()}
    base_flat = base.ravel()
    target_flat = target.ravel()
    ct_flat = ct.ravel()

    proposal_xyz = np.asanyarray(nib.load(records[0]["proposal"]).dataobj).astype(bool)
    component_xyz = cc3d.connected_components(proposal_xyz.astype(np.uint8), connectivity=26)
    component_map = np.transpose(component_xyz, (2, 1, 0))
    if component_map.shape != shape or not np.array_equal(component_map > 0, proposal):
        raise RuntimeError(f"component coordinate mismatch for {case}")

    proposal_voxels = int(np.count_nonzero(proposal))
    base_voxels = int(np.count_nonzero(base))
    case_features: dict[str, float] = {
        "case_base_empty": float(base_voxels == 0),
        "case_base_log_mm3": math.log1p(base_voxels * voxel_mm3),
        "case_proposal_log_mm3": math.log1p(proposal_voxels * voxel_mm3),
        "case_base_to_proposal_fraction": base_voxels / max(proposal_voxels, 1),
    }
    for name, mask in support_masks.items():
        voxels = int(np.count_nonzero(mask))
        case_features[f"case_{name}_log_mm3"] = math.log1p(voxels * voxel_mm3)
        case_features[f"case_{name}_to_proposal_fraction"] = (
            voxels / max(proposal_voxels, 1)
        )

    temporary = []
    for record in records:
        path = predicted_crops / f"{record['crop_id']}.nii.gz"
        crop = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
        indices, bbox = crop_indices_and_bbox(crop, shape, record["start_xyz"])
        if not indices.size or bbox is None:
            continue
        z, y, x = map(int, record["click_zyx"])
        component_id = int(component_map[z, y, x])
        source = component_rows.get((case, component_id))
        # The earlier grouped classifier excluded components already touching
        # the immutable base. Keep that same candidate definition.
        if source is None:
            continue
        temporary.append((record, source, component_id, indices, bbox))

    max_mutual = np.zeros(len(temporary), dtype=np.float64)
    agree_050 = np.zeros(len(temporary), dtype=np.int64)
    agree_070 = np.zeros(len(temporary), dtype=np.int64)
    for first in range(len(temporary)):
        for second in range(first + 1, len(temporary)):
            idx_a, box_a = temporary[first][3], temporary[first][4]
            idx_b, box_b = temporary[second][3], temporary[second][4]
            if not boxes_overlap(box_a, box_b):
                continue
            overlap = np.intersect1d(idx_a, idx_b, assume_unique=True).size
            mutual = 2.0 * overlap / max(idx_a.size + idx_b.size, 1)
            max_mutual[first] = max(max_mutual[first], mutual)
            max_mutual[second] = max(max_mutual[second], mutual)
            if mutual >= 0.50:
                agree_050[first] += 1
                agree_050[second] += 1
            if mutual >= 0.70:
                agree_070[first] += 1
                agree_070[second] += 1

    candidates = []
    proposal_labels = None
    for index, (record, source, component_id, refined, bbox) in enumerate(temporary):
        if proposal_labels is None:
            proposal_labels = component_map.ravel()
        seed_voxels = int(source["component_voxels"])
        seed_intersection = int(np.count_nonzero(proposal_labels[refined] == component_id))
        addition = refined[~base_flat[refined]]
        refined_voxels = int(refined.size)
        addition_voxels = int(addition.size)
        values = ct_flat[refined]
        extents = (bbox[1] - bbox[0]).astype(np.float64)
        row: dict[str, object] = {
            "case": case,
            "component_id": component_id,
            "crop_id": record["crop_id"],
            "addition_voxels": addition_voxels,
            "addition_target_overlap": int(np.count_nonzero(target_flat[addition])),
        }
        for name in SOURCE_FEATURES:
            row[name] = float(source[name])
        row.update(case_features)
        row.update(
            {
                "uls_log_refined_mm3": math.log1p(refined_voxels * voxel_mm3),
                "uls_log_addition_mm3": math.log1p(addition_voxels * voxel_mm3),
                "uls_seed_coverage": seed_intersection / max(seed_voxels, 1),
                "uls_refined_overlap": seed_intersection / max(refined_voxels, 1),
                "uls_volume_ratio_log": math.log1p(refined_voxels / max(seed_voxels, 1)),
                "uls_base_fraction": fraction(refined, base_flat),
                "uls_hu_mean": float(np.mean(values)),
                "uls_hu_std": float(np.std(values)),
                "uls_bbox_x_fraction": extents[2] / shape[2],
                "uls_bbox_y_fraction": extents[1] / shape[1],
                "uls_bbox_z_fraction": extents[0] / shape[0],
                "uls_max_mutual_dice": float(max_mutual[index]),
                "uls_agree_count_050_log": math.log1p(int(agree_050[index])),
                "uls_agree_count_070_log": math.log1p(int(agree_070[index])),
            }
        )
        for name in SUPPORTS:
            row[f"uls_{name}_fraction"] = fraction(refined, support_flat[name])
        precision = row["addition_target_overlap"] / max(addition_voxels, 1)
        row["is_true_refinement"] = int(precision >= 0.10)
        row["addition_target_precision"] = precision
        candidates.append(Candidate(row=row, addition_indices=addition))

    stats = CaseStats(
        shape=shape,
        base_voxels=int(base.sum()),
        target_voxels=int(target.sum()),
        base_overlap=int(np.count_nonzero(base & target)),
        target_indices=np.flatnonzero(target_flat),
        reference=proposal_img,
    )
    return candidates, stats


def macro_dice(
    candidates: list[Candidate],
    probabilities: np.ndarray,
    threshold: float,
    cases: set[str],
    stats: dict[str, CaseStats],
) -> float:
    chosen: dict[str, list[np.ndarray]] = {case: [] for case in cases}
    for candidate, value in zip(candidates, probabilities):
        case = str(candidate.row["case"])
        if case in chosen and value >= threshold and candidate.addition_indices.size:
            chosen[case].append(candidate.addition_indices)
    scores = []
    for case in sorted(cases):
        additions = chosen[case]
        if additions:
            added = np.unique(np.concatenate(additions))
            added_overlap = np.intersect1d(
                added, stats[case].target_indices, assume_unique=True
            ).size
        else:
            added = np.empty(0, dtype=np.int64)
            added_overlap = 0
        predicted = stats[case].base_voxels + int(added.size)
        overlap = stats[case].base_overlap + int(added_overlap)
        scores.append(2.0 * overlap / max(predicted + stats[case].target_voxels, 1))
    return fmean(scores)


def nested_predictions(
    candidates: list[Candidate], stats: dict[str, CaseStats], seed: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    x = np.asarray(
        [[float(candidate.row[name]) for name in FEATURES] for candidate in candidates]
    )
    y = np.asarray(
        [int(candidate.row["is_true_refinement"]) for candidate in candidates],
        dtype=np.int64,
    )
    groups = np.asarray([str(candidate.row["case"]) for candidate in candidates])
    outer = GroupKFold(n_splits=5)
    thresholds = np.linspace(0.05, 0.95, 19)
    out_of_fold = np.full(len(candidates), np.nan)
    fold_rows = []
    for fold, (train_index, test_index) in enumerate(outer.split(x, y, groups), 1):
        train_groups = groups[train_index]
        inner = GroupKFold(n_splits=4)
        best = None
        for model_name, model in model_grid(seed + fold):
            inner_probability = np.full(len(train_index), np.nan)
            x_train = x[train_index]
            y_train = y[train_index]
            for inner_train, inner_val in inner.split(x_train, y_train, train_groups):
                inner_probability[inner_val] = probability(
                    model,
                    x_train[inner_train],
                    y_train[inner_train],
                    x_train[inner_val],
                )
            train_candidates = [candidates[value] for value in train_index]
            train_cases = set(train_groups)
            for threshold in thresholds:
                score = macro_dice(
                    train_candidates,
                    inner_probability,
                    float(threshold),
                    train_cases,
                    stats,
                )
                choice = (score, float(threshold), model_name, model)
                if best is None or choice[:3] > best[:3]:
                    best = choice
        assert best is not None
        inner_dice, threshold, model_name, model = best
        out_of_fold[test_index] = probability(
            model, x[train_index], y[train_index], x[test_index]
        )
        test_candidates = [candidates[value] for value in test_index]
        test_dice = macro_dice(
            test_candidates,
            out_of_fold[test_index],
            threshold,
            set(groups[test_index]),
            stats,
        )
        for value in test_index:
            candidates[value].row["outer_fold"] = fold
            candidates[value].row["oof_probability"] = float(out_of_fold[value])
            candidates[value].row["oof_threshold"] = threshold
            candidates[value].row["oof_selected"] = int(out_of_fold[value] >= threshold)
        fold_rows.append(
            {
                "fold": fold,
                "model": model_name,
                "threshold": threshold,
                "inner_oof_DSC": inner_dice,
                "outer_DSC": test_dice,
            }
        )
    if not np.isfinite(out_of_fold).all():
        raise RuntimeError("incomplete grouped predictions")
    return out_of_fold, fold_rows


def materialize(
    candidates: list[Candidate], stats: dict[str, CaseStats], base: Path, output: Path
) -> None:
    by_case: dict[str, list[np.ndarray]] = {case: [] for case in stats}
    for candidate in candidates:
        if int(candidate.row["oof_selected"]):
            by_case[str(candidate.row["case"])].append(candidate.addition_indices)
    output.mkdir(parents=True, exist_ok=True)
    for case, case_stats in sorted(stats.items()):
        base_img = sitk.ReadImage(str(base / f"{case}.nii.gz"))
        prediction = sitk.GetArrayFromImage(base_img) > 0
        if by_case[case]:
            prediction.ravel()[np.unique(np.concatenate(by_case[case]))] = True
        image = sitk.GetImageFromArray(prediction.astype(np.uint8))
        image.CopyInformation(base_img)
        sitk.WriteImage(image, str(output / f"{case}.nii.gz"), True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--component-csv", type=Path, required=True)
    parser.add_argument("--support", type=parse_named_directory, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    supports = dict(args.support)
    if set(supports) != set(SUPPORTS):
        raise ValueError(f"supports must be exactly {sorted(SUPPORTS)}")

    records = json.loads((args.work / "manifest.json").read_text())
    by_case: dict[str, list[dict]] = {}
    for record in records:
        by_case.setdefault(record["case"], []).append(record)
    if len(by_case) != 50:
        raise ValueError(f"expected 50 manifest cases, found {len(by_case)}")
    component_rows = load_component_rows(args.component_csv)
    candidates = []
    stats = {}
    for index, (case, case_records) in enumerate(sorted(by_case.items()), 1):
        case_candidates, case_stats = extract_case(
            case,
            case_records,
            component_rows,
            args.images,
            args.labels,
            args.base,
            supports,
            args.work / "predicted_crops",
        )
        candidates.extend(case_candidates)
        stats[case] = case_stats
        print(f"[{index:02d}/50] {case} ULS_candidates={len(case_candidates)}", flush=True)
    probabilities, folds = nested_predictions(candidates, stats, args.seed)
    cases = set(stats)
    # The exact fold-specific thresholds are carried on each candidate.
    selected_probabilities = np.asarray(
        [1.0 if int(candidate.row["oof_selected"]) else 0.0 for candidate in candidates]
    )
    oof_dice = macro_dice(candidates, selected_probabilities, 0.5, cases, stats)
    anchor = fmean(
        2.0 * value.base_overlap / max(value.base_voxels + value.target_voxels, 1)
        for value in stats.values()
    )
    selected = [candidate for candidate in candidates if int(candidate.row["oof_selected"])]
    report = {
        "warning": "Public-only nested case-grouped ULS+ selector; no hidden model exists.",
        "cases": len(stats),
        "candidates": len(candidates),
        "true_refinements": sum(int(c.row["is_true_refinement"]) for c in candidates),
        "selected_refinements": len(selected),
        "true_selected_refinements": sum(int(c.row["is_true_refinement"]) for c in selected),
        "anchor_macro_DSC": anchor,
        "oof_macro_DSC": oof_dice,
        "oof_macro_DSC_gain": oof_dice - anchor,
        "features": FEATURES,
        "folds": folds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    serial_rows = [candidate.row for candidate in candidates]
    with (args.output_dir / "uls_candidates_oof.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=serial_rows[0].keys())
        writer.writeheader()
        writer.writerows(serial_rows)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    materialize(candidates, stats, args.base, args.output_dir / "predictions_oof")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
