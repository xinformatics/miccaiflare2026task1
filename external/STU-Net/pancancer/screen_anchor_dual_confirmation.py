#!/usr/bin/env python3
"""DSC-first screen for pruning anchor components with two independent models.

The anchor supplies every deployed contour. A whole anchor component is kept
when STU-Net, FreeTumor, either model, or both models overlap it by at least one
voxel. No public prediction is written and NSD/hidden inference are deferred
unless the immutable DSC preparation gate is reached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    import cc3d
except ImportError:  # pragma: no cover
    cc3d = None


DSC_GATE = 0.6154316379500749
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)
PROBABILITY_THRESHOLDS = (0.30, 0.40, 0.60, 0.70)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if cc3d is not None:
        return cc3d.connected_components(mask, connectivity=26, return_N=True)
    return ndimage.label(mask, CONNECTIVITY_26)


def supported_component_ids(
    component_map: np.ndarray, anchor: np.ndarray, support: np.ndarray
) -> np.ndarray:
    ids = np.zeros(int(component_map.max()) + 1, dtype=bool)
    observed = np.unique(component_map[anchor & support])
    ids[observed] = True
    ids[0] = False
    return ids


def component_dice(
    keep: np.ndarray,
    sizes: np.ndarray,
    target_overlap: np.ndarray,
    target_size: int,
) -> float:
    prediction_size = int(sizes[keep].sum())
    intersection = int(target_overlap[keep].sum())
    denominator = prediction_size + int(target_size)
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def selection_masks(stu: np.ndarray, ftu: np.ndarray) -> dict[str, np.ndarray]:
    if stu.shape != ftu.shape:
        raise ValueError("component-support vectors must have equal shape")
    anchor = np.ones(stu.shape, dtype=bool)
    anchor[0] = False
    return {
        "anchor": anchor,
        "stu_any": stu,
        "ftu_any": ftu,
        "either_any": stu | ftu,
        "both_any": stu & ftu,
    }


def load_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(bool), image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--stunet", required=True, type=Path)
    parser.add_argument("--freetumor", required=True, type=Path)
    parser.add_argument("--stunet-probabilities", type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    names = sorted(path.name for path in args.labels.glob("*.nii.gz"))
    if len(names) != 50:
        raise ValueError(f"fixed-public inventory must contain 50 cases, found {len(names)}")
    directories = (args.anchor, args.stunet, args.freetumor)
    for directory in directories:
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != set(names):
            raise ValueError(f"inventory mismatch: {directory}")

    if args.stunet_probabilities is not None:
        probability_names = {
            path.name for path in args.stunet_probabilities.glob("*.prob16.npz")
        }
        expected_probability_names = {
            name.removesuffix(".nii.gz") + ".prob16.npz" for name in names
        }
        if probability_names != expected_probability_names:
            raise ValueError("STU-Net probability inventory mismatch")

    configuration_names = ["anchor", "stu_any", "ftu_any", "either_any", "both_any"]
    if args.stunet_probabilities is not None:
        for threshold in PROBABILITY_THRESHOLDS:
            tag = int(round(threshold * 100))
            configuration_names.extend((f"stu_p{tag:02d}_any", f"either_p{tag:02d}_any"))
    values: dict[str, list[float]] = {name: [] for name in configuration_names}
    selected_counts = {name: 0 for name in values}
    selected_true_counts = {name: 0 for name in values}
    per_case = []
    for index, name in enumerate(names, 1):
        anchor, reference = load_mask(args.anchor / name)
        stu, stu_image = load_mask(args.stunet / name)
        ftu, ftu_image = load_mask(args.freetumor / name)
        target, target_image = load_mask(args.labels / name)
        for candidate, image, role in (
            (stu, stu_image, "STU-Net"),
            (ftu, ftu_image, "FreeTumor"),
            (target, target_image, "label"),
        ):
            if candidate.shape != anchor.shape:
                raise ValueError(f"{role} shape mismatch for {name}")
            if not np.allclose(image.affine, reference.affine, atol=1e-3):
                raise ValueError(f"{role} affine mismatch for {name}")

        component_map, count = connected_components(anchor)
        sizes = np.bincount(component_map.ravel(), minlength=count + 1)
        target_overlap = np.bincount(component_map[target], minlength=count + 1)
        stu_ids = supported_component_ids(component_map, anchor, stu)
        ftu_ids = supported_component_ids(component_map, anchor, ftu)
        selections = selection_masks(stu_ids, ftu_ids)
        if args.stunet_probabilities is not None:
            probability_path = (
                args.stunet_probabilities
                / (name.removesuffix(".nii.gz") + ".prob16.npz")
            )
            with np.load(probability_path) as archive:
                quantized = archive["lesion_probability_uint16"]
            if quantized.dtype != np.uint16 or quantized.shape != anchor.shape:
                raise ValueError(f"invalid compact probability: {probability_path}")
            for threshold in PROBABILITY_THRESHOLDS:
                tag = int(round(threshold * 100))
                threshold_ids = supported_component_ids(
                    component_map,
                    anchor,
                    quantized > int(threshold * 65535),
                )
                selections[f"stu_p{tag:02d}_any"] = threshold_ids
                selections[f"either_p{tag:02d}_any"] = threshold_ids | ftu_ids
        row = {"case": name}
        for configuration, keep in selections.items():
            score = component_dice(
                keep, sizes, target_overlap, int(np.count_nonzero(target))
            )
            values[configuration].append(score)
            selected_counts[configuration] += int(np.count_nonzero(keep))
            selected_true_counts[configuration] += int(
                np.count_nonzero(keep & (target_overlap > 0))
            )
            row[configuration] = score
        per_case.append(row)
        print(f"[{index}/50] {name}", flush=True)

    anchor_values = np.asarray(values["anchor"], dtype=np.float64)
    configurations = []
    for configuration in configuration_names:
        if configuration == "anchor":
            continue
        scores = np.asarray(values[configuration], dtype=np.float64)
        differences = scores - anchor_values
        configurations.append(
            {
                "configuration": configuration,
                "Lesion_DSC": float(scores.mean()),
                "DSC_gain": float(differences.mean()),
                "win_rate": float(np.mean(differences > 0)),
                "loss_rate": float(np.mean(differences < 0)),
                "selected_components": selected_counts[configuration],
                "selected_true_components": selected_true_counts[configuration],
                "passes_dsc_gate": bool(scores.mean() >= DSC_GATE),
            }
        )
    configurations.sort(key=lambda row: row["Lesion_DSC"], reverse=True)
    best = configurations[0]
    decision = "QUALIFY_EXACT_NSD_MATERIALIZATION" if best["passes_dsc_gate"] else "KEEP_TOUCHCOMP"
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "selection_policy": "whole_anchor_components_confirmed_by_STUNet_and_or_FreeTumor",
        "case_count": len(names),
        "connectivity": 26,
        "probability_thresholds": (
            list(PROBABILITY_THRESHOLDS) if args.stunet_probabilities else []
        ),
        "anchor_DSC": float(anchor_values.mean()),
        "fixed_public_DSC_gate": DSC_GATE,
        "configurations": configurations,
        "selected": best if best["passes_dsc_gate"] else None,
        "decision": decision,
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
        "per_case": per_case,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: payload[key] for key in ("decision", "anchor_DSC", "selected", "configurations")}, indent=2))


if __name__ == "__main__":
    main()
