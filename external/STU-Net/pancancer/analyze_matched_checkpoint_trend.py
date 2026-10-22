#!/usr/bin/env python3
"""Fast public-only trend audit for two STU-Net checkpoints on matched cases.

This intentionally omits NSD and the broad deployment grid. It is a screening
diagnostic for trained-weight movement, never a fixed-public promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)
STRUCTURE_26 = ndimage.generate_binary_structure(3, 3)
MINIMUM_LESION_MM3 = 3000.0
DETECTION_FRACTION = 0.10
PROBABILITY_THRESHOLD = 0.70


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_binary(path: Path, reference: nib.Nifti1Image | None = None):
    image = nib.load(str(path))
    if reference is not None and (
        image.shape != reference.shape
        or not np.allclose(image.affine, reference.affine, atol=1e-4)
    ):
        raise ValueError(f"geometry mismatch: {path}")
    values = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(values), (0, 1)).all():
        raise ValueError(f"nonbinary mask: {path}")
    return image, values.astype(bool, copy=False)


def load_probability(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    with np.load(path) as archive:
        values = archive["lesion_probability_uint16"]
    if values.dtype != np.uint16 or values.shape != shape:
        raise ValueError(f"invalid compact probability: {path}")
    return values


def probability_name(case: str) -> str:
    return case.removesuffix(".nii.gz") + ".prob16.npz"


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * int((prediction & target).sum()) / denominator


def confirmed_anchor_dice(
    anchor: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    freetumor: np.ndarray,
) -> float:
    if not anchor.any():
        return dice(anchor, target)
    coordinates = np.nonzero(anchor)
    extent = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)
    labels, count = ndimage.label(anchor[extent], structure=STRUCTURE_26)
    keep = np.zeros(count + 1, dtype=bool)
    support = (probability[extent] > PROBABILITY_THRESHOLD * 65535.0) | freetumor[extent]
    keep[np.unique(labels[support & (labels > 0)])] = True
    keep[0] = False
    confirmed = keep[labels]
    predicted_voxels = int(confirmed.sum())
    target_voxels = int(target.sum())
    overlap = int((confirmed & target[extent]).sum())
    denominator = predicted_voxels + target_voxels
    return 1.0 if denominator == 0 else 2.0 * overlap / denominator


def missed_large_recovery(
    target: np.ndarray,
    anchor: np.ndarray,
    candidates: dict[str, np.ndarray],
    voxel_mm3: float,
) -> tuple[int, dict[str, int]]:
    if not target.any():
        return 0, {key: 0 for key in candidates}
    coordinates = np.nonzero(target)
    extent = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)
    labels, count = ndimage.label(target[extent], structure=STRUCTURE_6)
    sizes = np.bincount(labels.ravel(), minlength=count + 1).astype(float)
    anchor_overlap = np.bincount(
        labels.ravel(), weights=anchor[extent].ravel(), minlength=count + 1
    )
    lesion_ids = np.arange(1, count + 1)
    missed = lesion_ids[
        (sizes[1:] * voxel_mm3 >= MINIMUM_LESION_MM3)
        & (anchor_overlap[1:] / sizes[1:] < DETECTION_FRACTION)
    ]
    recovered = {}
    for key, mask in candidates.items():
        overlap = np.bincount(
            labels.ravel(), weights=mask[extent].ravel(), minlength=count + 1
        )
        recovered[key] = int(np.count_nonzero(overlap[missed] / sizes[missed] >= DETECTION_FRACTION))
    return int(len(missed)), recovered


def validate_terminals(directory: Path, expected_sha: str) -> list[str]:
    paths = sorted(directory.glob("BASE_CASE_*.json"))
    cases = []
    for path in paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        if (
            item.get("status") != "COMPLETE"
            or item.get("dataset_role") != "public"
            or item.get("case_count") != 1
            or item.get("checkpoint", {}).get("sha256") != expected_sha
            or item.get("public_labels_read")
            or item.get("hidden_read")
            or item.get("submission_created")
            or item.get("uploaded")
        ):
            raise ValueError(f"invalid preserved terminal: {path}")
        cases.extend(item["cases"])
    if len(paths) != 31 or len(cases) != 31 or len(set(cases)) != 31:
        raise ValueError("expected exactly 31 unique preserved epoch-130 cases")
    return sorted(cases)


def bootstrap_interval(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260826)
    samples = array[rng.integers(0, len(array), size=(20000, len(array)))].mean(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--freetumor", required=True, type=Path)
    parser.add_argument("--epoch100", required=True, type=Path)
    parser.add_argument("--epoch130", required=True, type=Path)
    parser.add_argument("--epoch130-terminals", required=True, type=Path)
    parser.add_argument("--checkpoint100", required=True, type=Path)
    parser.add_argument("--checkpoint130", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint100_sha = sha256(args.checkpoint100)
    checkpoint130_sha = sha256(args.checkpoint130)
    cases = validate_terminals(args.epoch130_terminals, checkpoint130_sha)
    metrics = {
        "epoch100_standalone": [],
        "epoch130_standalone": [],
        "epoch100_confirm_either_p70": [],
        "epoch130_confirm_either_p70": [],
    }
    recovery = {
        "anchor_missed": 0,
        "epoch100_p50": 0,
        "epoch100_p70": 0,
        "epoch130_p50": 0,
        "epoch130_p70": 0,
    }
    per_case = []
    for index, case in enumerate(cases, 1):
        target_image, target = load_binary(args.labels / case)
        _, anchor = load_binary(args.anchor / case, target_image)
        _, freetumor = load_binary(args.freetumor / case, target_image)
        _, mask100 = load_binary(args.epoch100 / case, target_image)
        _, mask130 = load_binary(args.epoch130 / case, target_image)
        prob100 = load_probability(args.epoch100 / probability_name(case), target.shape)
        prob130 = load_probability(args.epoch130 / probability_name(case), target.shape)
        p70_100 = prob100 > PROBABILITY_THRESHOLD * 65535.0
        p70_130 = prob130 > PROBABILITY_THRESHOLD * 65535.0
        values = {
            "epoch100_standalone": dice(mask100, target),
            "epoch130_standalone": dice(mask130, target),
            "epoch100_confirm_either_p70": confirmed_anchor_dice(anchor, target, prob100, freetumor),
            "epoch130_confirm_either_p70": confirmed_anchor_dice(anchor, target, prob130, freetumor),
        }
        for key, value in values.items():
            metrics[key].append(value)
        voxel_mm3 = float(abs(np.linalg.det(target_image.affine[:3, :3])))
        missed, recovered = missed_large_recovery(
            target,
            anchor,
            {
                "epoch100_p50": mask100,
                "epoch100_p70": p70_100,
                "epoch130_p50": mask130,
                "epoch130_p70": p70_130,
            },
            voxel_mm3,
        )
        recovery["anchor_missed"] += missed
        for key, value in recovered.items():
            recovery[key] += value
        per_case.append({"case": case, **values})
        print(f"[{index:02d}/{len(cases):02d}] {case}", flush=True)

    standalone_delta = [b - a for a, b in zip(metrics["epoch100_standalone"], metrics["epoch130_standalone"])]
    confirmation_delta = [
        b - a for a, b in zip(
            metrics["epoch100_confirm_either_p70"],
            metrics["epoch130_confirm_either_p70"],
        )
    ]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "role": "matched_31_case_checkpoint_trend_diagnostic_only",
        "case_count": len(cases),
        "selection_authority": False,
        "reason": "preserved processing-order subset is not the fixed public 50 and NSD was intentionally omitted",
        "checkpoint100": {"path": str(args.checkpoint100.resolve()), "sha256": checkpoint100_sha},
        "checkpoint130": {"path": str(args.checkpoint130.resolve()), "sha256": checkpoint130_sha},
        "mean_DSC": {key: float(np.mean(values)) for key, values in metrics.items()},
        "paired_delta_epoch130_minus_epoch100": {
            "standalone_DSC": float(np.mean(standalone_delta)),
            "standalone_DSC_bootstrap95": bootstrap_interval(standalone_delta),
            "confirm_either_p70_DSC": float(np.mean(confirmation_delta)),
            "confirm_either_p70_DSC_bootstrap95": bootstrap_interval(confirmation_delta),
        },
        "high_value_recovery": recovery,
        "large_candidate_component_quality": {
            "omitted": True,
            "reason": "epoch-100 full-50 precision is already measured; matched subset is a trained-weight trend screen only"
        },
        "per_case": per_case,
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: payload[key] for key in (
        "mean_DSC", "paired_delta_epoch130_minus_epoch100",
        "high_value_recovery",
    )}, indent=2))


if __name__ == "__main__":
    main()
