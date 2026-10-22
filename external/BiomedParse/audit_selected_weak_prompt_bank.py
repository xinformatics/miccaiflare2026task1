#!/usr/bin/env python3
"""Audit one frozen BiomedParse weak-domain prompt bank without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np


def inventory(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def audit(
    inputs: Path,
    images: Path,
    predictions: Path,
    nifti: Path,
    plan_path: Path,
    bank: str,
    expected: int,
) -> dict:
    if bank not in {"autopet", "lndb"}:
        raise ValueError(f"invalid bank: {bank}")
    if expected < 1:
        raise ValueError("expected must be positive")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prompts = np.asarray(plan["prompts"][bank])
    threshold = float(plan["existence_threshold"])
    threshold_tag = f"{round(100 * threshold):03d}"
    key = f"segs_e{threshold_tag}"

    input_paths = sorted(inputs.glob("*.npz"))
    prediction_paths = sorted(predictions.glob("*.npz"))
    if len(input_paths) != expected or {p.name for p in prediction_paths} != {
        p.name for p in input_paths
    }:
        raise ValueError("NPZ coverage mismatch")
    for path in prediction_paths:
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive or "prompts" not in archive:
                raise ValueError(f"missing frozen outputs: {path}")
            if not np.array_equal(np.asarray(archive["prompts"]), prompts):
                raise ValueError(f"prompt mismatch: {path}")
            mask = np.asarray(archive[key])
            if not np.all(np.isfinite(mask)):
                raise ValueError(f"non-finite prediction: {path}")

    image_paths = sorted(images.glob("*_0000.nii.gz"))
    expected_nifti = {path.name[:-12] + ".nii.gz": path for path in image_paths}
    observed_nifti = {path.name: path for path in nifti.glob("*.nii.gz")}
    if len(image_paths) != expected or set(observed_nifti) != set(expected_nifti):
        raise ValueError("NIfTI coverage mismatch")
    for name, path in observed_nifti.items():
        image = nib.load(str(expected_nifti[name]))
        prediction = nib.load(str(path))
        if image.shape[:3] != prediction.shape[:3] or not np.allclose(
            image.affine, prediction.affine, atol=1e-5
        ):
            raise ValueError(f"NIfTI geometry mismatch: {name}")
        values = np.asanyarray(prediction.dataobj)
        if values.ndim != 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid NIfTI array: {name}")
        unique = np.unique(values)
        if not set(unique.tolist()) <= {0, 1}:
            raise ValueError(f"non-binary NIfTI: {name}")

    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "bank": bank,
        "cases": expected,
        "prompts": prompts.tolist(),
        "existence_threshold": threshold,
        "disable_nms": bool(plan["disable_nms"]),
        "segmentation_key": key,
        "inputs": str(inputs.resolve()),
        "predictions": str(predictions.resolve()),
        "nifti": str(nifti.resolve()),
        "plan": str(plan_path.resolve()),
        "prediction_inventory": inventory(prediction_paths),
        "nifti_inventory": inventory(sorted(observed_nifti.values())),
        "exact_npz_coverage": True,
        "exact_nifti_coverage": True,
        "all_geometries_match": True,
        "all_masks_finite_binary": True,
        "labels_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--nifti", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bank", choices=("autopet", "lndb"), required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(
        args.inputs,
        args.images,
        args.predictions,
        args.nifti,
        args.plan,
        args.bank,
        args.expected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
