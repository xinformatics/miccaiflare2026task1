#!/usr/bin/env python3
"""Materialize one frozen ResEnc S6 deployment without reading labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANCANCER_ROOT = PROJECT_ROOT / "external" / "STU-Net" / "pancancer"
sys.path.insert(0, str(PANCANCER_ROOT))
from analyze_stunet_gf_ensemble import (  # noqa: E402
    confirmed_anchor_components,
    load_compact_probability,
    selected_components,
)


DEPLOYMENTS = (
    "resenc_standalone_p50",
    "gf_anchor_confirm_either_p70",
    "gf_resenc_boundary3mm_p70p50_ftu",
    "gf_union_resenc_p40_ftu_s10_mm3000",
)


def ellipsoid_structure(spacing_mm: tuple[float, float, float], radius_mm: float):
    """Return a physical-radius morphology footprint for an array axis order."""
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("spacing must contain three positive finite values")
    if not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError("radius must be finite and nonnegative")
    radii = np.ceil(radius_mm / spacing).astype(int)
    grids = np.ogrid[tuple(slice(-value, value + 1) for value in radii)]
    distance_squared = sum(
        (grid * spacing[axis]) ** 2 for axis, grid in enumerate(grids)
    )
    return distance_squared <= radius_mm**2 + 1e-8


def boundary_refined_anchor(
    anchor: np.ndarray,
    resenc_p50: np.ndarray,
    resenc_p70: np.ndarray,
    freetumor: np.ndarray,
    spacing_mm: tuple[float, float, float],
    radius_mm: float = 3.0,
) -> np.ndarray:
    """Use learned shape near confirmed anchors without losing FTU-only lesions.

    A p70 touch selects whole incumbent components for learned refinement.  The
    p50 shape then replaces their boundary only inside a symmetric 3 mm band.
    Components supported only by the independent FreeTumor model remain whole.
    """
    anchor = np.asarray(anchor, dtype=bool)
    resenc_p50 = np.asarray(resenc_p50, dtype=bool)
    resenc_p70 = np.asarray(resenc_p70, dtype=bool)
    freetumor = np.asarray(freetumor, dtype=bool)
    empty = np.zeros_like(anchor, dtype=bool)
    model_confirmed = confirmed_anchor_components(
        anchor, resenc_p70, empty, "either"
    )
    independently_confirmed = confirmed_anchor_components(
        anchor, empty, freetumor, "either"
    )
    freetumor_only = independently_confirmed & ~model_confirmed
    if not model_confirmed.any():
        return freetumor_only
    structure = ellipsoid_structure(spacing_mm, radius_mm)
    shape_near_anchor = resenc_p50 & ndimage.binary_dilation(
        model_confirmed, structure=structure
    )
    anchor_near_shape = model_confirmed & ndimage.binary_dilation(
        resenc_p50, structure=structure
    )
    return freetumor_only | shape_near_anchor | anchor_near_shape


def deployment_mask(
    name: str,
    anchor: np.ndarray,
    resenc_p50: np.ndarray,
    freetumor: np.ndarray,
    probability: np.ndarray,
    voxel_mm3: float,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    if name == "resenc_standalone_p50":
        return np.asarray(resenc_p50, dtype=bool)
    if name == "gf_anchor_confirm_either_p70":
        return confirmed_anchor_components(
            np.asarray(anchor, dtype=bool),
            probability > 0.70,
            np.asarray(freetumor, dtype=bool),
            "either",
        )
    if name == "gf_resenc_boundary3mm_p70p50_ftu":
        return boundary_refined_anchor(
            anchor,
            resenc_p50,
            probability > 0.70,
            freetumor,
            spacing_mm,
            radius_mm=3.0,
        )
    if name == "gf_union_resenc_p40_ftu_s10_mm3000":
        additions = selected_components(
            probability > 0.40,
            voxel_mm3,
            3000.0,
            support=np.asarray(freetumor, dtype=bool),
            minimum_support_fraction=0.10,
        )
        return np.asarray(anchor, dtype=bool) | additions
    raise ValueError(f"unsupported frozen ResEnc deployment: {name}")


def load_binary(path: Path, reference: nib.Nifti1Image | None = None):
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(values), [0, 1]).all():
        raise ValueError(f"nonbinary prediction: {path}")
    if reference is not None and (
        image.shape != reference.shape
        or not np.allclose(image.affine, reference.affine, atol=1e-4)
    ):
        raise ValueError(f"prediction geometry mismatch: {path}")
    return image, values.astype(bool, copy=False)


def materialize_case(item: tuple[str, str, str, str, str, str]) -> tuple[str, int]:
    name, deployment, anchor_dir, resenc_dir, freetumor_dir, output_dir = item
    anchor_image, anchor = load_binary(Path(anchor_dir) / name)
    _, resenc = load_binary(Path(resenc_dir) / name, anchor_image)
    _, freetumor = load_binary(Path(freetumor_dir) / name, anchor_image)
    probability = load_compact_probability(
        Path(resenc_dir) / f"{name.removesuffix('.nii.gz')}.prob16.npz",
        anchor_image.shape,
    )
    voxel_mm3 = float(abs(np.linalg.det(anchor_image.affine[:3, :3])))
    spacing_mm = tuple(float(value) for value in anchor_image.header.get_zooms()[:3])
    result = deployment_mask(
        deployment, anchor, resenc, freetumor, probability, voxel_mm3, spacing_mm
    ).astype(np.uint8)
    destination = Path(output_dir) / name
    header = anchor_image.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = destination.with_name(f".{name}.{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(result, anchor_image.affine, header), str(temporary))
    written = nib.load(str(temporary))
    written_values = np.asanyarray(written.dataobj)
    if (
        written.shape != anchor_image.shape
        or not np.allclose(written.affine, anchor_image.affine, atol=1e-4)
        or not np.isin(np.unique(written_values), [0, 1]).all()
    ):
        raise ValueError(f"written deployment validation failed: {name}")
    os.replace(temporary, destination)
    return name, int(result.sum())


def inventory(directory: Path, suffix: str) -> set[str]:
    return {path.name for path in directory.glob(f"*{suffix}")}


def inventory_sha256(output: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = output / name
        digest.update(name.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", required=True, choices=DEPLOYMENTS)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--resenc", required=True, type=Path)
    parser.add_argument("--freetumor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument(
        "--dataset-role", choices=("public", "hidden"), required=True
    )
    args = parser.parse_args()
    if args.workers < 1 or args.expected_count < 1:
        raise ValueError("workers and expected count must be positive")
    names = sorted(inventory(args.anchor, ".nii.gz"))
    expected = set(names)
    if len(names) != args.expected_count:
        raise ValueError("anchor inventory count mismatch")
    for role, directory in (("resenc", args.resenc), ("freetumor", args.freetumor)):
        observed = inventory(directory, ".nii.gz")
        if observed != expected:
            raise ValueError(f"{role} inventory mismatch")
    probabilities = inventory(args.resenc, ".prob16.npz")
    expected_probabilities = {
        f"{name.removesuffix('.nii.gz')}.prob16.npz" for name in names
    }
    if probabilities != expected_probabilities:
        raise ValueError("ResEnc probability inventory mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    items = [
        (
            name,
            args.deployment,
            str(args.anchor),
            str(args.resenc),
            str(args.freetumor),
            str(args.output),
        )
        for name in names
    ]
    if args.workers == 1:
        results = list(map(materialize_case, items))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(materialize_case, items))
    observed_outputs = inventory(args.output, ".nii.gz")
    if observed_outputs != expected:
        raise ValueError("materialized output inventory mismatch")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "deployment": args.deployment,
        "case_count": len(names),
        "foreground_voxels": sum(value for _, value in results),
        "output": str(args.output.resolve()),
        "output_inventory_sha256": inventory_sha256(args.output, names),
        "dataset_role": args.dataset_role,
        "public_read": args.dataset_role == "public",
        "hidden_read": args.dataset_role == "hidden",
        "hidden_labels_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.terminal, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
