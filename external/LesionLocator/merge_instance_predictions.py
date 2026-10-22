#!/usr/bin/env python3
"""Merge LesionLocator instance outputs and audit conservative GF fusion modes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np


NIFTI_SUFFIX = ".nii.gz"


def case_name(reference: Path) -> str:
    name = reference.name
    if not name.endswith(NIFTI_SUFFIX):
        raise ValueError(f"not a NIfTI file: {reference}")
    stem = name[: -len(NIFTI_SUFFIX)]
    return stem[:-5] if stem.endswith("_0000") else stem


def merge_case(reference: Path, instances: Path, anchor: Path | None, mode: str) -> nib.Nifti1Image:
    ref = nib.load(reference)
    result = np.zeros(ref.shape, dtype=bool)
    instance_prefix = reference.name[: -len(NIFTI_SUFFIX)] + "_lesion_"
    candidates = sorted(
        instances.glob(instance_prefix + "*" + NIFTI_SUFFIX),
        key=lambda path: int(re.search(r"_lesion_(\d+)\.nii\.gz$", path.name).group(1)),
    )
    for candidate in candidates:
        image = nib.load(candidate)
        if image.shape != ref.shape or not np.allclose(image.affine, ref.affine, atol=1e-4):
            raise ValueError(f"geometry mismatch: {candidate} versus {reference}")
        result |= np.asarray(image.dataobj) > 0

    if mode != "refiner":
        if anchor is None:
            raise ValueError(f"mode {mode} requires an anchor")
        anchor_image = nib.load(anchor)
        if anchor_image.shape != ref.shape or not np.allclose(anchor_image.affine, ref.affine, atol=1e-4):
            raise ValueError(f"geometry mismatch: {anchor} versus {reference}")
        anchor_mask = np.asarray(anchor_image.dataobj) > 0
        if mode == "union":
            result |= anchor_mask
        elif mode == "intersection":
            result &= anchor_mask
        else:
            raise ValueError(f"unknown fusion mode: {mode}")

    header = ref.header.copy()
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(result.astype(np.uint8), ref.affine, header)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("refiner", "union", "intersection"), default="refiner")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    references = sorted(args.references.glob("*" + NIFTI_SUFFIX))
    if args.expected_count is not None and len(references) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} references, found {len(references)}")
    args.output.mkdir(parents=True, exist_ok=True)
    for index, reference in enumerate(references, 1):
        case = case_name(reference)
        anchor = args.anchor / f"{case}{NIFTI_SUFFIX}" if args.anchor else None
        output = args.output / f"{case}{NIFTI_SUFFIX}"
        nib.save(merge_case(reference, args.instances, anchor, args.mode), output)
        print(f"[{index:03d}/{len(references):03d}] {output.name}")


if __name__ == "__main__":
    main()
