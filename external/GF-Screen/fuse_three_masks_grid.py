#!/usr/bin/env python3
"""Materialize a small, declared Boolean fusion grid for three binary masks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


FORMULAS = {
    "majority": lambda a, b, c: (a.astype(np.uint8) + b + c) >= 2,
    "union": lambda a, b, c: a | b | c,
    "intersection": lambda a, b, c: a & b & c,
    "anchor_or_rich_consensus": lambda a, b, c: a | (b & c),
    "component_or_anchor_legacy": lambda a, b, c: b | (a & c),
    "legacy_or_anchor_component": lambda a, b, c: c | (a & b),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--formulas",
        default=",".join(FORMULAS),
        help="comma-separated subset of: " + ",".join(FORMULAS),
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    labels = np.unique(data)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"non-binary mask {path}: {labels[:20].tolist()}")
    return image, data.astype(bool, copy=False)


def process_case(
    name: str,
    anchor_dir: Path,
    component_dir: Path,
    legacy_dir: Path,
    output: Path,
    formulas: tuple[str, ...],
) -> str:
    anchor_image, anchor = load_binary(anchor_dir / name)
    component_image, component = load_binary(component_dir / name)
    legacy_image, legacy = load_binary(legacy_dir / name)
    for other in (component_image, legacy_image):
        if anchor_image.shape != other.shape or not np.allclose(
            anchor_image.affine, other.affine
        ):
            raise ValueError(f"geometry mismatch for {name}")

    header = anchor_image.header.copy()
    header.set_data_dtype(np.uint8)
    for formula in formulas:
        mask = FORMULAS[formula](anchor, component, legacy)
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), anchor_image.affine, header),
            output / formula / name,
        )
    return name


def main() -> None:
    args = parse_args()
    formulas = tuple(item.strip() for item in args.formulas.split(",") if item.strip())
    unknown = sorted(set(formulas) - set(FORMULAS))
    if unknown:
        raise ValueError(f"unknown formulas: {unknown}")
    if not formulas:
        raise ValueError("at least one formula is required")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    name_sets = [
        {path.name for path in directory.glob("*.nii.gz")}
        for directory in (args.anchor, args.component, args.legacy)
    ]
    if not name_sets[0] or name_sets[0] != name_sets[1] or name_sets[0] != name_sets[2]:
        raise ValueError("the three prediction directories must have identical case names")
    names = sorted(name_sets[0])
    for formula in formulas:
        (args.output / formula).mkdir(parents=True, exist_ok=True)

    def work(name: str) -> str:
        return process_case(
            name, args.anchor, args.component, args.legacy, args.output, formulas
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, name in enumerate(executor.map(work, names), start=1):
            print(f"[{index:03d}/{len(names):03d}] {name}", flush=True)


if __name__ == "__main__":
    main()
