#!/usr/bin/env python3
"""Materialize an exact component-addition configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--mode", choices=("disjoint", "all"), required=True)
    parser.add_argument("--min-mm3", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base_paths = sorted(args.base.glob("*.nii.gz"))
    if not base_paths:
        raise RuntimeError(f"no base masks found in {args.base}")
    for index, base_path in enumerate(base_paths, 1):
        support_path = args.support / base_path.name
        if not support_path.exists():
            raise FileNotFoundError(support_path)
        base_image = nib.load(str(base_path))
        base = np.asanyarray(base_image.dataobj) > 0
        support = np.asanyarray(nib.load(str(support_path)).dataobj) > 0
        if base.shape != support.shape:
            raise ValueError(f"shape mismatch for {base_path.name}")
        voxel_mm3 = float(np.prod(base_image.header.get_zooms()[:3]))
        component_map = cc3d.connected_components(support, connectivity=26)
        sizes = np.bincount(component_map.ravel())
        eligible = sizes * voxel_mm3 >= args.min_mm3
        eligible[0] = False
        if args.mode == "disjoint":
            overlaps_base = np.bincount(component_map[base], minlength=len(sizes))
            eligible &= overlaps_base == 0
        output = (base | eligible[component_map]).astype(np.uint8)
        header = base_image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(output, base_image.affine, header), str(args.output / base_path.name))
        print(f"[{index:03d}/{len(base_paths):03d}] {base_path.name}", flush=True)


if __name__ == "__main__":
    main()
