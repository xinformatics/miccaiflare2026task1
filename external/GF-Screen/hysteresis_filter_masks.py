#!/usr/bin/env python3
"""Keep low-threshold components only when they contain high-confidence seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    import cc3d
except ImportError:  # pragma: no cover
    cc3d = None


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def connected_components(mask):
    if cc3d is not None:
        return cc3d.connected_components(mask, connectivity=26, return_N=True)
    return ndimage.label(mask, structure=CONNECTIVITY_26)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--min-mm3", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = sorted(path.name for path in args.low.glob("*.nii.gz"))
    if not names:
        raise ValueError(f"no masks in {args.low}")
    args.output.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names, start=1):
        low_image = nib.load(args.low / name)
        high_image = nib.load(args.high / name)
        if low_image.shape != high_image.shape or not np.allclose(low_image.affine, high_image.affine):
            raise ValueError(f"geometry mismatch for {name}")
        low = np.asanyarray(low_image.dataobj).astype(bool)
        high = np.asanyarray(high_image.dataobj).astype(bool)
        components, count = connected_components(low)
        seed_ids = np.unique(components[high])
        keep = np.zeros(count + 1, dtype=bool)
        keep[seed_ids] = True
        keep[0] = False
        if args.min_mm3 > 0:
            voxel_volume = float(abs(np.linalg.det(low_image.affine[:3, :3])))
            sizes = np.bincount(components.ravel(), minlength=count + 1) * voxel_volume
            keep &= sizes >= args.min_mm3
        output = keep[components]
        header = low_image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(output.astype(np.uint8), low_image.affine, header), args.output / name)
        print(
            f"[{index:03d}/{len(names):03d}] {name}: kept {int(keep.sum())}/{count} components, "
            f"{int(output.sum()):,} voxels",
            flush=True,
        )


if __name__ == "__main__":
    main()
