#!/usr/bin/env python3
"""Materialize one preselected component-confirmation configuration."""

import argparse
import math
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
    return ndimage.label(mask, CONNECTIVITY_26)


def load_mask(path):
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(bool), image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gf", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distance-mm", required=True, type=float)
    parser.add_argument("--rescue-min-mm3", required=True, type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.gf.glob("*.nii.gz"))
    if not paths:
        raise SystemExit(f"no GF predictions in {args.gf}")
    for index, gf_path in enumerate(paths, 1):
        support_path = args.support / gf_path.name
        if not support_path.is_file():
            raise SystemExit(f"missing support prediction {support_path}")
        gf, gf_image = load_mask(gf_path)
        support, support_image = load_mask(support_path)
        if gf.shape != support.shape or not np.allclose(
            gf_image.affine, support_image.affine, atol=1e-3
        ):
            raise SystemExit(f"geometry mismatch for {gf_path.name}")

        spacing = tuple(float(value) for value in gf_image.header.get_zooms()[:3])
        voxel_volume = float(np.prod(spacing))
        component_map, component_count = connected_components(gf)
        component_sizes = np.bincount(component_map.ravel(), minlength=component_count + 1)
        if args.distance_mm <= 0:
            supported = support
        else:
            distance = ndimage.distance_transform_edt(~support, sampling=spacing)
            supported = distance <= args.distance_mm + 1e-6
        supported_ids = np.unique(component_map[supported & gf])
        keep_ids = np.zeros(component_count + 1, dtype=bool)
        keep_ids[supported_ids] = True
        keep_ids[0] = False
        if math.isfinite(args.rescue_min_mm3):
            keep_ids |= component_sizes * voxel_volume >= args.rescue_min_mm3
            keep_ids[0] = False
        fused = keep_ids[component_map].astype(np.uint8)

        header = gf_image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(fused, gf_image.affine, header), args.output / gf_path.name)
        print(f"{index}/{len(paths)} {gf_path.name} kept={int(fused.sum())}", flush=True)


if __name__ == "__main__":
    main()
