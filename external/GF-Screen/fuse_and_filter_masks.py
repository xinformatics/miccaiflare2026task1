#!/usr/bin/env python3
"""Fuse binary GF-Screen masks and remove tiny components in physical units."""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    import cc3d
except ImportError:  # pragma: no cover
    cc3d = None


CONNECTIVITY_6 = ndimage.generate_binary_structure(3, 1)
CONNECTIVITY_26 = ndimage.generate_binary_structure(3, 3)


def connected_components_6(mask: np.ndarray):
    """Preserve SciPy's historical 6-connectivity with a faster backend."""
    if cc3d is not None:
        return cc3d.connected_components(mask, connectivity=6, return_N=True)
    return ndimage.label(mask, structure=CONNECTIVITY_6)


def connected_components(mask: np.ndarray, connectivity: int):
    if connectivity not in (6, 26):
        raise ValueError("connectivity must be 6 or 26")
    if cc3d is not None:
        return cc3d.connected_components(
            mask, connectivity=connectivity, return_N=True
        )
    structure = CONNECTIVITY_6 if connectivity == 6 else CONNECTIVITY_26
    return ndimage.label(mask, structure=structure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a", type=Path, required=True)
    parser.add_argument("--pred-b", type=Path)
    parser.add_argument("--fusion", choices=("a", "union", "intersection"), default="a")
    parser.add_argument("--min-mm3", type=float, default=0.0)
    parser.add_argument(
        "--connectivity", choices=(6, 26), default=6, type=int,
        help="component connectivity; default 6 preserves historical outputs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="number of independent cases filtered concurrently (default: 4)",
    )
    return parser.parse_args()


def load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    labels = np.unique(data)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{path} is not binary: {labels[:20].tolist()}")
    return image, data.astype(bool, copy=False)


def filter_components(
    mask: np.ndarray,
    voxel_volume: float,
    min_mm3: float,
    connectivity: int = 6,
) -> np.ndarray:
    if min_mm3 <= 0 or not mask.any():
        return mask
    coordinates = np.nonzero(mask)
    extent = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates
    )
    mask_crop = mask[extent]
    components, count = connected_components(mask_crop, connectivity)
    if count == 0:
        return mask
    sizes = np.bincount(components.ravel()) * voxel_volume
    keep = sizes >= min_mm3
    keep[0] = False
    filtered = np.zeros_like(mask, dtype=bool)
    filtered[extent] = keep[components]
    return filtered


def process_prediction(
    name: str,
    pred_a: Path,
    pred_b: Path | None,
    fusion: str,
    min_mm3: float,
    output: Path,
    connectivity: int = 6,
) -> tuple[str, int | None]:
    """Filter one independent case; safe for ordered thread-pool execution."""
    if fusion == "a" and pred_b is None and min_mm3 <= 0:
        shutil.copyfile(pred_a / name, output / name)
        return name, None

    image_a, mask_a = load_binary(pred_a / name)
    mask = mask_a
    if pred_b is not None:
        image_b, mask_b = load_binary(pred_b / name)
        if image_a.shape != image_b.shape or not np.allclose(image_a.affine, image_b.affine):
            raise ValueError(f"geometry mismatch for {name}")
        if fusion == "union":
            mask = mask_a | mask_b
        elif fusion == "intersection":
            mask = mask_a & mask_b

    voxel_volume = float(abs(np.linalg.det(image_a.affine[:3, :3])))
    mask = filter_components(mask, voxel_volume, min_mm3, connectivity)
    header = image_a.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), image_a.affine, header), output / name)
    return name, int(mask.sum())


def process_predictions(
    names: list[str],
    pred_a: Path,
    pred_b: Path | None,
    fusion: str,
    min_mm3: float,
    output: Path,
    workers: int,
    connectivity: int = 6,
) -> list[tuple[str, int | None]]:
    operation = partial(
        process_prediction,
        pred_a=pred_a,
        pred_b=pred_b,
        fusion=fusion,
        min_mm3=min_mm3,
        output=output,
        connectivity=connectivity,
    )
    if workers == 1:
        return [operation(name) for name in names]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # executor.map preserves input order, so logs and failures remain
        # deterministic even though independent cases execute concurrently.
        return list(executor.map(operation, names))


def main() -> None:
    args = parse_args()
    if args.pred_b is None and args.fusion != "a":
        raise ValueError(f"--fusion {args.fusion} requires --pred-b")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    names = sorted(path.name for path in args.pred_a.glob("*.nii.gz"))
    if not names:
        raise ValueError(f"no predictions in {args.pred_a}")
    args.output.mkdir(parents=True, exist_ok=True)

    results = process_predictions(
        names, args.pred_a, args.pred_b, args.fusion, args.min_mm3,
        args.output, args.workers, args.connectivity,
    )
    for index, (name, voxels) in enumerate(results, start=1):
        message = "copied unchanged" if voxels is None else f"{voxels:,} voxels"
        print(f"[{index:03d}/{len(names):03d}] {name}: {message}", flush=True)


if __name__ == "__main__":
    main()
