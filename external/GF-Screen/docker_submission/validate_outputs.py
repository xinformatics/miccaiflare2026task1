#!/usr/bin/env python3
"""Validate challenge output names, geometry, datatype, and component filtering."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def output_name(name: str) -> str:
    return name.removesuffix("_0000.nii.gz") + ".nii.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-mm3", type=float, default=1000.0)
    args = parser.parse_args()

    inputs = sorted(path for path in args.input.rglob("*.nii.gz") if path.is_file())
    outputs = sorted(path for path in args.output.glob("*.nii.gz") if path.is_file())
    expected = {output_name(path.name) for path in inputs}
    actual = {path.name for path in outputs}
    if expected != actual:
        raise RuntimeError(f"Output mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")

    for source in inputs:
        prediction_path = args.output / output_name(source.name)
        image = nib.load(str(source))
        prediction = nib.load(str(prediction_path))
        mask = np.asanyarray(prediction.dataobj)
        values = np.unique(mask)
        if image.shape != prediction.shape or not np.allclose(image.affine, prediction.affine):
            raise RuntimeError(f"Geometry mismatch: {prediction_path.name}")
        if not np.isin(values, (0, 1)).all():
            raise RuntimeError(f"Nonbinary output: {prediction_path.name}: {values.tolist()}")
        labels, count = ndimage.label(mask.astype(bool, copy=False))
        if count:
            sizes = np.bincount(labels.ravel())[1:] * abs(np.linalg.det(prediction.affine[:3, :3]))
            if sizes.min() + 1e-3 < args.minimum_mm3:
                raise RuntimeError(f"Small component in {prediction_path.name}: {sizes.min():.3f} mm3")
        print(f"OK {prediction_path.name}: shape={prediction.shape}, values={values.tolist()}")

    print(f"Validated {len(outputs)} outputs")


if __name__ == "__main__":
    main()
