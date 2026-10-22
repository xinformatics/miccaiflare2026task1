"""Filename helpers shared by challenge and auxiliary GF inference sets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def prediction_name(image_name: str) -> str:
    """Map an input CT filename to its flat segmentation filename.

    Challenge images use the nnU-Net channel suffix ``_0000``.  Auxiliary
    healthy scans do not consistently use that convention, so only remove the
    suffix when it is actually present.
    """

    if image_name.endswith(".nii.gz"):
        stem = image_name[:-7]
    elif image_name.endswith(".nii"):
        stem = image_name[:-4]
    else:
        raise ValueError(f"not a NIfTI filename: {image_name}")
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return f"{stem}.nii.gz"


def incomplete_inputs(
    inputs: Iterable[Path], prediction_directories: Iterable[Path]
) -> list[Path]:
    """Keep inputs whose final mask is absent from at least one output root."""

    directories = [Path(directory) for directory in prediction_directories]
    if not directories:
        raise ValueError("at least one prediction directory is required")
    return [
        Path(image)
        for image in inputs
        if not all(
            (directory / prediction_name(Path(image).name)).is_file()
            for directory in directories
        )
    ]
