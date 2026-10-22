#!/usr/bin/env python3
"""Lossless geometry adapter between challenge NIfTI and BiomedParse NPZ.

BiomedParse v2 expects a z-y-x NumPy volume in [0, 255] plus textual class
prompts.  The model output is converted back to binary uint8 NIfTI while
copying the source CT's origin, spacing, and direction exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def case_name(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def prediction_name(path: Path) -> str:
    """Return the flat challenge mask name, removing nnU-Net channel suffix."""
    stem = case_name(path)
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return f"{stem}.nii.gz"


def window_to_uint8(image: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"invalid intensity window [{lower}, {upper}]")
    scaled = (np.clip(image.astype(np.float32), lower, upper) - lower) * (255.0 / (upper - lower))
    return np.rint(scaled).astype(np.uint8)


def prepare_case(source: Path, destination: Path, prompts: list[str], lower: float, upper: float) -> None:
    image = sitk.ReadImage(str(source))
    volume = sitk.GetArrayFromImage(image)
    text_prompts = {str(index + 1): prompt for index, prompt in enumerate(prompts)}
    text_prompts["instance_label"] = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        imgs=window_to_uint8(volume, lower, upper),
        spacing=np.asarray(image.GetSpacing(), dtype=np.float32)[::-1],
        text_prompts=text_prompts,
    )


def convert_case(
    source_ct: Path,
    prediction: Path,
    destination: Path,
    segmentation_key: str = "segs",
) -> None:
    reference = sitk.ReadImage(str(source_ct))
    with np.load(prediction, allow_pickle=False) as archive:
        if segmentation_key not in archive:
            raise KeyError(f"{prediction}: missing {segmentation_key!r}")
        segmentation = np.asarray(archive[segmentation_key])
    expected = tuple(reversed(reference.GetSize()))
    if segmentation.shape != expected:
        raise ValueError(f"{prediction}: shape {segmentation.shape}, expected {expected}")
    output = sitk.GetImageFromArray((segmentation > 0).astype(np.uint8))
    output.CopyInformation(reference)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(destination), True)


def nifti_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.nii.gz"))


def prepare(args: argparse.Namespace) -> None:
    sources = nifti_files(args.images)
    if not sources:
        raise FileNotFoundError(f"no NIfTI files in {args.images}")
    if not args.prompt:
        raise ValueError("at least one --prompt is required")
    args.output.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources, 1):
        destination = args.output / f"{case_name(source)}.npz"
        if args.continue_prediction and destination.exists():
            continue
        prepare_case(source, destination, args.prompt, args.a_min, args.a_max)
        print(f"prepared {index}/{len(sources)} {source.name}", flush=True)


def convert(args: argparse.Namespace) -> None:
    sources = nifti_files(args.images)
    if not sources:
        raise FileNotFoundError(f"no NIfTI files in {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources, 1):
        prediction = args.predictions / f"{case_name(source)}.npz"
        if not prediction.exists():
            raise FileNotFoundError(prediction)
        destination = args.output / prediction_name(source)
        if args.continue_prediction and destination.exists():
            continue
        convert_case(
            source, prediction, destination, segmentation_key=args.segmentation_key
        )
        print(f"converted {index}/{len(sources)} {source.name}", flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--images", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--prompt", action="append", default=[])
    prep.add_argument("--a-min", type=float, default=-160.0)
    prep.add_argument("--a-max", type=float, default=240.0)
    prep.add_argument("--continue-prediction", action="store_true")
    prep.set_defaults(function=prepare)
    conv = commands.add_parser("convert")
    conv.add_argument("--images", type=Path, required=True)
    conv.add_argument("--predictions", type=Path, required=True)
    conv.add_argument("--output", type=Path, required=True)
    conv.add_argument("--segmentation-key", default="segs")
    conv.add_argument("--continue-prediction", action="store_true")
    conv.set_defaults(function=convert)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
