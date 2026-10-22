#!/usr/bin/env python3
"""Validate hidden-validation segmentations and create a Codabench ZIP."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np


INPUT_SUFFIX = "_0000.nii.gz"
OUTPUT_SUFFIX = ".nii.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate binary NIfTI predictions against the hidden-validation "
            "images and package them at the root of a Codabench ZIP."
        )
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Folder containing FLARE-Val_*_0000.nii.gz images.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        action="append",
        required=True,
        help="Prediction folder. Repeat for outputs split across GPUs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .zip file.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=100,
        help="Required number of cases (default: 100).",
    )
    return parser.parse_args()


def prediction_name(image: Path) -> str:
    if not image.name.endswith(INPUT_SUFFIX):
        raise ValueError(f"Unexpected image name: {image.name}")
    return image.name[: -len(INPUT_SUFFIX)] + OUTPUT_SUFFIX


def fail(message: str) -> None:
    raise RuntimeError(message)


def collect_predictions(pred_dirs: list[Path]) -> dict[str, Path]:
    predictions: dict[str, Path] = {}
    for pred_dir in pred_dirs:
        if not pred_dir.is_dir():
            fail(f"Prediction directory does not exist: {pred_dir}")
        for path in sorted(pred_dir.glob("*.nii.gz")):
            if path.name in predictions:
                fail(
                    f"Duplicate prediction {path.name} in {predictions[path.name].parent} "
                    f"and {path.parent}"
                )
            predictions[path.name] = path
    return predictions


def validate_case(image_path: Path, prediction_path: Path) -> int:
    try:
        image = nib.load(str(image_path))
        prediction = nib.load(str(prediction_path))
    except Exception as exc:
        fail(f"Could not read {prediction_path}: {exc}")

    if prediction.shape != image.shape:
        fail(
            f"Shape mismatch for {prediction_path.name}: prediction "
            f"{prediction.shape}, image {image.shape}"
        )
    if not np.allclose(prediction.affine, image.affine, rtol=1e-5, atol=1e-4):
        fail(f"Affine mismatch for {prediction_path.name}")

    data = np.asanyarray(prediction.dataobj)
    if not np.issubdtype(data.dtype, np.number) or not np.all(np.isfinite(data)):
        fail(f"Non-numeric or non-finite values in {prediction_path.name}")
    labels = np.unique(data)
    if not np.isin(labels, (0, 1)).all():
        shown = labels[:20].tolist()
        fail(
            f"{prediction_path.name} is not binary; found labels {shown}"
            + (" ..." if labels.size > 20 else "")
        )
    return int(np.count_nonzero(data))


def write_zip(output: Path, ordered_predictions: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for path in ordered_predictions:
                # arcname deliberately has no directory: Codabench expects a flat ZIP.
                archive.write(path, arcname=path.name)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_zip(output: Path, expected_names: list[str]) -> None:
    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        if names != expected_names:
            fail("Internal error: ZIP member list does not match expected flat layout")
        bad_member = archive.testzip()
        if bad_member is not None:
            fail(f"ZIP integrity check failed at {bad_member}")


def main() -> int:
    args = parse_args()
    if not args.images.is_dir():
        fail(f"Image directory does not exist: {args.images}")
    if args.output.suffix.lower() != ".zip":
        fail(f"Output must end in .zip: {args.output}")

    images = sorted(args.images.glob(f"*{INPUT_SUFFIX}"))
    if len(images) != args.expected_count:
        fail(
            f"Expected {args.expected_count} hidden-validation images, found "
            f"{len(images)} in {args.images}"
        )

    expected_names = [prediction_name(path) for path in images]
    expected_set = set(expected_names)
    predictions = collect_predictions(args.pred_dir)
    actual_set = set(predictions)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing ({len(missing)}): {', '.join(missing[:10])}")
        if extra:
            details.append(f"extra ({len(extra)}): {', '.join(extra[:10])}")
        fail("Prediction set is invalid; " + "; ".join(details))

    ordered_predictions: list[Path] = []
    empty_cases: list[str] = []
    total_foreground = 0
    for index, (image_path, name) in enumerate(zip(images, expected_names), start=1):
        prediction_path = predictions[name]
        foreground = validate_case(image_path, prediction_path)
        total_foreground += foreground
        if foreground == 0:
            empty_cases.append(name)
        ordered_predictions.append(prediction_path)
        print(f"[{index:03d}/{len(images):03d}] valid {name}: {foreground:,} foreground voxels")

    write_zip(args.output, ordered_predictions)
    verify_zip(args.output, expected_names)

    size_mib = args.output.stat().st_size / (1024**2)
    print(f"\nCodabench submission ready: {args.output}")
    print(f"Cases: {len(ordered_predictions)}")
    print(f"Archive size: {size_mib:.1f} MiB")
    print(f"Total foreground voxels: {total_foreground:,}")
    if empty_cases:
        print(
            f"Warning: {len(empty_cases)} predictions are empty: "
            + ", ".join(empty_cases),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
