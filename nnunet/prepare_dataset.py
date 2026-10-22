#!/usr/bin/env python3
"""Assemble the supplied pan-cancer CT training data into nnU-Net v2 format.

No image data is copied. The generated Dataset501_PanCancerCT tree contains
symlinks to the downloaded source files. Challenge validation cases are kept
strictly inference-only; fold 0 is a deterministic source-stratified split made
only from the provided training cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


DATASET_ID = 501
DATASET_NAME = "Dataset501_PanCancerCT"
ENDING = ".nii.gz"


def image_case(path: Path, allow_missing_channel: bool = False) -> str:
    suffix = "_0000" + ENDING
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    if allow_missing_channel and path.name.endswith(ENDING):
        return path.name[: -len(ENDING)]
    raise ValueError(f"Image does not follow nnU-Net channel naming: {path}")


def label_case(path: Path) -> str:
    if not path.name.endswith(ENDING):
        raise ValueError(f"Label does not end in {ENDING}: {path}")
    return path.name[: -len(ENDING)]


def collect(
    directories: Iterable[Path],
    kind: str,
    allow_missing_channel: bool = False,
) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing input directory: {directory}")
        for path in sorted(directory.glob(f"*{ENDING}")):
            case = (
                image_case(path, allow_missing_channel)
                if kind == "image"
                else label_case(path)
            )
            if case in found:
                raise RuntimeError(
                    f"Duplicate {kind} case {case}: {found[case]} and {path}"
                )
            found[case] = path.resolve()
    return found


def assert_paired(images: dict[str, Path], labels: dict[str, Path], name: str) -> None:
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        details = []
        if missing_labels:
            details.append(f"missing labels: {missing_labels[:10]}")
        if missing_images:
            details.append(f"missing images: {missing_images[:10]}")
        raise RuntimeError(f"Unpaired {name} data ({'; '.join(details)})")


def link(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        raise FileExistsError(f"Wrong existing symlink: {destination}")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(source)


def reconcile_managed_directory(directory: Path, expected: set[str]) -> None:
    unexpected = sorted(p.name for p in directory.iterdir() if p.name not in expected)
    for name in unexpected:
        path = directory / name
        if not path.is_symlink():
            raise RuntimeError(f"Refusing to remove unmanaged path: {path}")
        path.unlink()


def source_group(case: str) -> str:
    """Recover the source-dataset prefix from flattened case identifiers."""
    # Every flattened case is SOURCE_CASEID. CASEID is not always numeric
    # (some datasets use hashes or R01-xxx), so split at the final underscore.
    return case.rsplit("_", 1)[0] if "_" in case else case


def make_internal_split(cases: Iterable[str], validation_fraction: float = 0.05) -> dict:
    groups: dict[str, list[str]] = {}
    for case in cases:
        groups.setdefault(source_group(case), []).append(case)

    train: list[str] = []
    val: list[str] = []
    for group_cases in groups.values():
        ranked = sorted(
            group_cases,
            key=lambda case: hashlib.sha256(f"501:{case}".encode()).digest(),
        )
        # Preserve representation from each source with at least two cases.
        n_val = max(1, round(len(ranked) * validation_fraction)) if len(ranked) > 1 else 0
        val.extend(ranked[:n_val])
        train.extend(ranked[n_val:])
    return {"train": sorted(train), "val": sorted(val)}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Workspace containing train_label and validation (default: script parent)",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()

    training_images = collect(
        [
            root / "train_label/imagesTr_part1",
            root / "train_label/imagesTr_part2",
        ],
        "image",
    )
    training_labels = collect(
        [
            root / "train_label/labelsTr_part1",
            root / "train_label/labelsTr_part2",
        ],
        "label",
    )
    public_images = collect([root / "validation/Validation-Public-Images"], "image")
    public_labels = collect([root / "validation/Validation-Public-Labels"], "label")
    hidden_images = collect([root / "validation/Validation-Hidden-Images"], "image")
    # A subset of the provided healthy files lacks nnU-Net's _0000 suffix. The
    # generated links normalize those filenames without touching source data.
    healthy_images = collect(
        [root / "validation/HealthyImages-noLesion"],
        "image",
        allow_missing_channel=True,
    )

    assert_paired(training_images, training_labels, "training")
    assert_paired(public_images, public_labels, "public validation")

    groups = {
        "training": set(training_images),
        "public_validation": set(public_images),
        "codabench_development_hidden_validation": set(hidden_images),
        "healthy_test": set(healthy_images),
    }
    group_items = list(groups.items())
    for index, (left_name, left) in enumerate(group_items):
        for right_name, right in group_items[index + 1 :]:
            overlap = sorted(left & right)
            if overlap:
                raise RuntimeError(
                    f"Case IDs overlap between {left_name} and {right_name}: {overlap[:10]}"
                )

    all_training_images = training_images
    all_training_labels = training_labels
    # Public validation is inference-only. Its labels stay outside nnUNet_raw so
    # neither fingerprinting nor training can consume them accidentally.
    all_test_images = {**public_images, **hidden_images, **healthy_images}

    output = root / "nnUNet_raw" / DATASET_NAME
    images_tr = output / "imagesTr"
    labels_tr = output / "labelsTr"
    images_ts = output / "imagesTs"
    for directory in (images_tr, labels_tr, images_ts):
        directory.mkdir(parents=True, exist_ok=True)

    expected_images_tr = {f"{case}_0000{ENDING}" for case in all_training_images}
    expected_labels_tr = {f"{case}{ENDING}" for case in all_training_labels}
    expected_images_ts = {f"{case}_0000{ENDING}" for case in all_test_images}
    reconcile_managed_directory(images_tr, expected_images_tr)
    reconcile_managed_directory(labels_tr, expected_labels_tr)
    reconcile_managed_directory(images_ts, expected_images_ts)

    for case, source in all_training_images.items():
        link(source, images_tr / f"{case}_0000{ENDING}")
    for case, source in all_training_labels.items():
        link(source, labels_tr / f"{case}{ENDING}")
    for case, source in all_test_images.items():
        link(source, images_ts / f"{case}_0000{ENDING}")

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(all_training_images),
        "file_ending": ENDING,
        "name": "PanCancerCT",
        "description": (
            "Pan-cancer primary-lesion segmentation training data. Masks are "
            "partial and may omit additional lesions. Challenge validation is "
            "strictly inference-only."
        ),
    }
    split = [make_internal_split(training_images)]
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "counts": {
            "train": len(training_images),
            "public_validation": len(public_images),
            "codabench_development_hidden_validation": len(hidden_images),
            "healthy_test": len(healthy_images),
        },
        "internal_fold_0": {
            "train": len(split[0]["train"]),
            "val": len(split[0]["val"]),
            "method": "deterministic 95/5 source-stratified split",
        },
        "note": (
            "imagesTs contains public validation, hidden validation, and "
            "healthy/no-lesion scans. Public labels are evaluation-only."
        ),
    }
    write_json(output / "dataset.json", dataset_json)
    # The ResEnc experiment planner copies this into nnUNet_preprocessed.
    write_json(output / "splits_final.json", split)
    write_json(output / "source_manifest.json", manifest)

    print(f"Prepared {output}")
    print(f"  training:          {len(training_images):6d}")
    print(f"  public validation: {len(public_images):6d}")
    print(f"  Codabench dev/hidden val:{len(hidden_images):6d}")
    print(f"  healthy test:      {len(healthy_images):6d}")
    print(f"  internal fold train:{len(split[0]['train']):6d}")
    print(f"  internal fold val:  {len(split[0]['val']):6d}")
    print(f"  imagesTr/labelsTr: {len(all_training_images):6d}")
    print(f"  imagesTs:          {len(all_test_images):6d}")


if __name__ == "__main__":
    main()
