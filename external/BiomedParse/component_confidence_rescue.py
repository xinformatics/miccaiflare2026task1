#!/usr/bin/env python3
"""Consensus rescue for GF components rejected by binary prompt confirmation.

The rule first keeps every GF component overlapping the OR of two BiomedParse
prompt masks. An otherwise rejected component is restored only when it is in a
bounded physical-size range, survives a higher GF Focus threshold, and its CT
slices receive sufficient BiomedParse ``Tumor`` existence confidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def configurations() -> list[dict[str, object]]:
    """Return the finite validation grid declared before exact scoring."""

    common: dict[str, object] = {
        "kind": "confirm_confidence_rescue",
        "rescue_min_mm3": 1000.0,
        "rescue_max_mm3": 10000.0,
    }
    return [
        {**common, "existence_stat": "mean", "existence_threshold": 0.10},
        {**common, "existence_stat": "mean", "existence_threshold": 0.30},
        {**common, "existence_stat": "max", "existence_threshold": 0.10},
        {**common, "existence_stat": "max", "existence_threshold": 0.30},
    ]


def config_tag(config: dict[str, object]) -> str:
    threshold = int(round(100 * float(config["existence_threshold"])))
    minimum = int(float(config["rescue_min_mm3"]))
    maximum = int(float(config["rescue_max_mm3"]))
    return f"focus075_tumor_{config['existence_stat']}_e{threshold:03d}_mm{minimum}_{maximum}"


def apply_confidence_confirmation(
    base: np.ndarray,
    prompt_support: np.ndarray,
    high_focus_support: np.ndarray,
    tumor_existence: np.ndarray,
    voxel_mm3: float,
    config: dict[str, object],
) -> np.ndarray:
    base = np.asarray(base, dtype=bool)
    prompt_support = np.asarray(prompt_support, dtype=bool)
    high_focus_support = np.asarray(high_focus_support, dtype=bool)
    tumor_existence = np.asarray(tumor_existence, dtype=np.float32)
    if base.shape != prompt_support.shape or base.shape != high_focus_support.shape:
        raise ValueError("base and support masks must have identical shapes")
    if tumor_existence.shape != (base.shape[0],):
        raise ValueError(
            f"existence shape {tumor_existence.shape} does not match z depth {base.shape[0]}"
        )
    statistic = str(config["existence_stat"])
    if statistic not in {"mean", "max"}:
        raise ValueError(f"unknown existence statistic: {statistic}")

    component_map, count = ndimage.label(
        base, structure=np.ones((3, 3, 3), dtype=bool)
    )
    sizes = np.bincount(component_map.ravel(), minlength=count + 1) * float(voxel_mm3)
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(component_map[base & prompt_support])] = True
    keep[0] = False
    minimum = float(config["rescue_min_mm3"])
    maximum = float(config["rescue_max_mm3"])
    threshold = float(config["existence_threshold"])
    eligible_ids = np.flatnonzero((~keep) & (sizes >= minimum) & (sizes <= maximum))
    for component_id in eligible_ids:
        if component_id == 0:
            continue
        component = component_map == component_id
        if not np.any(component & high_focus_support):
            continue
        z = np.flatnonzero(component.any(axis=(1, 2)))
        score = (
            float(tumor_existence[z].mean())
            if statistic == "mean"
            else float(tumor_existence[z].max())
        )
        if score >= threshold:
            keep[component_id] = True
    keep[0] = False
    return keep[component_map]


def read_mask(path: Path) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image) > 0, image


def same_geometry(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing(), atol=1e-5)
        and np.allclose(left.GetOrigin(), right.GetOrigin(), atol=1e-3)
        and np.allclose(left.GetDirection(), right.GetDirection(), atol=1e-5)
    )


def read_existence(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["object_existence"], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError(f"{path}: expected one prompt, got {values.shape}")
    return values[0]


def write_mask(mask: np.ndarray, reference: sitk.Image, path: Path) -> None:
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), True)


def process_case(
    base_path: Path,
    prompt_dir: Path,
    high_focus_dir: Path,
    tumor_npz_dir: Path,
    outputs: dict[str, Path],
    configs: list[dict[str, object]],
) -> str:
    base, reference = read_mask(base_path)
    prompt, prompt_image = read_mask(prompt_dir / base_path.name)
    high_focus, high_focus_image = read_mask(high_focus_dir / base_path.name)
    if not same_geometry(reference, prompt_image) or not same_geometry(reference, high_focus_image):
        raise ValueError(f"geometry mismatch for {base_path.name}")
    case = base_path.name.removesuffix(".nii.gz")
    tumor_existence = read_existence(tumor_npz_dir / f"{case}_0000.npz")
    voxel_mm3 = float(np.prod(reference.GetSpacing()))
    for config in configs:
        tag = config_tag(config)
        prediction = apply_confidence_confirmation(
            base, prompt, high_focus, tumor_existence, voxel_mm3, config
        )
        write_mask(prediction, reference, outputs[tag] / base_path.name)
    return base_path.name


def process_case_tuple(arguments: tuple[object, ...]) -> str:
    return process_case(*arguments)  # type: ignore[arg-type]


def materialize_grid(args: argparse.Namespace) -> None:
    configs = configurations()
    outputs = {config_tag(config): args.output / config_tag(config) for config in configs}
    args.output.mkdir(parents=True, exist_ok=True)
    for config in configs:
        tag = config_tag(config)
        outputs[tag].mkdir(parents=True, exist_ok=True)
        (args.output / f"{tag}.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
    files = sorted(args.base.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"no masks in {args.base}")
    jobs = [
        (path, args.prompt_support, args.high_focus_support, args.tumor_npz, outputs, configs)
        for path in files
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, name in enumerate(executor.map(process_case_tuple, jobs), 1):
            print(f"[{index:03d}/{len(files):03d}] {name}", flush=True)
    manifest = args.output / "candidates.tsv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["tag", "config", "predictions"])
        for config in configs:
            tag = config_tag(config)
            writer.writerow([tag, args.output / f"{tag}.json", outputs[tag]])
    print(manifest.read_text(), end="")


def materialize_one(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    tag = config_tag(config)
    files = sorted(args.base.glob("*.nii.gz"))
    outputs = {tag: args.output}
    for index, path in enumerate(files, 1):
        process_case(
            path, args.prompt_support, args.high_focus_support,
            args.tumor_npz, outputs, [config],
        )
        print(f"[{index:03d}/{len(files):03d}] {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize-grid", "materialize"):
        command = commands.add_parser(name)
        command.add_argument("--base", type=Path, required=True)
        command.add_argument("--prompt-support", type=Path, required=True)
        command.add_argument("--high-focus-support", type=Path, required=True)
        command.add_argument("--tumor-npz", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    grid = commands.choices["materialize-grid"]
    grid.add_argument("--workers", type=int, default=4)
    single = commands.choices["materialize"]
    single.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "materialize-grid":
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        materialize_grid(args)
    else:
        materialize_one(args)


if __name__ == "__main__":
    main()
