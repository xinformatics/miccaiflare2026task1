#!/usr/bin/env python3
"""Screen conservative new-lesion additions to an immutable segmentation.

Only connected components from ``candidate`` that have no voxel overlap with
the immutable ``base`` are eligible.  This prevents the auxiliary model from
changing boundaries of lesions already accepted by the deployed method.  The
remaining components are gated by physical volume and distance from ``base``.
Public labels are used only for scoring the small, predeclared grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "handoff_bundle" / "code"))
from evaluate import score_case  # noqa: E402


# Stages 1-2 established that eligible components below 500 mm3 were absent,
# distance gates through 10 mm were identical, and 2,000 mm3 was the strongest
# upper-volume guard. Stage 3 fixes those axes and tests only how much of a new
# GF component must be independently supported by the prompt mask.
MIN_MM3 = (0.0,)
MAX_MM3 = (2000.0,)
MIN_DISTANCE_MM = (0.0,)
MIN_SUPPORT_FRACTION = (0.25,)
MIN_SUPPORT_MM3 = (100.0,)
SUPPORT_DILATION_MM = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, math.inf)


@dataclass(frozen=True)
class Config:
    min_mm3: float
    max_mm3: float
    min_distance_mm: float
    min_support_fraction: float
    min_support_mm3: float
    support_dilation_mm: float

    @property
    def tag(self) -> str:
        upper = "inf" if math.isinf(self.max_mm3) else str(int(self.max_mm3))
        dilation = "inf" if math.isinf(self.support_dilation_mm) else f"{self.support_dilation_mm:g}"
        return (
            f"new_min{int(self.min_mm3)}_max{upper}"
            f"_dist{int(self.min_distance_mm)}"
            f"_sf{self.min_support_fraction:g}_sm{int(self.min_support_mm3)}"
            f"_sd{dilation}"
        )


CONFIGS = tuple(
    Config(minimum, maximum, distance, fraction, support_mm3, dilation)
    for minimum in MIN_MM3
    for maximum in MAX_MM3
    for distance in MIN_DISTANCE_MM
    for fraction in MIN_SUPPORT_FRACTION
    for support_mm3 in MIN_SUPPORT_MM3
    for dilation in SUPPORT_DILATION_MM
)


def load_bool(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(bool), image


def validate_geometry(name: str, reference: nib.Nifti1Image, other: nib.Nifti1Image) -> None:
    if reference.shape != other.shape or not np.allclose(
        reference.affine, other.affine, atol=1e-3
    ):
        raise ValueError(f"geometry mismatch: {name}")


def eligible_component_stats(
    base: np.ndarray,
    candidate: np.ndarray,
    support: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = cc3d.connected_components(candidate.astype(np.uint8), connectivity=26)
    counts = np.bincount(labels.ravel())
    overlap = np.bincount(labels[base], minlength=len(counts))
    voxel_mm3 = float(np.prod(spacing))
    volumes = counts.astype(np.float64) * voxel_mm3
    support_counts = np.bincount(labels[support], minlength=len(counts))
    support_mm3 = support_counts.astype(np.float64) * voxel_mm3
    support_fraction = np.divide(
        support_counts,
        counts,
        out=np.zeros_like(volumes),
        where=counts > 0,
    )
    if np.any(base):
        distance = distance_transform_edt(~base, sampling=spacing)
        minimum_distance = np.full(len(counts), np.inf, dtype=np.float64)
        np.minimum.at(minimum_distance, labels.ravel(), distance.ravel())
    else:
        minimum_distance = np.full(len(counts), np.inf, dtype=np.float64)
    eligible = overlap == 0
    eligible[0] = False
    return (
        labels,
        volumes,
        np.where(eligible, minimum_distance, -np.inf),
        support_fraction,
        support_mm3,
    )


def apply_config(
    base: np.ndarray,
    labels: np.ndarray,
    volumes: np.ndarray,
    eligible_distance: np.ndarray,
    support_fraction: np.ndarray,
    support_mm3: np.ndarray,
    support_distance: np.ndarray,
    config: Config,
) -> np.ndarray:
    keep = eligible_distance >= config.min_distance_mm
    keep &= volumes >= config.min_mm3
    keep &= volumes <= config.max_mm3
    keep &= support_fraction >= config.min_support_fraction
    keep &= support_mm3 >= config.min_support_mm3
    keep[0] = False
    addition = keep[labels]
    if not math.isinf(config.support_dilation_mm):
        addition &= support_distance <= config.support_dilation_mm
    return base | addition


def selected_labels(
    volumes: np.ndarray,
    eligible_distance: np.ndarray,
    support_fraction: np.ndarray,
    support_mm3: np.ndarray,
    config: Config,
) -> np.ndarray:
    keep = eligible_distance >= config.min_distance_mm
    keep &= volumes >= config.min_mm3
    keep &= volumes <= config.max_mm3
    keep &= support_fraction >= config.min_support_fraction
    keep &= support_mm3 >= config.min_support_mm3
    keep[0] = False
    return np.flatnonzero(keep)


def score_one(item: tuple[str, str, str, str, str]) -> tuple[str, list[tuple[str, float, float, int]]]:
    name, base_path, candidate_path, support_path, target_path = item
    base, base_image = load_bool(Path(base_path))
    candidate, candidate_image = load_bool(Path(candidate_path))
    support, support_image = load_bool(Path(support_path))
    target, target_image = load_bool(Path(target_path))
    validate_geometry(f"candidate/{name}", base_image, candidate_image)
    validate_geometry(f"support/{name}", base_image, support_image)
    validate_geometry(f"target/{name}", base_image, target_image)
    spacing = tuple(float(value) for value in base_image.header.get_zooms()[:3])
    support_distance = distance_transform_edt(~support, sampling=spacing)
    labels, volumes, eligible_distance, support_fraction, support_mm3 = eligible_component_stats(
        base, candidate, support, spacing
    )
    values = []
    # Sparse candidate sets often make many grid configurations identical.
    # Cache exact metric results by selected component IDs so expensive NSD
    # surface distances are computed only once per distinct prediction.
    score_cache: dict[tuple[tuple[int, ...], float], tuple[float, float, int]] = {}
    for config in CONFIGS:
        component_key = tuple(int(value) for value in selected_labels(
            volumes, eligible_distance, support_fraction, support_mm3, config
        ))
        key = (component_key, float(config.support_dilation_mm))
        if key not in score_cache:
            keep = np.zeros(len(volumes), dtype=bool)
            keep[list(component_key)] = True
            addition = keep[labels]
            if not math.isinf(config.support_dilation_mm):
                addition &= support_distance <= config.support_dilation_mm
            prediction = base | addition
            dsc, nsd = score_case(
                target.astype(np.uint8), prediction.astype(np.uint8), spacing
            )
            score_cache[key] = (float(dsc), float(nsd), int(prediction.sum()))
        dsc, nsd, foreground = score_cache[key]
        values.append((config.tag, dsc, nsd, foreground))
    return name.removesuffix(".nii.gz"), values


def common_names(directory: Path) -> list[str]:
    names = sorted(path.name for path in directory.glob("*.nii.gz"))
    if not names:
        raise FileNotFoundError(f"no NIfTI masks in {directory}")
    return names


def validate_names(names: list[str], directories: tuple[Path, ...]) -> None:
    expected = set(names)
    for directory in directories:
        observed = set(common_names(directory))
        if observed != expected:
            raise ValueError(
                f"case mismatch in {directory}: "
                f"missing={sorted(expected-observed)[:3]} "
                f"extra={sorted(observed-expected)[:3]}"
            )


def screen(args: argparse.Namespace) -> None:
    names = common_names(args.gt)
    validate_names(names, (args.base, args.candidate, args.support))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {config.tag: [] for config in CONFIGS}
    items = [
        (
            name,
            str(args.base / name),
            str(args.candidate / name),
            str(args.support / name),
            str(args.gt / name),
        )
        for name in names
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (case, values) in enumerate(executor.map(score_one, items), 1):
            for tag, dsc, nsd, foreground in values:
                rows[tag].append((case, dsc, nsd, foreground))
            print(f"[{index:03d}/{len(names):03d}] {case}", flush=True)

    summary = []
    config_by_tag = {config.tag: config for config in CONFIGS}
    metrics_dir = args.output / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    for tag, case_rows in rows.items():
        metrics = metrics_dir / f"{tag}.metrics.csv"
        with metrics.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD", "pred_fg_voxels"))
            for row in case_rows:
                writer.writerow((row[0], round(row[1], 4), round(row[2], 4), row[3]))
        mean_dsc = float(np.mean([row[1] for row in case_rows]))
        mean_nsd = float(np.mean([row[2] for row in case_rows]))
        config = config_by_tag[tag]
        summary.append(
            {
                "tag": tag,
                "min_mm3": config.min_mm3,
                "max_mm3": config.max_mm3,
                "min_distance_mm": config.min_distance_mm,
                "min_support_fraction": config.min_support_fraction,
                "min_support_mm3": config.min_support_mm3,
                "support_dilation_mm": config.support_dilation_mm,
                "DSC": mean_dsc,
                "NSD": mean_nsd,
                "combined": (mean_dsc + mean_nsd) / 2,
                "metrics": str(metrics.resolve()),
            }
        )
    summary.sort(key=lambda row: (-row["combined"], -row["DSC"], row["tag"]))
    with (args.output / "summary.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "grid.json").write_text(
        json.dumps([config.__dict__ | {"tag": config.tag} for config in CONFIGS], indent=2)
        + "\n"
    )
    for row in summary[:10]:
        print(
            f"{row['tag']} DSC={row['DSC']:.6f} NSD={row['NSD']:.6f} "
            f"combined={row['combined']:.6f}"
        )


def materialize_one(item: tuple[str, str, str, str, str, Config]) -> str:
    name, base_path, candidate_path, support_path, output_path, config = item
    base, base_image = load_bool(Path(base_path))
    candidate, candidate_image = load_bool(Path(candidate_path))
    support, support_image = load_bool(Path(support_path))
    validate_geometry(f"candidate/{name}", base_image, candidate_image)
    validate_geometry(f"support/{name}", base_image, support_image)
    spacing = tuple(float(value) for value in base_image.header.get_zooms()[:3])
    # Infinite dilation accepts every voxel of an otherwise supported
    # component, so the full-volume EDT is mathematically unused.  Skipping it
    # avoids several seconds and multiple GiB of transient memory per large CT
    # during deployed materialization while producing the identical mask.
    support_distance = (
        np.zeros((), dtype=np.float32)
        if math.isinf(config.support_dilation_mm)
        else distance_transform_edt(~support, sampling=spacing)
    )
    labels, volumes, eligible_distance, support_fraction, support_mm3 = eligible_component_stats(
        base, candidate, support, spacing
    )
    prediction = apply_config(
        base,
        labels,
        volumes,
        eligible_distance,
        support_fraction,
        support_mm3,
        support_distance,
        config,
    )
    header = base_image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(prediction.astype(np.uint8), base_image.affine, header), output_path)
    return name


def materialize(args: argparse.Namespace) -> None:
    names = common_names(args.base)
    validate_names(names, (args.candidate, args.support))
    config = Config(
        args.min_mm3,
        args.max_mm3,
        args.min_distance_mm,
        args.min_support_fraction,
        args.min_support_mm3,
        args.support_dilation_mm,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    items = [
        (
            name,
            str(args.base / name),
            str(args.candidate / name),
            str(args.support / name),
            str(args.output / name),
            config,
        )
        for name in names
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, name in enumerate(executor.map(materialize_one, items), 1):
            print(f"[{index:03d}/{len(names):03d}] {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen_parser = subparsers.add_parser("screen")
    screen_parser.add_argument("--base", type=Path, required=True)
    screen_parser.add_argument("--candidate", type=Path, required=True)
    screen_parser.add_argument("--support", type=Path, required=True)
    screen_parser.add_argument("--gt", type=Path, required=True)
    screen_parser.add_argument("--output", type=Path, required=True)
    screen_parser.add_argument("--workers", type=int, default=8)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--base", type=Path, required=True)
    materialize_parser.add_argument("--candidate", type=Path, required=True)
    materialize_parser.add_argument("--support", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--min-mm3", type=float, required=True)
    materialize_parser.add_argument("--max-mm3", type=float, required=True)
    materialize_parser.add_argument("--min-distance-mm", type=float, required=True)
    materialize_parser.add_argument("--min-support-fraction", type=float, required=True)
    materialize_parser.add_argument("--min-support-mm3", type=float, required=True)
    materialize_parser.add_argument("--support-dilation-mm", type=float, required=True)
    materialize_parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "screen":
        screen(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()
