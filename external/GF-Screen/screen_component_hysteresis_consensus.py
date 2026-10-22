#!/usr/bin/env python3
"""Screen anchor-preserving disconnected lesions with hysteresis and consensus.

The deployed ``base`` mask is immutable. Candidate components are taken from a
sensitive low-threshold segmentation only when they are completely disjoint
from the base. A retained component must also contain a high-threshold seed and
pass predeclared agreement checks from a second GF checkpoint and the separate
BiomedParse whole-lesion and tumor prompts. Finite growth radii implement true
hysteresis: only low-threshold voxels near a high-confidence seed are imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "handoff_bundle" / "code"))
from evaluate import score_case  # noqa: E402


@dataclass(frozen=True)
class Config:
    name: str
    max_mm3: float
    min_seed_fraction: float
    min_seed_mm3: float
    min_persistence_fraction: float
    min_persistence_mm3: float
    min_union_prompt_fraction: float
    min_union_prompt_mm3: float
    min_each_prompt_mm3: float
    min_consensus_votes: int
    grow_mm: float


# A deliberately small, mechanistically distinct grid. The two prompts count
# separately; persistence is the third consensus vote. Every rule additionally
# requires a high-threshold seed from the same model as the low-threshold mask.
CONFIGS = (
    Config("union25_persist_2k_full", 2000, .10, 100, .10, 100, .25, 100, 0, 0, math.inf),
    Config("union25_persist_5k_full", 5000, .10, 100, .10, 100, .25, 100, 0, 0, math.inf),
    Config("union25_persist_2k_g10", 2000, .10, 100, .10, 100, .25, 100, 0, 0, 10),
    Config("union25_persist_5k_g10", 5000, .10, 100, .10, 100, .25, 100, 0, 0, 10),
    Config("dual10_persist_2k_full", 2000, .10, 100, .10, 100, 0, 0, 10, 0, math.inf),
    Config("dual10_persist_5k_full", 5000, .10, 100, .10, 100, 0, 0, 10, 0, math.inf),
    Config("dual10_persist_2k_g10", 2000, .10, 100, .10, 100, 0, 0, 10, 0, 10),
    Config("dual10_persist_5k_g10", 5000, .10, 100, .10, 100, 0, 0, 10, 0, 10),
    Config("vote2_seed_2k_full", 2000, .10, 100, 0, 0, 0, 0, 10, 2, math.inf),
    Config("vote2_seed_5k_full", 5000, .10, 100, 0, 0, 0, 0, 10, 2, math.inf),
    Config("vote3_seed_2k_full", 2000, .10, 100, 0, 0, 0, 0, 10, 3, math.inf),
    Config("vote3_seed_5k_full", 5000, .10, 100, 0, 0, 0, 0, 10, 3, math.inf),
    Config("vote2_seed_2k_g10", 2000, .10, 100, 0, 0, 0, 0, 10, 2, 10),
    Config("vote2_seed_5k_g10", 5000, .10, 100, 0, 0, 0, 0, 10, 2, 10),
)
CONFIG_BY_NAME = {config.name: config for config in CONFIGS}


def load_bool(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(bool), image


def validate_geometry(name: str, reference: nib.Nifti1Image, other: nib.Nifti1Image) -> None:
    if reference.shape != other.shape or not np.allclose(reference.affine, other.affine, atol=1e-3):
        raise ValueError(f"geometry mismatch: {name}")


def overlap_stats(labels: np.ndarray, mask: np.ndarray, counts: np.ndarray, voxel_mm3: float):
    overlap = np.bincount(labels[mask], minlength=len(counts))
    fraction = np.divide(
        overlap, counts, out=np.zeros(len(counts), dtype=np.float64), where=counts > 0
    )
    return overlap, fraction, overlap.astype(np.float64) * voxel_mm3


def component_features(
    base: np.ndarray,
    candidate: np.ndarray,
    seed: np.ndarray,
    persistence: np.ndarray,
    whole: np.ndarray,
    tumor: np.ndarray,
    spacing: tuple[float, float, float],
):
    labels = cc3d.connected_components(candidate.astype(np.uint8), connectivity=26)
    counts = np.bincount(labels.ravel())
    voxel_mm3 = float(np.prod(spacing))
    volumes = counts.astype(np.float64) * voxel_mm3
    base_overlap = np.bincount(labels[base], minlength=len(counts))
    _, seed_fraction, seed_mm3 = overlap_stats(labels, seed, counts, voxel_mm3)
    _, persistence_fraction, persistence_mm3 = overlap_stats(
        labels, persistence, counts, voxel_mm3
    )
    whole_counts, _, whole_mm3 = overlap_stats(labels, whole, counts, voxel_mm3)
    tumor_counts, _, tumor_mm3 = overlap_stats(labels, tumor, counts, voxel_mm3)
    union_counts, union_fraction, union_mm3 = overlap_stats(
        labels, whole | tumor, counts, voxel_mm3
    )
    eligible = base_overlap == 0
    eligible[0] = False
    return {
        "labels": labels,
        "counts": counts,
        "volumes": volumes,
        "eligible": eligible,
        "seed_fraction": seed_fraction,
        "seed_mm3": seed_mm3,
        "persistence_fraction": persistence_fraction,
        "persistence_mm3": persistence_mm3,
        "whole_counts": whole_counts,
        "whole_mm3": whole_mm3,
        "tumor_counts": tumor_counts,
        "tumor_mm3": tumor_mm3,
        "union_counts": union_counts,
        "union_fraction": union_fraction,
        "union_mm3": union_mm3,
    }


def selected_labels(features: dict[str, np.ndarray], config: Config) -> np.ndarray:
    keep = features["eligible"].copy()
    keep &= features["volumes"] <= config.max_mm3
    keep &= features["seed_fraction"] >= config.min_seed_fraction
    keep &= features["seed_mm3"] >= config.min_seed_mm3
    if config.min_persistence_fraction or config.min_persistence_mm3:
        keep &= features["persistence_fraction"] >= config.min_persistence_fraction
        keep &= features["persistence_mm3"] >= config.min_persistence_mm3
    if config.min_union_prompt_fraction or config.min_union_prompt_mm3:
        keep &= features["union_fraction"] >= config.min_union_prompt_fraction
        keep &= features["union_mm3"] >= config.min_union_prompt_mm3
    if config.min_each_prompt_mm3 and not config.min_consensus_votes:
        keep &= features["whole_mm3"] >= config.min_each_prompt_mm3
        keep &= features["tumor_mm3"] >= config.min_each_prompt_mm3
    if config.min_consensus_votes:
        votes = (
            (features["whole_mm3"] >= config.min_each_prompt_mm3).astype(np.uint8)
            + (features["tumor_mm3"] >= config.min_each_prompt_mm3).astype(np.uint8)
            + (
                (features["persistence_fraction"] >= 0.10)
                & (features["persistence_mm3"] >= 100)
            ).astype(np.uint8)
        )
        keep &= votes >= config.min_consensus_votes
    keep[0] = False
    return np.flatnonzero(keep)


def materialize_prediction(
    base: np.ndarray,
    seed: np.ndarray,
    features: dict[str, np.ndarray],
    config: Config,
    spacing: tuple[float, float, float],
    seed_distance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    chosen = selected_labels(features, config)
    keep = np.zeros(len(features["counts"]), dtype=bool)
    keep[chosen] = True
    addition = keep[features["labels"]]
    if not math.isinf(config.grow_mm):
        if seed_distance is None:
            seed_distance = distance_transform_edt(~seed, sampling=spacing)
        addition &= seed_distance <= config.grow_mm
    return base | addition, chosen


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
                f"case mismatch in {directory}: missing={sorted(expected-observed)[:3]} "
                f"extra={sorted(observed-expected)[:3]}"
            )


def load_case(paths: tuple[str, ...]):
    name, *mask_paths = paths
    loaded = [load_bool(Path(path)) for path in mask_paths]
    arrays = [item[0] for item in loaded]
    images = [item[1] for item in loaded]
    for label, image in zip(
        ("candidate", "seed", "persistence", "whole", "tumor", "target"), images[1:]
    ):
        validate_geometry(f"{label}/{name}", images[0], image)
    spacing = tuple(float(value) for value in images[0].header.get_zooms()[:3])
    return name, arrays, images[0], spacing


def score_one(paths: tuple[str, ...]):
    name, arrays, _, spacing = load_case(paths)
    base, candidate, seed, persistence, whole, tumor, target = arrays
    features = component_features(base, candidate, seed, persistence, whole, tumor, spacing)
    seed_distance = distance_transform_edt(~seed, sampling=spacing)
    values = []
    cache = {}
    for config in CONFIGS:
        prediction, chosen = materialize_prediction(
            base, seed, features, config, spacing, seed_distance=seed_distance
        )
        key = (tuple(int(value) for value in chosen), config.grow_mm)
        if key not in cache:
            dsc, nsd = score_case(target.astype(np.uint8), prediction.astype(np.uint8), spacing)
            target_overlap = np.bincount(
                features["labels"][target], minlength=len(features["counts"])
            )
            cache[key] = (
                float(dsc),
                float(nsd),
                int((prediction & ~base).sum()),
                int(len(chosen)),
                int((target_overlap[chosen] > 0).sum()),
            )
        values.append((config.name, *cache[key]))
    return name.removesuffix(".nii.gz"), values


def build_items(args, names: list[str], include_target: bool):
    directories = (args.base, args.candidate, args.seed, args.persistence, args.whole, args.tumor)
    if include_target:
        directories += (args.gt,)
    validate_names(names, directories)
    return [tuple([name] + [str(directory / name) for directory in directories]) for name in names]


def screen(args) -> None:
    names = common_names(args.base)
    items = build_items(args, names, include_target=True)
    rows = {config.name: [] for config in CONFIGS}
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (case, values) in enumerate(executor.map(score_one, items), 1):
            for name, *values_for_config in values:
                rows[name].append((case, *values_for_config))
            print(f"[{index:03d}/{len(names):03d}] {case}", flush=True)
    metrics_dir = args.output / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    summary = []
    for config in CONFIGS:
        case_rows = rows[config.name]
        metrics = metrics_dir / f"{config.name}.metrics.csv"
        with metrics.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("Name", "Lesion_DSC", "Lesion_NSD", "added_voxels", "selected", "true_selected")
            )
            for row in case_rows:
                writer.writerow(row)
        dsc = float(np.mean([row[1] for row in case_rows]))
        nsd = float(np.mean([row[2] for row in case_rows]))
        summary.append(
            asdict(config)
            | {
                "DSC": dsc,
                "NSD": nsd,
                "combined": (dsc + nsd) / 2,
                "changed_cases": sum(row[3] > 0 for row in case_rows),
                "selected_components": sum(row[4] for row in case_rows),
                "true_selected_components": sum(row[5] for row in case_rows),
                "metrics": str(metrics.resolve()),
            }
        )
    summary.sort(key=lambda row: (-row["combined"], -row["DSC"], row["name"]))
    with (args.output / "summary.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "configs.json").write_text(
        json.dumps([asdict(config) for config in CONFIGS], indent=2) + "\n", encoding="utf-8"
    )
    for row in summary:
        print(
            f"{row['name']} DSC={row['DSC']:.6f} NSD={row['NSD']:.6f} "
            f"changed={row['changed_cases']} components={row['selected_components']} "
            f"true={row['true_selected_components']}"
        )


def materialize_one(paths: tuple[str, ...], config: Config, output: Path) -> str:
    name, arrays, reference, spacing = load_case(paths)
    base, candidate, seed, persistence, whole, tumor = arrays
    features = component_features(base, candidate, seed, persistence, whole, tumor, spacing)
    seed_distance = None
    if not math.isinf(config.grow_mm):
        seed_distance = distance_transform_edt(~seed, sampling=spacing)
    prediction, _ = materialize_prediction(
        base, seed, features, config, spacing, seed_distance=seed_distance
    )
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(prediction.astype(np.uint8), reference.affine, header), output / name)
    return name


def materialize(args) -> None:
    names = common_names(args.base)
    items = build_items(args, names, include_target=False)
    config = CONFIG_BY_NAME[args.config]
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(materialize_one, item, config, args.output) for item in items]
        for index, future in enumerate(futures, 1):
            print(f"[{index:03d}/{len(names):03d}] {future.result()}", flush=True)


def add_common(parser):
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--persistence", type=Path, required=True)
    parser.add_argument("--whole", type=Path, required=True)
    parser.add_argument("--tumor", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen_parser = subparsers.add_parser("screen")
    add_common(screen_parser)
    screen_parser.add_argument("--gt", type=Path, required=True)
    screen_parser.add_argument("--output", type=Path, required=True)
    materialize_parser = subparsers.add_parser("materialize")
    add_common(materialize_parser)
    materialize_parser.add_argument("--config", choices=sorted(CONFIG_BY_NAME), required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "screen":
        screen(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()
