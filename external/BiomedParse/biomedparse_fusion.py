#!/usr/bin/env python3
"""Bounded component fusion of BiomedParse and the immutable GF champion.

The public screen is deliberately finite and declared in :func:`configurations`.
It evaluates standalone support, recall-oriented support-component addition,
and precision-oriented confirmation of GF components.  Only a small
family-balanced shortlist is materialized for the official DSC/NSD scorer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import re
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage


def configurations() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for minimum in (0.0, 500.0, 1000.0, 3000.0):
        configs.append({"kind": "standalone", "support_min_mm3": minimum})
    for mode in ("disjoint", "all"):
        for minimum in (500.0, 1000.0, 3000.0):
            configs.append(
                {"kind": "union", "mode": mode, "support_min_mm3": minimum}
            )
    for distance in (0.0, 5.0):
        for rescue in (3000.0, 10000.0, "inf"):
            configs.append(
                {"kind": "confirm", "distance_mm": distance, "rescue_min_mm3": rescue}
            )
    return configs


def top_k_rescue_configurations() -> list[dict[str, object]]:
    """Finite opt-in safeguards for external-model false-negative cases."""

    return [
        {
            "kind": "confirm", "distance_mm": 0.0, "rescue_min_mm3": "inf",
            "rescue_top_k": 1, "rescue_top_k_min_mm3": 1000.0,
            "rescue_top_k_max_mm3": 10000.0,
        },
        {
            "kind": "confirm", "distance_mm": 0.0, "rescue_min_mm3": "inf",
            "rescue_top_k": 1, "rescue_top_k_min_mm3": 3000.0,
            "rescue_top_k_max_mm3": 10000.0,
        },
        {
            "kind": "confirm", "distance_mm": 0.0, "rescue_min_mm3": "inf",
            "rescue_top_k": 2, "rescue_top_k_min_mm3": 1000.0,
            "rescue_top_k_max_mm3": 10000.0,
        },
    ]


def confirmation_support_strength_configurations() -> list[dict[str, object]]:
    """Predeclared stricter variants of exact prompt-overlap confirmation.

    The promoted rule keeps a GF component after even one overlapping support
    voxel.  These configurations test whether requiring a small physical
    overlap or component-relative overlap rejects accidental contacts while
    retaining genuinely prompt-supported lesions.  When both thresholds are
    present they are conjunctive; this is deliberately precision-oriented.
    """

    common: dict[str, object] = {
        "kind": "confirm",
        "distance_mm": 0.0,
        "rescue_min_mm3": "inf",
    }
    configs: list[dict[str, object]] = []
    for minimum in (3.0, 10.0, 30.0, 100.0):
        configs.append({**common, "min_support_overlap_mm3": minimum})
    for fraction in (0.001, 0.005, 0.01, 0.025, 0.05):
        configs.append({**common, "min_support_overlap_fraction": fraction})
    configs.extend(
        [
            {
                **common,
                "min_support_overlap_mm3": 3.0,
                "min_support_overlap_fraction": 0.001,
            },
            {
                **common,
                "min_support_overlap_mm3": 10.0,
                "min_support_overlap_fraction": 0.001,
            },
            {
                **common,
                "min_support_overlap_mm3": 10.0,
                "min_support_overlap_fraction": 0.005,
            },
            {
                **common,
                "min_support_overlap_fraction": 0.003,
                "weak_support_rescue_min_mm3": 3000.0,
            },
            {
                **common,
                "min_support_overlap_fraction": 0.003,
                "weak_support_rescue_min_mm3": 5000.0,
            },
            {
                **common,
                "min_support_overlap_fraction": 0.003,
                "weak_support_rescue_min_mm3": 10000.0,
            },
        ]
    )
    return configs


def confirmation_keep_from_overlap(
    overlap_counts: np.ndarray,
    component_counts: np.ndarray,
    voxel_mm3: float,
    config: dict[str, object],
) -> np.ndarray:
    """Return component IDs meeting the configured prompt-support strength."""

    overlap_counts = np.asarray(overlap_counts)
    component_counts = np.asarray(component_counts)
    if overlap_counts.shape != component_counts.shape:
        raise ValueError("overlap and component counts must have identical shapes")
    keep = overlap_counts > 0
    if "min_support_overlap_mm3" in config:
        minimum_mm3 = float(config["min_support_overlap_mm3"])
        if minimum_mm3 < 0:
            raise ValueError("min_support_overlap_mm3 must be non-negative")
        keep &= overlap_counts * float(voxel_mm3) >= minimum_mm3
    if "min_support_overlap_fraction" in config:
        minimum_fraction = float(config["min_support_overlap_fraction"])
        if not 0 <= minimum_fraction <= 1:
            raise ValueError("min_support_overlap_fraction must be in [0, 1]")
        fraction = np.divide(
            overlap_counts,
            component_counts,
            out=np.zeros_like(overlap_counts, dtype=np.float64),
            where=component_counts > 0,
        )
        keep &= fraction >= minimum_fraction
    if "weak_support_rescue_min_mm3" in config:
        rescue_min_mm3 = float(config["weak_support_rescue_min_mm3"])
        if rescue_min_mm3 < 0:
            raise ValueError("weak_support_rescue_min_mm3 must be non-negative")
        # Unlike the legacy size rescue, this never restores a completely
        # unsupported component. It only relaxes the strength requirement for
        # a large component that has at least one prompt-support voxel.
        keep |= (
            (overlap_counts > 0)
            & (component_counts * float(voxel_mm3) >= rescue_min_mm3)
        )
    keep[0] = False
    return keep


def apply_confirmation_rescue(
    keep: np.ndarray, sizes: np.ndarray, config: dict[str, object]
) -> np.ndarray:
    """Apply global and optional bounded top-k rescue to component IDs."""

    keep = keep.copy()
    rescue = config["rescue_min_mm3"]
    rescue_value = math.inf if rescue == "inf" else float(rescue)
    if math.isfinite(rescue_value):
        keep |= sizes >= rescue_value
        keep[0] = False

    top_k = int(config.get("rescue_top_k", 0))
    if top_k > 0:
        minimum = float(config["rescue_top_k_min_mm3"])
        maximum = float(config["rescue_top_k_max_mm3"])
        candidate_ids = np.flatnonzero(
            (~keep) & (sizes >= minimum) & (sizes <= maximum)
        )
        candidate_ids = candidate_ids[candidate_ids != 0]
        order = np.argsort(-sizes[candidate_ids], kind="stable")
        keep[candidate_ids[order[:top_k]]] = True
    keep[0] = False
    return keep


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(prediction & target) / denominator)


def component_filter(mask: np.ndarray, voxel_mm3: float, minimum: float) -> np.ndarray:
    component_map = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    sizes = np.bincount(component_map.ravel()) * voxel_mm3
    keep = sizes >= minimum
    keep[0] = False
    return keep[component_map]


def support_within_distance(
    support: np.ndarray,
    spacing: tuple[float, float, float],
    distance_mm: float,
) -> np.ndarray:
    """Return the exact voxel-centre Euclidean dilation of ``support``.

    The fusion grid queries only a small fixed radius. Constructing its
    anisotropic ellipsoidal footprint avoids allocating and computing a dense
    float64 distance map over every CT voxel while preserving the EDT result.
    """

    support = np.asarray(support, dtype=bool)
    distance_mm = float(distance_mm)
    if distance_mm <= 0 or not np.any(support):
        return support
    spacing_array = np.asarray(spacing, dtype=np.float64)
    radii = np.ceil(distance_mm / spacing_array).astype(int)
    offsets = np.ogrid[tuple(slice(-radius, radius + 1) for radius in radii)]
    squared_distance = sum(
        (offset * axis_spacing) ** 2
        for offset, axis_spacing in zip(offsets, spacing_array)
    )
    footprint = squared_distance <= distance_mm**2 + 1e-6
    return ndimage.binary_dilation(support, structure=footprint)


def apply_config(
    base: np.ndarray,
    support: np.ndarray,
    spacing: tuple[float, float, float],
    config: dict[str, object],
) -> np.ndarray:
    base = np.asarray(base, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if base.shape != support.shape:
        raise ValueError(f"shape mismatch: {base.shape} != {support.shape}")
    voxel_mm3 = float(np.prod(spacing))
    kind = str(config["kind"])
    if kind in {"standalone", "union"}:
        support_map = cc3d.connected_components(support.astype(np.uint8), connectivity=26)
        sizes = np.bincount(support_map.ravel()) * voxel_mm3
        eligible = sizes >= float(config["support_min_mm3"])
        eligible[0] = False
        if kind == "standalone":
            return eligible[support_map]
        mode = str(config["mode"])
        if mode == "disjoint":
            overlaps_base = np.bincount(support_map[base], minlength=len(sizes))
            eligible &= overlaps_base == 0
        elif mode != "all":
            raise ValueError(f"unknown union mode: {mode}")
        return base | eligible[support_map]
    if kind == "confirm":
        base_map = cc3d.connected_components(base.astype(np.uint8), connectivity=26)
        component_counts = np.bincount(base_map.ravel())
        sizes = component_counts * voxel_mm3
        distance_mm = float(config["distance_mm"])
        supported = support
        if distance_mm > 0:
            supported = support_within_distance(support, spacing, distance_mm)
        overlap_counts = np.bincount(
            base_map[supported & base], minlength=len(component_counts)
        )
        keep = confirmation_keep_from_overlap(
            overlap_counts, component_counts, voxel_mm3, config
        )
        keep = apply_confirmation_rescue(keep, sizes, config)
        return keep[base_map]
    raise ValueError(f"unknown fusion kind: {kind}")


def apply_configs(
    base: np.ndarray,
    support: np.ndarray,
    spacing: tuple[float, float, float],
    configs: list[dict[str, object]],
) -> list[np.ndarray]:
    """Apply a grid while caching connected components and distance maps."""

    base = np.asarray(base, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if base.shape != support.shape:
        raise ValueError(f"shape mismatch: {base.shape} != {support.shape}")
    voxel_mm3 = float(np.prod(spacing))
    support_map = cc3d.connected_components(support.astype(np.uint8), connectivity=26)
    support_sizes = np.bincount(support_map.ravel()) * voxel_mm3
    support_overlaps_base = np.bincount(support_map[base], minlength=len(support_sizes))
    base_map = cc3d.connected_components(base.astype(np.uint8), connectivity=26)
    base_counts = np.bincount(base_map.ravel())
    base_sizes = base_counts * voxel_mm3
    distances = {
        float(config["distance_mm"])
        for config in configs
        if config["kind"] == "confirm" and float(config["distance_mm"]) > 0
    }
    support_by_distance = {0.0: support}
    support_by_distance.update(
        {
            distance: support_within_distance(support, spacing, distance)
            for distance in distances
        }
    )

    outputs: list[np.ndarray] = []
    for config in configs:
        kind = str(config["kind"])
        if kind in {"standalone", "union"}:
            eligible = support_sizes >= float(config["support_min_mm3"])
            eligible[0] = False
            if kind == "standalone":
                outputs.append(eligible[support_map])
                continue
            if config["mode"] == "disjoint":
                eligible &= support_overlaps_base == 0
            elif config["mode"] != "all":
                raise ValueError(f"unknown union mode: {config['mode']}")
            outputs.append(base | eligible[support_map])
            continue
        if kind == "confirm":
            distance = float(config["distance_mm"])
            overlap_counts = np.bincount(
                base_map[support_by_distance[distance] & base],
                minlength=len(base_counts),
            )
            keep = confirmation_keep_from_overlap(
                overlap_counts, base_counts, voxel_mm3, config
            )
            keep = apply_confirmation_rescue(keep, base_sizes, config)
            outputs.append(keep[base_map])
            continue
        raise ValueError(f"unknown fusion kind: {kind}")
    return outputs


def config_tag(support_name: str, config: dict[str, object]) -> str:
    parts = [support_name, str(config["kind"])]
    for key in (
        "mode", "support_min_mm3", "distance_mm", "rescue_min_mm3",
        "rescue_top_k", "rescue_top_k_min_mm3", "rescue_top_k_max_mm3",
        "min_support_overlap_mm3", "min_support_overlap_fraction",
        "weak_support_rescue_min_mm3",
    ):
        if key in config:
            value = str(config[key]).replace(".0", "").replace(".", "p")
            parts.append(f"{key}_{value}")
    return re.sub(r"[^A-Za-z0-9_]+", "_", "_".join(parts))


def load_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(bool), image


def write_mask(mask: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header), str(path))


def parse_support(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"support must be NAME=PATH, got {value}")
        name, path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"invalid or duplicate support name: {name}")
        result[name] = Path(path)
    return result


def materialize(
    base_dir: Path,
    support_dir: Path,
    output_dir: Path,
    config: dict[str, object],
) -> None:
    base_paths = sorted(base_dir.glob("*.nii.gz"))
    if not base_paths:
        raise FileNotFoundError(f"no base masks in {base_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, base_path in enumerate(base_paths, 1):
        support_path = support_dir / base_path.name
        if not support_path.exists():
            raise FileNotFoundError(support_path)
        base, base_image = load_mask(base_path)
        support, support_image = load_mask(support_path)
        if base.shape != support.shape or not np.allclose(
            base_image.affine, support_image.affine, atol=1e-3
        ):
            raise ValueError(f"geometry mismatch for {base_path.name}")
        spacing = tuple(float(value) for value in base_image.header.get_zooms()[:3])
        fused = apply_config(base, support, spacing, config)
        write_mask(fused, base_image, output_dir / base_path.name)
        print(f"[{index:03d}/{len(base_paths):03d}] {base_path.name}", flush=True)


def score_case(
    label_path: Path,
    base_dir: Path,
    supports: dict[str, Path],
    all_configs: list[dict[str, object]],
) -> tuple[str, float, list[float]]:
    """Score one case independently so the bounded screen can use CPU cores."""

    base_path = base_dir / label_path.name
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    target, target_image = load_mask(label_path)
    base, base_image = load_mask(base_path)
    if base.shape != target.shape or not np.allclose(
        base_image.affine, target_image.affine, atol=1e-3
    ):
        raise ValueError(f"base geometry mismatch for {label_path.name}")
    spacing = tuple(float(value) for value in base_image.header.get_zooms()[:3])
    values = [0.0] * len(all_configs)
    for support_name, directory in supports.items():
        support_path = directory / label_path.name
        if not support_path.exists():
            raise FileNotFoundError(support_path)
        support, support_image = load_mask(support_path)
        if support.shape != base.shape or not np.allclose(
            support_image.affine, base_image.affine, atol=1e-3
        ):
            raise ValueError(f"support geometry mismatch for {label_path.name}: {support_name}")
        indices = [
            config_index
            for config_index, config in enumerate(all_configs)
            if config["support"] == support_name
        ]
        local_configs = [all_configs[config_index] for config_index in indices]
        predictions = apply_configs(base, support, spacing, local_configs)
        for config_index, prediction in zip(indices, predictions):
            values[config_index] = dice(prediction, target)
    return label_path.name, dice(base, target), values


def score_case_from_tuple(
    arguments: tuple[Path, Path, dict[str, Path], list[dict[str, object]]],
) -> tuple[str, float, list[float]]:
    return score_case(*arguments)


def materialize_from_tuple(
    arguments: tuple[Path, Path, Path, dict[str, object]],
) -> None:
    materialize(*arguments)


def materialize_shortlist(
    report: dict[str, object], output: Path, workers: int = 3
) -> Path:
    """Materialize the fixed shortlist stored in a completed screen report."""

    rows = list(report["results"])
    by_tag = {str(row["tag"]): row for row in rows}
    shortlist = [by_tag[str(tag)] for tag in report["shortlist_tags"]]
    base = Path(str(report["base"]))
    supports = {
        str(name): Path(str(path))
        for name, path in dict(report["supports"]).items()
    }
    shortlist_rows = []
    jobs = []
    for row in shortlist:
        tag = str(row["tag"])
        config_path = output / "configs" / f"{tag}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            key: row[key]
            for key in (
                "kind", "mode", "support_min_mm3", "distance_mm", "rescue_min_mm3",
                "rescue_top_k", "rescue_top_k_min_mm3", "rescue_top_k_max_mm3",
                "min_support_overlap_mm3", "min_support_overlap_fraction",
                "weak_support_rescue_min_mm3",
            )
            if key in row
        }
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        predictions = output / "predictions" / tag
        jobs.append((base, supports[str(row["support"])], predictions, config))
        shortlist_rows.append(
            {
                **row,
                "config": str(config_path),
                "predictions": str(predictions),
            }
        )
    if workers == 1:
        for job in jobs:
            materialize_from_tuple(job)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(materialize_from_tuple, jobs))
    fields = [
        "support", "kind", "mode", "support_min_mm3", "distance_mm",
        "rescue_min_mm3", "tag", "mean_dice", "mean_gain", "median_gain",
        "win_rate", "loss_rate", "catastrophic_regression_rate", "config",
        "predictions", "min_support_overlap_mm3",
        "min_support_overlap_fraction", "weak_support_rescue_min_mm3",
    ]
    for row in shortlist_rows:
        for field in fields:
            row.setdefault(field, "-")
    shortlist_path = output / "shortlist.tsv"
    temporary = shortlist_path.with_suffix(shortlist_path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        # Keep the manifest shell-safe on Linux.  The csv module otherwise
        # emits CRLF, leaving a literal carriage return on the final path when
        # Bash reads the TSV and making an existing directory appear missing.
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(shortlist_rows)
    temporary.replace(shortlist_path)
    print(shortlist_path.read_text(), end="")
    return shortlist_path


def run_screen(args: argparse.Namespace) -> None:
    supports = parse_support(args.support)
    label_paths = sorted(args.labels.glob("*.nii.gz"))
    if not label_paths:
        raise FileNotFoundError(f"no labels in {args.labels}")
    if getattr(args, "top_k_rescue_only", False):
        declared_configs = top_k_rescue_configurations()
    else:
        declared_configs = configurations()
    if (
        getattr(args, "include_top_k_rescue", False)
        and not getattr(args, "top_k_rescue_only", False)
    ):
        declared_configs.extend(top_k_rescue_configurations())
    if getattr(args, "include_support_strength", False):
        declared_configs.extend(confirmation_support_strength_configurations())
    selected_kinds = getattr(args, "kind", None)
    if selected_kinds:
        selected = set(selected_kinds)
        declared_configs = [
            config for config in declared_configs if config["kind"] in selected
        ]
    all_configs = [
        {"support": support_name, **config}
        for support_name in sorted(supports)
        for config in declared_configs
    ]
    workers = max(1, int(getattr(args, "workers", 1)))
    score_arguments = [
        (label_path, args.base, supports, all_configs) for label_path in label_paths
    ]
    if workers == 1:
        case_results = (score_case(*arguments) for arguments in score_arguments)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        case_results = executor.map(score_case_from_tuple, score_arguments)

    scores = [[] for _ in all_configs]
    baseline_scores: list[float] = []
    try:
        for index, (case_name, baseline_score, values) in enumerate(case_results, 1):
            baseline_scores.append(baseline_score)
            for config_index, value in enumerate(values):
                scores[config_index].append(value)
            print(f"[{index:03d}/{len(label_paths):03d}] {case_name}", flush=True)
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)

    baseline = np.asarray(baseline_scores)
    rows = []
    for config, values_list in zip(all_configs, scores):
        values = np.asarray(values_list)
        gain = values - baseline
        rows.append(
            {
                **config,
                "tag": config_tag(str(config["support"]), config),
                "mean_dice": float(values.mean()),
                "mean_gain": float(gain.mean()),
                "median_gain": float(np.median(gain)),
                "win_rate": float(np.mean(gain > 0)),
                "loss_rate": float(np.mean(gain < 0)),
                "catastrophic_regression_rate": float(np.mean(gain < -0.20)),
            }
        )
    rows.sort(key=lambda row: (float(row["mean_dice"]), float(row["win_rate"])), reverse=True)
    shortlist: list[dict[str, object]] = []
    for kind in ("standalone", "union", "confirm"):
        shortlist.extend([row for row in rows if row["kind"] == kind][:2])
    for row in rows[:2]:
        if row not in shortlist:
            shortlist.append(row)
    if getattr(args, "include_top_k_rescue", False):
        top_k_rows = [row for row in rows if int(row.get("rescue_top_k", 0)) > 0]
        if top_k_rows and top_k_rows[0] not in shortlist:
            # The bounded safeguard is selected for tail robustness, not only
            # mean rank, so guarantee exact scoring of its strongest member.
            shortlist.append(top_k_rows[0])

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "base": str(args.base.resolve()),
        "supports": {name: str(path.resolve()) for name, path in supports.items()},
        "labels": str(args.labels.resolve()),
        "cases": len(label_paths),
        "baseline_mean_dice": float(baseline.mean()),
        "declared_configurations_per_support": len(declared_configs),
        "results": rows,
        "shortlist_tags": [row["tag"] for row in shortlist],
    }
    (args.output / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    materialize_shortlist(report, args.output, workers=min(args.workers, 4))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen = subparsers.add_parser("screen")
    screen.add_argument("--base", type=Path, required=True)
    screen.add_argument("--support", action="append", required=True)
    screen.add_argument("--labels", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--workers", type=int, default=8)
    screen.add_argument(
        "--kind", action="append", choices=("standalone", "union", "confirm"),
        help="optionally restrict the fixed configuration family",
    )
    screen.add_argument(
        "--include-top-k-rescue", action="store_true",
        help="include three fixed bounded top-k confirmation safeguards",
    )
    screen.add_argument(
        "--top-k-rescue-only", action="store_true",
        help="evaluate only the three fixed bounded top-k confirmation safeguards",
    )
    screen.add_argument(
        "--include-support-strength", action="store_true",
        help="include predeclared stricter component-overlap confirmation rules",
    )
    screen.set_defaults(function=run_screen)
    apply = subparsers.add_parser("materialize")
    apply.add_argument("--base", type=Path, required=True)
    apply.add_argument("--support", type=Path, required=True)
    apply.add_argument("--config", type=Path, required=True)
    resume = subparsers.add_parser("resume-screen")
    resume.add_argument("--screen-json", type=Path, required=True)
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--workers", type=int, default=3)
    apply.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "materialize":
        config = json.loads(args.config.read_text())
        materialize(args.base, args.support, args.output, config)
    elif args.command == "resume-screen":
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        materialize_shortlist(
            json.loads(args.screen_json.read_text()), args.output, workers=args.workers
        )
    else:
        args.function(args)


if __name__ == "__main__":
    main()
