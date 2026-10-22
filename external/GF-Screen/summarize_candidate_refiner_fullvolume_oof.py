#!/usr/bin/env python3
"""Select and audit a candidate-refiner deployment from full-volume OOF metrics.

All thresholds entering this tool must have been declared from patch-level OOF.
This final calibration uses only case-held-out full-volume predictions and never
reads public or hidden data. Bootstrap resampling is stratified by source cohort
so confidence intervals do not accidentally reflect a different cohort mix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


METRICS = ("Lesion_DSC", "Lesion_NSD")
GEOMETRY_GATE_KEYS = {
    "min_seed_coverage",
    "min_refined_overlap",
    "min_volume_ratio",
    "max_volume_ratio",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_metrics_path(summary: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = summary.parent / path
    return local if local.is_file() else path


def load_manifest(path: Path) -> tuple[list[str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    names = [str(item["name"]) for item in items]
    if not names or len(names) != len(set(names)):
        raise ValueError("manifest must contain unique cases")
    cohorts = {str(item["name"]): str(item["cohort"]) for item in items}
    if any(item.get("complete_label") is not True for item in items):
        raise ValueError("full-volume OOF requires complete labels")
    return sorted(names), cohorts


def load_metric_csv(path: Path, expected_names: list[str]) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row.get("Name", ""))
        if not name or name in by_name:
            raise ValueError(f"missing or duplicate case in {path}: {name!r}")
        by_name[name] = row
    observed = set(by_name)
    expected = set(expected_names)
    if observed != expected:
        raise ValueError(
            f"case coverage mismatch in {path}: expected={len(expected)} "
            f"observed={len(observed)} missing={sorted(expected-observed)[:3]} "
            f"extra={sorted(observed-expected)[:3]}"
        )
    arrays: dict[str, np.ndarray] = {}
    for metric in METRICS:
        values = np.asarray([float(by_name[name][metric]) for name in expected_names])
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"invalid {metric} values in {path}")
        arrays[metric] = values
    return arrays


def parse_source_thresholds(
    raw: str, label: str = "source classification"
) -> dict[str, float]:
    """Parse a canonical per-source threshold mapping from one TSV cell."""

    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} thresholds must be a JSON object")
    parsed: dict[str, float] = {}
    for raw_name, raw_value in payload.items():
        name = str(raw_name).strip()
        value = float(raw_value)
        if not name or name in parsed or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"invalid {label} threshold mapping")
        parsed[name] = value
    return dict(sorted(parsed.items()))


def parse_source_modes(raw: str) -> dict[str, str]:
    """Parse a canonical per-source deployment-mode mapping from one TSV cell."""

    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("source deployment modes must be a JSON object")
    allowed = {
        "refined", "refined_or_seed", "seed", "seed_crop", "seed_union", "prune_base"
    }
    parsed: dict[str, str] = {}
    for raw_name, raw_mode in payload.items():
        name = str(raw_name).strip()
        mode = str(raw_mode)
        if not name or name in parsed or mode not in allowed:
            raise ValueError("invalid source deployment mode mapping")
        parsed[name] = mode
    return dict(sorted(parsed.items()))


def parse_geometry_gates(raw: str) -> dict[str, float]:
    """Parse the bounded, deployment-visible contour consistency gates."""

    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) - GEOMETRY_GATE_KEYS:
        raise ValueError("geometry gates must use the supported keys")
    parsed = {str(name): float(value) for name, value in payload.items()}
    if not all(math.isfinite(value) and value >= 0 for value in parsed.values()):
        raise ValueError("geometry gates must be finite and nonnegative")
    if any(parsed.get(name, 0.0) > 1 for name in ("min_seed_coverage", "min_refined_overlap")):
        raise ValueError("overlap geometry gates must be in [0,1]")
    if parsed.get("min_volume_ratio", 0.0) > parsed.get(
        "max_volume_ratio", float("inf")
    ):
        raise ValueError("geometry volume-ratio bounds are inverted")
    return dict(sorted(parsed.items()))


def load_candidates(
    summary: Path, manifest_names: list[str]
) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {
        "variant",
        "classification_threshold",
        "segmentation_threshold",
        "DSC",
        "NSD",
        "metrics",
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError("invalid or empty full-volume OOF summary")
    has_mode_grid = "deployment_mode" in rows[0]
    source_maps = [
        parse_source_thresholds(str(row.get("source_classification_thresholds", "")))
        for row in rows
    ]
    source_segmentation_maps = [
        parse_source_thresholds(
            str(row.get("source_segmentation_thresholds", "")),
            "source segmentation",
        )
        for row in rows
    ]
    source_mode_maps = [
        parse_source_modes(str(row.get("source_deployment_modes", "")))
        for row in rows
    ]
    geometry_gate_maps = [
        parse_geometry_gates(str(row.get("geometry_gates", ""))) for row in rows
    ]
    has_source_grid = any(
        classification or segmentation or modes or geometry
        for classification, segmentation, modes, geometry in zip(
            source_maps,
            source_segmentation_maps,
            source_mode_maps,
            geometry_gate_maps,
        )
    )
    if not has_mode_grid and len(rows) != 3:
        raise ValueError(
            f"expected exactly three legacy variants; found {len(rows)}"
        )
    classification_thresholds = [float(row["classification_threshold"]) for row in rows]
    segmentation_thresholds = [float(row["segmentation_threshold"]) for row in rows]
    unique_classification = set(classification_thresholds)
    global_rows = [
        row
        for row, classification, segmentation, modes, geometry in zip(
            rows,
            source_maps,
            source_segmentation_maps,
            source_mode_maps,
            geometry_gate_maps,
        )
        if not classification and not segmentation and not modes and not geometry
    ]
    global_thresholds = {
        float(row["classification_threshold"]) for row in global_rows
    }
    if len(global_thresholds) != 3:
        raise ValueError("global classification thresholds must be distinct")
    if len(set(segmentation_thresholds)) != 1:
        raise ValueError("segmentation threshold must be frozen across variants")
    if has_mode_grid:
        modes = {str(row["deployment_mode"]) for row in rows}
        full_modes = {"refined", "seed", "seed_union"}
        if modes not in ({"refined"}, full_modes):
            raise ValueError(
                "deployment modes must be refined-only or exactly "
                f"{sorted(full_modes)}, "
                f"found {sorted(modes)}"
            )
        source_rows = [
            (row, classification, segmentation, source_modes, geometry)
            for row, classification, segmentation, source_modes, geometry in zip(
                rows,
                source_maps,
                source_segmentation_maps,
                source_mode_maps,
                geometry_gate_maps,
            )
            if classification or segmentation or source_modes or geometry
        ]
        canonical_source_maps = {
            json.dumps(
                {
                    "classification": classification,
                    "segmentation": segmentation,
                    "deployment_modes": source_modes,
                    "geometry_gates": geometry,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for _, classification, segmentation, source_modes, geometry in source_rows
        }
        source_points = len(canonical_source_maps)
        if has_source_grid and source_points not in set(range(1, 10)):
            raise ValueError(
                "source/geometry-calibrated grid must contain one to nine distinct operating points"
            )
        expected_variants = len(modes) * (3 + source_points)
        if len(rows) != expected_variants:
            raise ValueError(
                "expected three global variants per mode plus one to nine "
                "complete source/geometry-calibrated points per mode; "
                f"found {len(rows)} variants across {len(modes)} mode(s)"
            )
        combinations = {
            (float(row["classification_threshold"]), str(row["deployment_mode"]))
            for row, mapping in zip(rows, source_maps)
            if not mapping
        }
        expected_combinations = {
            (threshold, mode)
            for threshold in global_thresholds
            for mode in modes
        }
        if combinations != expected_combinations:
            raise ValueError("mode grid must contain every threshold/mode combination once")
        if has_source_grid:
            source_combinations = {
                (
                    json.dumps(
                        {
                            "classification": classification,
                            "segmentation": segmentation,
                            "deployment_modes": source_modes,
                            "geometry_gates": geometry,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    str(row["deployment_mode"]),
                )
                for row, classification, segmentation, source_modes, geometry in source_rows
            }
            expected_source_combinations = {
                (mapping, mode)
                for mapping in canonical_source_maps
                for mode in modes
            }
            if source_combinations != expected_source_combinations:
                raise ValueError(
                    "every source-calibrated mapping must occur once per output mode"
                )
    candidates: list[dict[str, object]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for row, source_mapping, source_segmentation_mapping, source_mode_mapping, geometry_gates in zip(
        rows,
        source_maps,
        source_segmentation_maps,
        source_mode_maps,
        geometry_gate_maps,
    ):
        variant = str(row["variant"])
        if not variant or variant in arrays:
            raise ValueError(f"missing or duplicate variant: {variant!r}")
        metrics_path = _resolve_metrics_path(summary, str(row["metrics"])).resolve(strict=True)
        metric_arrays = load_metric_csv(metrics_path, manifest_names)
        means = {metric: float(np.mean(metric_arrays[metric])) for metric in METRICS}
        stated = {
            "Lesion_DSC": float(row["DSC"]),
            "Lesion_NSD": float(row["NSD"]),
        }
        for metric in METRICS:
            if not math.isclose(means[metric], stated[metric], abs_tol=5e-4):
                raise ValueError(
                    f"summary mean mismatch for {variant} {metric}: "
                    f"csv={means[metric]:.8f} summary={stated[metric]:.8f}"
                )
        candidates.append(
            {
                "variant": variant,
                "classification_threshold": float(row["classification_threshold"]),
                "segmentation_threshold": float(row["segmentation_threshold"]),
                "deployment_mode": (
                    str(row["deployment_mode"]) if has_mode_grid else "refined"
                ),
                "source_classification_thresholds": source_mapping,
                "source_segmentation_thresholds": source_segmentation_mapping,
                "source_deployment_modes": source_mode_mapping,
                "geometry_gates": geometry_gates,
                "metrics": str(metrics_path),
                **means,
            }
        )
        arrays[variant] = metric_arrays
    return candidates, arrays


def stratified_bootstrap_indices(
    names: list[str], cohorts: dict[str, str], samples: int, seed: int
) -> np.ndarray:
    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(names):
        grouped[cohorts[name]].append(index)
    rng = np.random.default_rng(seed)
    parts = []
    for cohort in sorted(grouped):
        indices = np.asarray(grouped[cohort], dtype=np.int64)
        parts.append(rng.choice(indices, size=(samples, len(indices)), replace=True))
    return np.concatenate(parts, axis=1)


def interval(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, (0.025, 0.975))
    return {"low": float(low), "high": float(high)}


def bootstrap_selection_frequency(
    dsc_matrix: np.ndarray, nsd_matrix: np.ndarray
) -> np.ndarray:
    """Fractionally allocate exact metric ties instead of favoring row order."""

    if dsc_matrix.shape != nsd_matrix.shape or dsc_matrix.ndim != 2:
        raise ValueError("bootstrap metric matrices must have equal 2-D shapes")
    dsc_best = np.max(dsc_matrix, axis=1, keepdims=True)
    eligible = np.isclose(dsc_matrix, dsc_best, atol=1e-12, rtol=0)
    eligible_nsd = np.where(eligible, nsd_matrix, -np.inf)
    nsd_best = np.max(eligible_nsd, axis=1, keepdims=True)
    winners = eligible & np.isclose(nsd_matrix, nsd_best, atol=1e-12, rtol=0)
    weights = winners / winners.sum(axis=1, keepdims=True)
    return weights.mean(axis=0)


def variant_report(
    candidate: dict[str, object],
    arrays: dict[str, np.ndarray],
    bootstrap_indices: np.ndarray,
    names: list[str],
    cohorts: dict[str, str],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    boot = {
        metric: np.mean(values[bootstrap_indices], axis=1)
        for metric, values in arrays.items()
    }
    boot["combined"] = (boot["Lesion_DSC"] + boot["Lesion_NSD"]) / 2
    cohort_rows = []
    for cohort in sorted(set(cohorts.values())):
        indices = np.asarray(
            [index for index, name in enumerate(names) if cohorts[name] == cohort],
            dtype=np.int64,
        )
        dsc = arrays["Lesion_DSC"][indices]
        nsd = arrays["Lesion_NSD"][indices]
        cohort_rows.append(
            {
                "cohort": cohort,
                "cases": int(len(indices)),
                "Lesion_DSC": float(np.mean(dsc)),
                "Lesion_NSD": float(np.mean(nsd)),
                "combined": float(np.mean((dsc + nsd) / 2)),
                "zero_DSC_cases": int(np.count_nonzero(dsc == 0)),
            }
        )
    dsc = arrays["Lesion_DSC"]
    nsd = arrays["Lesion_NSD"]
    report = {
        **candidate,
        "combined": float(np.mean((dsc + nsd) / 2)),
        "bootstrap_95ci": {metric: interval(values) for metric, values in boot.items()},
        "case_distribution": {
            "median_DSC": float(np.median(dsc)),
            "median_NSD": float(np.median(nsd)),
            "zero_DSC_cases": int(np.count_nonzero(dsc == 0)),
            "DSC_below_0p25_cases": int(np.count_nonzero(dsc < 0.25)),
        },
        "cohorts": cohort_rows,
    }
    return report, boot


def summarize(
    summary: Path,
    checkpoint_selection: Path,
    manifest: Path,
    output: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    names, cohorts = load_manifest(manifest)
    candidates, arrays = load_candidates(summary, names)
    has_mode_grid = len(
        {str(candidate["deployment_mode"]) for candidate in candidates}
    ) > 1
    has_source_grid = any(
        bool(candidate["source_classification_thresholds"])
        or bool(candidate["source_segmentation_thresholds"])
        or bool(candidate["source_deployment_modes"])
        or bool(candidate["geometry_gates"])
        for candidate in candidates
    )
    if has_source_grid:
        has_mode_grid = True
    bootstrap_indices = stratified_bootstrap_indices(
        names, cohorts, bootstrap_samples, seed
    )
    reports: list[dict[str, object]] = []
    boots: dict[str, dict[str, np.ndarray]] = {}
    for candidate in candidates:
        variant = str(candidate["variant"])
        report, boot = variant_report(
            candidate, arrays[variant], bootstrap_indices, names, cohorts
        )
        reports.append(report)
        boots[variant] = boot
    ranked = sorted(
        reports,
        key=lambda row: (float(row["Lesion_DSC"]), float(row["Lesion_NSD"])),
        reverse=True,
    )
    best = ranked[0]
    variants = [str(candidate["variant"]) for candidate in candidates]
    dsc_matrix = np.column_stack([boots[variant]["Lesion_DSC"] for variant in variants])
    nsd_matrix = np.column_stack([boots[variant]["Lesion_NSD"] for variant in variants])
    bootstrap_frequencies = bootstrap_selection_frequency(dsc_matrix, nsd_matrix)
    stability = {
        variant: float(bootstrap_frequencies[index])
        for index, variant in enumerate(variants)
    }
    paired = None
    if len(ranked) > 1:
        chosen = str(best["variant"])
        runner = str(ranked[1]["variant"])
        paired = {
            "runner_up": runner,
            "mean_DSC_difference": float(
                float(best["Lesion_DSC"]) - float(ranked[1]["Lesion_DSC"])
            ),
            "DSC_difference_95ci": interval(
                boots[chosen]["Lesion_DSC"] - boots[runner]["Lesion_DSC"]
            ),
            "mean_NSD_difference": float(
                float(best["Lesion_NSD"]) - float(ranked[1]["Lesion_NSD"])
            ),
            "NSD_difference_95ci": interval(
                boots[chosen]["Lesion_NSD"] - boots[runner]["Lesion_NSD"]
            ),
        }
    checkpoint = json.loads(checkpoint_selection.read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "summary": str(summary.resolve()),
        "summary_sha256": file_sha256(summary),
        "checkpoint_selection_path": str(checkpoint_selection.resolve()),
        "checkpoint_selection_sha256": file_sha256(checkpoint_selection),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "selection_basis": (
            "case-held-out grouped full-volume OOF only; three patch-OOF-declared "
            "classification thresholds; "
            + (
                "three prespecified output modes; "
                if has_mode_grid
                else "legacy fixed refined output mode; "
            )
            + (
                "one to nine predeclared source-calibrated or geometry-calibrated operating points; "
                if has_source_grid
                else ""
            )
            + "no public or hidden labels"
        ),
        "cases": len(names),
        "cohorts": len(set(cohorts.values())),
        "bootstrap": {
            "method": "case bootstrap stratified by source cohort",
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "checkpoint_selection": checkpoint,
        # Backward-compatible deployment keys consumed by public/hidden scripts.
        "classification_threshold": best["classification_threshold"],
        "source_classification_thresholds": best[
            "source_classification_thresholds"
        ],
        "source_segmentation_thresholds": best[
            "source_segmentation_thresholds"
        ],
        "source_deployment_modes": best["source_deployment_modes"],
        "geometry_gates": best["geometry_gates"],
        "segmentation_threshold": best["segmentation_threshold"],
        "deployment_mode": best["deployment_mode"],
        "Lesion_DSC": best["Lesion_DSC"],
        "Lesion_NSD": best["Lesion_NSD"],
        "metrics": best["metrics"],
        "variant": best["variant"],
        "selected": best,
        "all_variants": reports,
        "bootstrap_selection_frequency": stability,
        "paired_vs_runner_up": paired,
        "calibration_caveat": (
            "The winning threshold is selected on these OOF cases; confidence "
            "intervals quantify case sampling uncertainty but not this bounded "
            "threshold/mode selection effect or public-to-hidden domain shift."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    payload = summarize(
        args.summary,
        args.checkpoint_selection,
        args.manifest,
        args.output,
        args.bootstrap_samples,
        args.seed,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
