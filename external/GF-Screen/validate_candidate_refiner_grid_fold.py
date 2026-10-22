#!/usr/bin/env python3
"""Validate one completed fold from the shared candidate-refiner materializer.

The materializer writes its terminal marker last. A restart may otherwise see
an old marker while masks or the streamed component audit are being rewritten.
This validator binds the marker to the current configuration, exact fold case
set, base deployment, audit counts, and artifact modification order.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


GATES = (
    "min_seed_coverage",
    "min_refined_overlap",
    "min_volume_ratio",
    "max_volume_ratio",
)


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve()


def _load_configurations(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configurations = payload.get("configurations", payload)
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("configuration grid is empty")
    tags = [str(item.get("tag", "")) for item in configurations]
    if any(not tag for tag in tags) or len(tags) != len(set(tags)):
        raise ValueError("configuration tags are empty or duplicated")
    return configurations


def validate_fold(
    configurations: Path,
    output_root: Path,
    fold: int,
    expected_dir: Path,
    base: Path,
    segmentation_threshold: float,
    empty_base: bool,
    gates: dict[str, float],
) -> dict:
    configs = _load_configurations(configurations)
    expected_names = {path.name for path in expected_dir.glob("*.nii.gz")}
    if not expected_names:
        raise ValueError(f"no expected cases in {expected_dir}")
    result = {"fold": fold, "cases": len(expected_names), "configurations": []}
    expected_audit_cases = {
        name.removesuffix(".nii.gz") for name in expected_names
    }
    for config in configs:
        tag = str(config["tag"])
        directory = output_root / tag / f"fold{fold}"
        marker_path = directory / "CANDIDATE_REFINER_COMPLETE.json"
        if not marker_path.is_file():
            raise ValueError(f"missing marker: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        outputs = list(directory.glob("*.nii.gz"))
        output_names = {path.name for path in outputs if path.is_file()}
        if output_names != expected_names or len(outputs) != len(expected_names):
            raise ValueError(
                f"output coverage mismatch for {tag}/fold{fold}: "
                f"{len(output_names)}/{len(expected_names)}"
            )
        if any(path.stat().st_size <= 0 for path in outputs):
            raise ValueError(f"empty NIfTI output for {tag}/fold{fold}")

        audit_path = _resolved(marker.get("audit", ""))
        expected_audit = directory.with_name(
            directory.name + ".components.csv"
        ).resolve()
        if audit_path != expected_audit or not audit_path.is_file():
            raise ValueError(f"audit path mismatch for {tag}/fold{fold}")
        rows = accepted = 0
        audit_cases: set[str] = set()
        with audit_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"case", "accepted"} <= set(
                reader.fieldnames
            ):
                raise ValueError(f"invalid audit header for {tag}/fold{fold}")
            for row in reader:
                rows += 1
                accepted += int(row["accepted"])
                audit_cases.add(str(row["case"]))
        if audit_cases - expected_audit_cases:
            raise ValueError(f"unexpected audit case for {tag}/fold{fold}")
        if int(marker.get("candidates", -1)) != rows:
            raise ValueError(f"candidate count mismatch for {tag}/fold{fold}")
        if int(marker.get("accepted", -1)) != accepted:
            raise ValueError(f"accepted count mismatch for {tag}/fold{fold}")

        expected_segmentation = float(
            config.get("segmentation_threshold")
            if config.get("segmentation_threshold") is not None
            else segmentation_threshold
        )
        expected_fields = {
            "cases": len(expected_names),
            "classification_threshold": float(config["classification_threshold"]),
            "source_classification_thresholds": dict(
                config.get("source_classification_thresholds", {})
            ),
            "segmentation_threshold": expected_segmentation,
            "fallback_segmentation_threshold": float(segmentation_threshold),
            "source_segmentation_thresholds": dict(
                config.get("source_segmentation_thresholds", {})
            ),
            "source_deployment_modes": dict(
                config.get("source_deployment_modes", {})
            ),
            "mode": str(config["mode"]),
            "empty_base": bool(empty_base),
            "component_policy": "source_separate",
            "gates": gates,
            "materializer": "shared_case_grid_v1",
        }
        for name, expected in expected_fields.items():
            if marker.get(name) != expected:
                raise ValueError(
                    f"marker field mismatch for {tag}/fold{fold}: {name} "
                    f"{marker.get(name)!r} != {expected!r}"
                )
        if _resolved(marker.get("base", "")) != base.resolve():
            raise ValueError(f"base mismatch for {tag}/fold{fold}")
        if _resolved(marker.get("configuration_grid", "")) != configurations.resolve():
            raise ValueError(f"configuration path mismatch for {tag}/fold{fold}")

        marker_mtime = marker_path.stat().st_mtime_ns
        latest_payload_mtime = max(
            audit_path.stat().st_mtime_ns,
            *(path.stat().st_mtime_ns for path in outputs),
        )
        if marker_mtime < latest_payload_mtime:
            raise ValueError(
                f"stale marker predates rewritten payload for {tag}/fold{fold}"
            )
        result["configurations"].append(
            {"tag": tag, "candidates": rows, "accepted": accepted}
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configurations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--expected-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--segmentation-threshold", type=float, required=True)
    parser.add_argument("--empty-base", action="store_true")
    parser.add_argument("--min-seed-coverage", type=float, required=True)
    parser.add_argument("--min-refined-overlap", type=float, required=True)
    parser.add_argument("--min-volume-ratio", type=float, required=True)
    parser.add_argument("--max-volume-ratio", type=float, required=True)
    args = parser.parse_args()
    result = validate_fold(
        args.configurations,
        args.output_root,
        args.fold,
        args.expected_dir,
        args.base,
        args.segmentation_threshold,
        args.empty_base,
        {name: float(getattr(args, name)) for name in GATES},
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
