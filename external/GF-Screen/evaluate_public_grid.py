#!/usr/bin/env python3
"""Score many prediction directories with one persistent case-worker pool.

Each worker loads a public ground-truth volume once and evaluates every missing
candidate for that case with the unchanged organizer metric. This preserves
exact per-candidate CSVs while avoiding repeated label I/O and process startup.
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CODE = PROJECT_ROOT / "handoff_bundle" / "code"
sys.path.insert(0, str(OFFICIAL_CODE))
from evaluate import score_case  # noqa: E402


def parse_candidates(values: list[str]) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"candidate must be TAG=PATH, got {value}")
        tag, raw_path = value.split("=", 1)
        if not tag or tag in candidates:
            raise ValueError(f"invalid or duplicate candidate tag: {tag}")
        candidates[tag] = Path(raw_path)
    return candidates


def csv_is_complete(path: Path, expected_names: list[str]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return False
    return [row.get("Name") for row in rows] == [
        name.removesuffix(".nii.gz") for name in expected_names
    ]


def score_case_grid(
    item: tuple[str, str, tuple[tuple[str, str], ...]],
) -> tuple[str, list[tuple[str, float, float, int]]]:
    name, gt_path, candidates = item
    gt_image = nib.load(gt_path)
    target = np.uint8(gt_image.get_fdata())
    spacing = gt_image.header.get_zooms()[:3]
    values: list[tuple[str, float, float, int]] = []
    for tag, directory in candidates:
        prediction_path = Path(directory) / name
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        prediction = np.uint8(nib.load(prediction_path).get_fdata())
        if prediction.shape != target.shape:
            raise ValueError(
                f"shape mismatch for {tag}/{name}: {prediction.shape} != {target.shape}"
            )
        dsc, nsd = score_case(target, prediction, spacing)
        values.append((tag, float(dsc), float(nsd), int((prediction > 0).sum())))
    return name.removesuffix(".nii.gz"), values


def evaluate_grid(
    gt_dir: Path,
    candidates: dict[str, Path],
    output_dir: Path,
    workers: int,
) -> dict[str, Path]:
    names = sorted(path.name for path in gt_dir.glob("*.nii.gz"))
    if not names:
        raise FileNotFoundError(f"no ground truth in {gt_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {tag: output_dir / f"{tag}.metrics.csv" for tag in candidates}
    missing = {
        tag: directory
        for tag, directory in candidates.items()
        if not csv_is_complete(outputs[tag], names)
    }
    if not missing:
        print(f"all {len(candidates)} candidate metric CSVs are complete")
        return outputs

    for tag, directory in missing.items():
        absent = [name for name in names if not (directory / name).is_file()]
        if absent:
            raise FileNotFoundError(
                f"{tag} is missing {len(absent)} predictions; first={absent[0]}"
            )
    packed = tuple((tag, str(path)) for tag, path in missing.items())
    items = [(name, str(gt_dir / name), packed) for name in names]
    rows = {tag: [] for tag in missing}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, (case, values) in enumerate(
            executor.map(score_case_grid, items), start=1
        ):
            for tag, dsc, nsd, foreground in values:
                rows[tag].append((case, dsc, nsd, foreground))
            print(f"[{index:03d}/{len(names):03d}] {case}", flush=True)

    for tag, candidate_rows in rows.items():
        path = outputs[tag]
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD", "pred_fg_voxels"))
            for case, dsc, nsd, foreground in candidate_rows:
                writer.writerow((case, round(dsc, 4), round(nsd, 4), foreground))
        temporary.replace(path)
        dsc_mean = float(np.mean([row[1] for row in candidate_rows]))
        nsd_mean = float(np.mean([row[2] for row in candidate_rows]))
        print(f"{tag}\tDSC={dsc_mean:.4f}\tNSD={nsd_mean:.4f}\t{path}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    evaluate_grid(
        args.gt,
        parse_candidates(args.candidate),
        args.output_dir,
        args.workers,
    )


if __name__ == "__main__":
    main()
