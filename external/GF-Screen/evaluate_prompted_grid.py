#!/usr/bin/env python3
"""Score a frozen prompt-confirmation transform without writing every mask.

Each case worker loads its label and prompt support once, applies the same
BiomedParse fusion function to every candidate mask, and emits the organizer's
unchanged DSC/NSD values.  Only the eventual winner needs NIfTI materialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "handoff_bundle" / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "BiomedParse"))
from evaluate import score_case  # noqa: E402
from biomedparse_fusion import apply_config  # noqa: E402


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


def load_bool(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(bool), image


def score_prompted_case(
    item: tuple[str, str, str, tuple[tuple[str, str], ...], dict[str, object]],
) -> tuple[str, list[tuple[str, float, float, int]]]:
    name, gt_path, support_path, candidates, config = item
    target, target_image = load_bool(Path(gt_path))
    support, support_image = load_bool(Path(support_path))
    if target.shape != support.shape or not np.allclose(
        target_image.affine, support_image.affine, atol=1e-3
    ):
        raise ValueError(f"support geometry mismatch: {name}")
    spacing = tuple(float(value) for value in target_image.header.get_zooms()[:3])
    values: list[tuple[str, float, float, int]] = []
    for tag, directory in candidates:
        base_path = Path(directory) / name
        base, base_image = load_bool(base_path)
        if base.shape != target.shape or not np.allclose(
            base_image.affine, target_image.affine, atol=1e-3
        ):
            raise ValueError(f"candidate geometry mismatch: {tag}/{name}")
        prediction = apply_config(base, support, spacing, config)
        dsc, nsd = score_case(
            target.astype(np.uint8), prediction.astype(np.uint8), spacing
        )
        values.append((tag, float(dsc), float(nsd), int(prediction.sum())))
    return name.removesuffix(".nii.gz"), values


def evaluate_prompted_grid(
    gt_dir: Path,
    support_dir: Path,
    config: dict[str, object],
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
        print(f"all {len(candidates)} prompted metric CSVs are complete")
        return outputs
    for name in names:
        if not (support_dir / name).is_file():
            raise FileNotFoundError(support_dir / name)
    for tag, directory in missing.items():
        absent = [name for name in names if not (directory / name).is_file()]
        if absent:
            raise FileNotFoundError(f"{tag}: missing {len(absent)} masks; first={absent[0]}")

    packed = tuple((tag, str(path)) for tag, path in missing.items())
    items = [
        (name, str(gt_dir / name), str(support_dir / name), packed, config)
        for name in names
    ]
    rows = {tag: [] for tag in missing}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, (case, values) in enumerate(executor.map(score_prompted_case, items), 1):
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
        print(
            f"{tag}\tDSC={np.mean([row[1] for row in candidate_rows]):.4f}"
            f"\tNSD={np.mean([row[2] for row in candidate_rows]):.4f}\t{path}"
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    evaluate_prompted_grid(
        args.gt,
        args.support,
        json.loads(args.config.read_text()),
        parse_candidates(args.candidate),
        args.output_dir,
        args.workers,
    )


if __name__ == "__main__":
    main()
