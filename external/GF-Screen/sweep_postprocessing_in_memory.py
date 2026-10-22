#!/usr/bin/env python3
"""Score GF-Screen fusion/component-filter candidates while loading each case once."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


THRESHOLDS = (0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a", type=Path, required=True)
    parser.add_argument("--pred-b", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--source-a", required=True)
    parser.add_argument("--source-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(data), (0, 1)).all():
        raise ValueError(f"non-binary mask: {path}")
    return image, data.astype(bool, copy=False)


def component_masks(mask: np.ndarray, voxel_volume: float) -> list[np.ndarray]:
    if not mask.any():
        return [mask] * len(THRESHOLDS)
    components, count = ndimage.label(mask)
    sizes = np.bincount(components.ravel(), minlength=count + 1) * voxel_volume
    masks = [mask]
    for threshold in THRESHOLDS[1:]:
        keep = sizes >= threshold
        keep[0] = False
        masks.append(keep[components])
    return masks


def score_one(task: tuple[str, str, str, str, str]) -> tuple[str, list[tuple[float, float]]]:
    name, pred_a_dir, pred_b_dir, gt_dir, evaluator_dir = task
    sys.path.insert(0, evaluator_dir)
    from evaluate import score_case  # pylint: disable=import-outside-toplevel

    image_a, mask_a = binary(Path(pred_a_dir) / name)
    image_b, mask_b = binary(Path(pred_b_dir) / name)
    gt_image = nib.load(str(Path(gt_dir) / name))
    gt = np.asanyarray(gt_image.dataobj).astype(np.uint8, copy=False)
    if image_a.shape != image_b.shape or image_a.shape != gt_image.shape:
        raise ValueError(f"shape mismatch for {name}")
    if not np.allclose(image_a.affine, image_b.affine) or not np.allclose(image_a.affine, gt_image.affine):
        raise ValueError(f"affine mismatch for {name}")

    spacing = gt_image.header.get_zooms()[:3]
    voxel_volume = float(abs(np.linalg.det(image_a.affine[:3, :3])))
    candidates = component_masks(mask_a, voxel_volume)
    candidates.extend(component_masks(mask_a | mask_b, voxel_volume))
    candidates.append(mask_a & mask_b)
    scores = [score_case(gt, mask.astype(np.uint8, copy=False), spacing) for mask in candidates]
    return name, scores


def main() -> None:
    args = parse_args()
    names = sorted(path.name for path in args.gt.glob("*.nii.gz"))
    if not names:
        raise ValueError(f"no labels in {args.gt}")
    for name in names:
        if not (args.pred_a / name).exists() or not (args.pred_b / name).exists():
            raise ValueError(f"missing prediction for {name}")

    tasks = [
        (name, str(args.pred_a), str(args.pred_b), str(args.gt), str(args.evaluator_dir))
        for name in names
    ]
    all_scores: list[list[tuple[float, float]]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, scores) in enumerate(pool.map(score_one, tasks), start=1):
            all_scores.append(scores)
            print(f"[{index:03d}/{len(names):03d}] scored {name}", flush=True)

    score_array = np.asarray(all_scores, dtype=np.float64)
    rows: list[tuple[str, float, float, str, str, str, float]] = []
    for index, threshold in enumerate(THRESHOLDS):
        rows.append((
            f"single_{args.source_a}_mm{int(threshold)}",
            float(score_array[:, index, 0].mean()),
            float(score_array[:, index, 1].mean()),
            "a", args.source_a, "", threshold,
        ))
    offset = len(THRESHOLDS)
    for index, threshold in enumerate(THRESHOLDS):
        rows.append((
            f"union_{args.source_a}_{args.source_b}_mm{int(threshold)}",
            float(score_array[:, offset + index, 0].mean()),
            float(score_array[:, offset + index, 1].mean()),
            "union", args.source_a, args.source_b, threshold,
        ))
    rows.append((
        f"intersection_{args.source_a}_{args.source_b}_mm0",
        float(score_array[:, -1, 0].mean()),
        float(score_array[:, -1, 1].mean()),
        "intersection", args.source_a, args.source_b, 0.0,
    ))

    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / "summary.tsv"
    with summary.open("w", encoding="utf-8") as stream:
        stream.write("configuration\tDSC\tNSD\tfusion\tsource_a\tsource_b\tmin_mm3\n")
        for name, dsc, nsd, fusion, source_a, source_b, threshold in rows:
            stream.write(
                f"{name}\t{dsc:.4f}\t{nsd:.4f}\t{fusion}\t{source_a}\t{source_b}\t{threshold:g}\n"
            )
    best = max(rows, key=lambda row: row[1] + row[2])
    with (args.output / "best.tsv").open("w", encoding="utf-8") as stream:
        name, dsc, nsd, fusion, source_a, source_b, threshold = best
        stream.write(
            f"{name}\t{dsc:.4f}\t{nsd:.4f}\t{fusion}\t{source_a}\t{source_b}\t{threshold:g}\n"
        )
    print(f"wrote {summary}")
    print("best:", (args.output / "best.tsv").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
