#!/usr/bin/env python3
"""Score timeout-aware GF overlap policies on the fixed public 50."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


HERE = Path(__file__).resolve().parent
GF_ROOT = HERE.parent
PROJECT = GF_ROOT.parents[1]
sys.path.insert(0, str(GF_ROOT))
from fast_official_lesion_metrics import score_case  # noqa: E402
from docker_submission.winner_mask_recipe import filter_components_physical_6  # noqa: E402


THRESHOLDS = (750, 1000, 1500, 2000)
SELECTED_THRESHOLDS = (200, 250, 300, 350, 400)


def evaluate(item: tuple[str, str, str, str, int, int]):
    name, label_dir, overlap75_dir, overlap625_dir, grid_crops, selected_crops = item
    target_image = nib.load(str(Path(label_dir) / name))
    target = np.asanyarray(target_image.dataobj).astype(np.uint8, copy=False)
    spacing = tuple(float(v) for v in target_image.header.get_zooms()[:3])
    raw75 = np.asanyarray(nib.load(str(Path(overlap75_dir) / name)).dataobj) > 0
    mask75 = filter_components_physical_6(raw75, spacing, 1000.0)
    mask625 = np.asanyarray(nib.load(str(Path(overlap625_dir) / name)).dataobj) > 0
    masks = {"overlap_075": mask75, "overlap_0625": mask625}
    for threshold in THRESHOLDS:
        masks[f"adaptive_grid_gt_{threshold}"] = mask625 if grid_crops > threshold else mask75
    for threshold in SELECTED_THRESHOLDS:
        masks[f"adaptive_selected_gt_{threshold}"] = (
            mask625 if grid_crops > 2000 or selected_crops > threshold else mask75
        )
    scores = {
        key: score_case(target, mask.astype(np.uint8, copy=False), spacing)
        for key, mask in masks.items()
    }
    return name, grid_crops, selected_crops, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap625", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    labels = PROJECT / "validation/Validation-Public-Labels"
    overlap75 = PROJECT / "external/GF-Screen/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
    audit = PROJECT / "external/GF-Screen/analysis/glance_focus_stage_recall_20260827/cases"
    names = sorted(path.name for path in labels.glob("*.nii.gz"))
    if len(names) != 50:
        raise RuntimeError(f"expected fixed public 50, got {len(names)}")
    items = []
    for name in names:
        case = name.removesuffix(".nii.gz")
        metadata = json.loads((audit / f"{case}.json").read_text(encoding="utf-8"))
        items.append((
            name, str(labels), str(overlap75), str(args.overlap625),
            int(metadata["grid_crops"]), int(metadata["bbox_selected_crops"]),
        ))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(evaluate, items))
    keys = list(rows[0][3])
    def uses_overlap625(key: str, row) -> bool:
        if key == "overlap_0625":
            return True
        if key == "overlap_075":
            return False
        threshold = int(key.rsplit("_", 1)[1])
        if key.startswith("adaptive_grid"):
            return row[1] > threshold
        return row[1] > 2000 or row[2] > threshold
    metrics = {
        key: {
            "Lesion_DSC": float(np.mean([row[3][key][0] for row in rows])),
            "Lesion_NSD": float(np.mean([row[3][key][1] for row in rows])),
            "overlap625_cases": int(sum(uses_overlap625(key, row) for row in rows)),
        }
        for key in keys
    }
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "role": "fixed_public_timeout_aware_overlap_ablation",
        "cases": len(rows),
        "metrics": metrics,
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
