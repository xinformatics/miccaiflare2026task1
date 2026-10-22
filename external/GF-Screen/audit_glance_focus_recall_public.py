#!/usr/bin/env python3
"""Attribute fixed-public lesion misses across Glance, Focus, and postprocessing.

This is a public-label diagnostic only.  It replays the exact deployed bbox
and CBG Glance models on the exact submission preprocessing/grid, converts the
maximum crop-selection probabilities back to native image geometry, and joins
them to preserved bbox-Glance, all-crop Focus, and anchor masks.

No prediction or submission mask is created.  The output is evidence for
choosing whether the next trained intervention belongs in Glance or Focus.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import inspect
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from scipy import ndimage

from docker_submission.predict_dual_glance_shared_focus import (
    BBOX_COMPACT_SHA,
    CBG_COMPACT_SHA,
    checkpoint_payload,
    transforms,
)
from models.glance import Glance_model


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)
BBOX_THRESHOLD = 0.50
CBG_THRESHOLD = 0.20
DETECTION_FRACTION = 0.10
SIZE_BINS = (
    ("lt100", 0.0, 100.0),
    ("100to500", 100.0, 500.0),
    ("500to1000", 500.0, 1000.0),
    ("1000to3000", 1000.0, 3000.0),
    ("ge3000", 3000.0, float("inf")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def size_bin(volume_mm3: float) -> str:
    for name, low, high in SIZE_BINS:
        if low <= volume_mm3 < high:
            return name
    raise AssertionError(volume_mm3)


def dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    denominator = int(reference.sum()) + int(prediction.sum())
    return 1.0 if denominator == 0 else 2.0 * int((reference & prediction).sum()) / denominator


def overlap_fraction(component: np.ndarray, prediction: np.ndarray) -> float:
    count = int(component.sum())
    return 1.0 if count == 0 else float(prediction[component].mean())


def score_volume(
    shape: tuple[int, int, int],
    coordinates: list[tuple[int, int, int]],
    probabilities: np.ndarray,
    crop_shape: tuple[int, int, int] = (96, 96, 64),
) -> np.ndarray:
    if len(coordinates) != len(probabilities):
        raise ValueError("coordinate/probability count mismatch")
    result = np.zeros(shape, dtype=np.float32)
    for (x, y, z), probability in zip(coordinates, probabilities):
        region = result[x : x + crop_shape[0], y : y + crop_shape[1], z : z + crop_shape[2]]
        np.maximum(region, float(probability), out=region)
    return result


def restore_score(score: np.ndarray, batch: dict, inverse) -> np.ndarray:
    payload = dict(batch)
    payload["pred"] = torch.from_numpy(score).unsqueeze(0).unsqueeze(0)
    restored = inverse(decollate_batch(payload)[0])["pred"]
    return restored.detach().cpu().numpy().squeeze().astype(np.float32, copy=False)


def load_glance_models(bbox_path: Path, cbg_path: Path, device: torch.device):
    if sha256_file(bbox_path) != BBOX_COMPACT_SHA:
        raise RuntimeError("bbox deployment checkpoint identity mismatch")
    if sha256_file(cbg_path) != CBG_COMPACT_SHA:
        raise RuntimeError("CBG deployment checkpoint identity mismatch")
    bbox_payload, bbox_full_state = checkpoint_payload(bbox_path)
    cbg_payload, cbg_state = checkpoint_payload(cbg_path)
    if (
        bbox_payload.get("role") != "bbox_glance_and_shared_focus"
        or cbg_payload.get("role") != "cbg_glance_only"
    ):
        raise RuntimeError("unexpected compact deployment checkpoint roles")
    bbox_state = {
        key.removeprefix("glance_model."): value
        for key, value in bbox_full_state.items()
        if key.startswith("glance_model.")
    }
    if len(bbox_state) != 104 or len(cbg_state) != 104:
        raise RuntimeError("unexpected Glance tensor inventory")
    bbox = Glance_model(backbone="resnet18", grid_mode="legacy")
    cbg = Glance_model(backbone="resnet18", grid_mode="legacy")
    bbox.load_state_dict(bbox_state, strict=True)
    cbg.load_state_dict(cbg_state, strict=True)
    del bbox_payload, cbg_payload, bbox_full_state, bbox_state, cbg_state
    return bbox.eval().to(device), cbg.eval().to(device)


def native_mask(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
        raise ValueError(f"geometry mismatch: {path}")
    return np.asanyarray(image.dataobj) > 0


def case_name(image_path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not image_path.name.endswith(suffix):
        raise ValueError(f"unexpected public image name: {image_path.name}")
    return image_path.name[: -len(suffix)] + ".nii.gz"


def run_inference(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact Glance replay")
    if args.num_parts < 1 or not 0 <= args.part_id < args.num_parts:
        raise ValueError("invalid deterministic shard")
    images = sorted(args.images.glob("*_0000.nii.gz"))
    if len(images) != 50:
        raise RuntimeError(f"expected exact public 50, found {len(images)}")
    for directory in (args.labels, args.bbox_raw, args.allcrop_focus, args.anchor):
        if len(list(directory.glob("*.nii.gz"))) != 50:
            raise RuntimeError(f"expected 50 masks in {directory}")
    selected_images = images[args.part_id :: args.num_parts]
    forward, inverse = transforms(False)
    dataset = Dataset(
        data=[{"image": str(path), "source": str(path)} for path in selected_images],
        transform=forward,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda", 0)
    bbox, cbg = load_glance_models(args.bbox_checkpoint, args.cbg_checkpoint, device)
    torch.backends.cudnn.benchmark = True
    case_root = args.output / "cases"
    case_root.mkdir(parents=True, exist_ok=True)

    for index, (image_path, batch) in enumerate(zip(selected_images, loader), start=1):
        name = case_name(image_path)
        destination = case_root / f"{name.removesuffix('.nii.gz')}.json"
        if destination.is_file() and not args.overwrite:
            continue
        data = batch["image"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            crops, coordinates_all, bbox_logits = bbox(data)
            cbg_logits = cbg.get_actions(crops)
            bbox_probability = bbox_logits.softmax(1)[:, 1].float().cpu().numpy()
            cbg_probability = cbg_logits.softmax(1)[:, 1].float().cpu().numpy()
        coordinates = coordinates_all[0]
        transformed_shape = tuple(int(value) for value in data.shape[2:])
        bbox_score = score_volume(transformed_shape, coordinates, bbox_probability)
        cbg_score = score_volume(transformed_shape, coordinates, cbg_probability)
        bbox_native_score = restore_score(bbox_score, batch, inverse)
        cbg_native_score = restore_score(cbg_score, batch, inverse)
        del crops, bbox_logits, cbg_logits, data, bbox_score, cbg_score
        torch.cuda.empty_cache()

        label_path = args.labels / name
        reference = nib.load(str(label_path))
        target = np.asanyarray(reference.dataobj) > 0
        if bbox_native_score.shape != target.shape or cbg_native_score.shape != target.shape:
            raise ValueError(f"restored Glance geometry mismatch for {name}")
        bbox_coverage = bbox_native_score > BBOX_THRESHOLD
        cbg_coverage = cbg_native_score > CBG_THRESHOLD
        dual_coverage = bbox_coverage | cbg_coverage
        bbox_raw = native_mask(args.bbox_raw / name, reference)
        allcrop_focus = native_mask(args.allcrop_focus / name, reference)
        anchor = native_mask(args.anchor / name, reference)
        component_map, component_count = ndimage.label(target, structure=CONNECTIVITY_26)
        spacing = np.asarray(reference.header.get_zooms()[:3], dtype=np.float64)
        voxel_mm3 = float(np.prod(spacing))
        lesion_rows = []
        for component_id in range(1, component_count + 1):
            component = component_map == component_id
            voxels = int(component.sum())
            volume_mm3 = voxels * voxel_mm3
            fractions = {
                "bbox_glance_coverage": overlap_fraction(component, bbox_coverage),
                "cbg_glance_coverage": overlap_fraction(component, cbg_coverage),
                "dual_glance_coverage": overlap_fraction(component, dual_coverage),
                "bbox_focus_overlap": overlap_fraction(component, bbox_raw),
                "allcrop_focus_overlap": overlap_fraction(component, allcrop_focus),
                "anchor_overlap": overlap_fraction(component, anchor),
            }
            anchor_dice = dice(target, anchor)
            recovered = anchor | component
            lesion_rows.append(
                {
                    "case": name,
                    "component_id": component_id,
                    "voxels": voxels,
                    "volume_mm3": volume_mm3,
                    "size_bin": size_bin(volume_mm3),
                    "bbox_max_probability": float(bbox_native_score[component].max(initial=0.0)),
                    "cbg_max_probability": float(cbg_native_score[component].max(initial=0.0)),
                    **fractions,
                    "bbox_glance_detected": fractions["bbox_glance_coverage"] >= DETECTION_FRACTION,
                    "cbg_glance_detected": fractions["cbg_glance_coverage"] >= DETECTION_FRACTION,
                    "dual_glance_detected": fractions["dual_glance_coverage"] >= DETECTION_FRACTION,
                    "bbox_focus_detected": fractions["bbox_focus_overlap"] >= DETECTION_FRACTION,
                    "allcrop_focus_detected": fractions["allcrop_focus_overlap"] >= DETECTION_FRACTION,
                    "anchor_detected": fractions["anchor_overlap"] >= DETECTION_FRACTION,
                    "glance_recoverable_by_allcrop": (
                        fractions["bbox_focus_overlap"] < DETECTION_FRACTION
                        and fractions["allcrop_focus_overlap"] >= DETECTION_FRACTION
                    ),
                    "focus_missed_with_all_crops": fractions["allcrop_focus_overlap"] < DETECTION_FRACTION,
                    "postprocess_lost_after_bbox_focus": (
                        fractions["bbox_focus_overlap"] >= DETECTION_FRACTION
                        and fractions["anchor_overlap"] < DETECTION_FRACTION
                    ),
                    "perfect_addition_macro_dsc_gain": (dice(target, recovered) - anchor_dice) / 50.0,
                }
            )
        payload = {
            "schema_version": 1,
            "status": "COMPLETE",
            "case": name,
            "grid_crops": len(coordinates),
            "bbox_selected_crops": int((bbox_probability > BBOX_THRESHOLD).sum()),
            "cbg_selected_crops": int((cbg_probability > CBG_THRESHOLD).sum()),
            "dual_selected_crops": int(
                np.count_nonzero(
                    (bbox_probability > BBOX_THRESHOLD) | (cbg_probability > CBG_THRESHOLD)
                )
            ),
            "native_gt_components": int(component_count),
            "native_gt_voxels": int(target.sum()),
            "anchor_case_dice": dice(target, anchor),
            "lesions": lesion_rows,
            "public_images_read": True,
            "public_labels_read": True,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
        }
        atomic_json(destination, payload)
        print(
            f"part {args.part_id}: {index}/{len(selected_images)} {name} "
            f"crops={len(coordinates)} lesions={component_count}",
            flush=True,
        )


def category_mask(row: dict, category: str) -> bool:
    if category == "bbox_glance_missed":
        return not row["bbox_glance_detected"]
    if category == "dual_glance_missed":
        return not row["dual_glance_detected"]
    if category == "glance_recoverable_by_allcrop":
        return bool(row["glance_recoverable_by_allcrop"])
    if category == "focus_missed_with_all_crops":
        return bool(row["focus_missed_with_all_crops"])
    if category == "postprocess_lost_after_bbox_focus":
        return bool(row["postprocess_lost_after_bbox_focus"])
    if category == "anchor_missed":
        return not row["anchor_detected"]
    raise KeyError(category)


def summarize(args: argparse.Namespace) -> None:
    case_paths = sorted((args.output / "cases").glob("FLARE23Ts_*.json"))
    if len(case_paths) != 50:
        raise RuntimeError(f"expected 50 complete case records, found {len(case_paths)}")
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_paths]
    if any(case.get("status") != "COMPLETE" for case in cases):
        raise RuntimeError("incomplete Glance audit case record")
    lesions = [row for case in cases for row in case["lesions"]]
    if len(lesions) != 239:
        raise RuntimeError(f"expected organizer 239 public lesions, found {len(lesions)}")
    categories = (
        "bbox_glance_missed",
        "dual_glance_missed",
        "glance_recoverable_by_allcrop",
        "focus_missed_with_all_crops",
        "postprocess_lost_after_bbox_focus",
        "anchor_missed",
    )
    summary = {}
    for category in categories:
        selected = [row for row in lesions if category_mask(row, category)]
        by_size = {}
        for name, _, _ in SIZE_BINS:
            band = [row for row in selected if row["size_bin"] == name]
            by_size[name] = {
                "lesions": len(band),
                "volume_mm3": float(sum(row["volume_mm3"] for row in band)),
                "summed_individual_perfect_addition_macro_dsc_gain": float(
                    sum(row["perfect_addition_macro_dsc_gain"] for row in band)
                ),
            }
        summary[category] = {
            "lesions": len(selected),
            "volume_mm3": float(sum(row["volume_mm3"] for row in selected)),
            "summed_individual_perfect_addition_macro_dsc_gain": float(
                sum(row["perfect_addition_macro_dsc_gain"] for row in selected)
            ),
            "by_size": by_size,
        }
    fields = list(lesions[0])
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "lesions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lesions)
    worst = sorted(
        (row for row in lesions if not row["anchor_detected"]),
        key=lambda row: row["perfect_addition_macro_dsc_gain"],
        reverse=True,
    )[:30]
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision_role": "PUBLIC_DIAGNOSTIC_ONLY",
        "cases": len(cases),
        "lesions": len(lesions),
        "thresholds": {
            "bbox_glance": BBOX_THRESHOLD,
            "cbg_glance": CBG_THRESHOLD,
            "lesion_detection_fraction": DETECTION_FRACTION,
            "allcrop_focus_mask_threshold": 0.55,
        },
        "mean_crop_counts": {
            key: float(np.mean([case[key] for case in cases]))
            for key in ("grid_crops", "bbox_selected_crops", "cbg_selected_crops", "dual_selected_crops")
        },
        "stage_attribution": summary,
        "highest_value_anchor_misses": worst,
        "assets": {
            "bbox_checkpoint": {"path": str(args.bbox_checkpoint.resolve()), "sha256": sha256_file(args.bbox_checkpoint)},
            "cbg_checkpoint": {"path": str(args.cbg_checkpoint.resolve()), "sha256": sha256_file(args.cbg_checkpoint)},
            "bbox_raw": str(args.bbox_raw.resolve()),
            "allcrop_focus": str(args.allcrop_focus.resolve()),
            "anchor": str(args.anchor.resolve()),
        },
        "interpretation_contract": {
            "bbox_glance_missed": "GT lesion has <10% coverage by crops selected by bbox Glance at p>0.50.",
            "dual_glance_missed": "GT lesion has <10% coverage after bbox p>0.50 OR CBG p>0.20.",
            "glance_recoverable_by_allcrop": "All-crop Focus detects lesion but deployed bbox-Glance Focus output does not.",
            "focus_missed_with_all_crops": "Focus at lesion p>=0.55 misses lesion even when every crop is evaluated.",
            "postprocess_lost_after_bbox_focus": "Deployed bbox-Glance Focus detects lesion but the current anchor does not.",
        },
        "public_images_read": True,
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output / "summary.json", payload)
    print(json.dumps(payload, indent=2))


def add_common(parser: argparse.ArgumentParser) -> None:
    project = APP_ROOT.parents[1]
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--bbox-checkpoint",
        type=Path,
        default=APP_ROOT / "docker_submission/assets/gf_bbox_full_state.pt",
    )
    parser.add_argument(
        "--cbg-checkpoint",
        type=Path,
        default=APP_ROOT / "docker_submission/assets/gf_cbg_glance_state.pt",
    )
    parser.add_argument(
        "--bbox-raw",
        type=Path,
        default=APP_ROOT / "predictions/gf_a055_bbox_pu_best_ep280/public_raw",
    )
    parser.add_argument(
        "--allcrop-focus",
        type=Path,
        default=APP_ROOT / "public_sweeps/full_coverage_focus_20260820/merged/lesion_0.55",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        default=APP_ROOT / "analysis/lesion_error_touchcomp_20260825/predictions",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer")
    add_common(infer)
    project = APP_ROOT.parents[1]
    infer.add_argument("--images", type=Path, default=project / "validation/Validation-Public-Images")
    infer.add_argument("--labels", type=Path, default=project / "validation/Validation-Public-Labels")
    infer.add_argument("--part-id", type=int, default=0)
    infer.add_argument("--num-parts", type=int, default=1)
    infer.add_argument("--overwrite", action="store_true")
    report = commands.add_parser("summarize")
    add_common(report)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "infer":
        run_inference(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
