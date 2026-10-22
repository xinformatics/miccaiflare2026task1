#!/usr/bin/env python3
"""Build label-safe hard-negative crops from proxy-healthy CT predictions.

The workflow has two explicit phases so GPU inference remains a separately
logged operation:

1. ``stage`` creates a flat, collision-checked directory of symlinks for the
   training images listed in the healthy Decathlon JSON.
2. Run ``val_GF_FLARE.py`` on that directory with the immutable champion.
3. ``build`` aligns each binary prediction with its CT, keeps connected
   components at or above a physical-volume threshold, and saves 96x96x64
   crops with all-zero trusted labels plus a derived training JSON.

No public or hidden validation label is accepted by this utility. The output
JSON includes the original healthy cases as well as the mined crops, and its
manifest reports a repeat value that approximately preserves the original
healthy-stream mass used by ``--healthy_repeat 100``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from monai import transforms
from scipy import ndimage

from inference_naming import prediction_name


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def load_healthy_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for split in ("training", "validation"):
        if not isinstance(payload.get(split), list):
            raise ValueError(f"{path}: missing list split {split!r}")
        for item in payload[split]:
            for key in ("image", "label", "name"):
                if not item.get(key):
                    raise ValueError(f"{path}: {split} item missing {key!r}")
            for key in ("image", "label"):
                parts = {part.casefold() for part in Path(item[key]).resolve().parts}
                if "validation" in parts or "healthyimages-nolesion" in parts:
                    raise ValueError(
                        f"evaluation data is forbidden in hard-negative source JSON: {item[key]}"
                    )
    names = [str(item.get("name", "")) for item in payload["training"]]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("healthy training names must be non-empty and unique")
    return payload


def file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def stage_images(healthy_json: Path, output: Path) -> None:
    payload = load_healthy_json(healthy_json)
    output.mkdir(parents=True, exist_ok=True)
    staged = []
    basenames: set[str] = set()
    for item in payload["training"]:
        source = Path(item["image"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        basename = source.name
        if basename in basenames:
            raise ValueError(f"staging basename collision: {basename}")
        basenames.add(basename)
        destination = output / basename
        if destination.is_symlink() or destination.exists():
            if destination.resolve() != source:
                raise FileExistsError(f"wrong existing stage target: {destination}")
        else:
            destination.symlink_to(source)
        staged.append(
            {
                "name": item["name"],
                "source": str(source),
                "staged": str(destination.absolute()),
                "prediction_name": prediction_name(basename),
            }
        )
    manifest = {
        "mode": "stage",
        "healthy_json": file_identity(healthy_json.resolve()),
        "cases": len(staged),
        "items": staged,
    }
    # val_GF_FLARE enumerates every directory entry without an extension
    # filter, so metadata must never be written inside the flat image folder.
    manifest_path = output.parent / f"{output.name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(staged),
                "output": str(output.resolve()),
                "manifest": str(manifest_path.resolve()),
            },
            indent=2,
        )
    )


def centered_crop(array: np.ndarray, center: tuple[int, int, int], roi: tuple[int, int, int]) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return a padded spatial crop and its possibly-negative source start."""
    if array.ndim != 3:
        raise ValueError(f"expected 3D array, found {array.shape}")
    start = tuple(int(c) - int(size) // 2 for c, size in zip(center, roi))
    stop = tuple(begin + int(size) for begin, size in zip(start, roi))
    source_start = tuple(max(0, begin) for begin in start)
    source_stop = tuple(min(limit, end) for limit, end in zip(array.shape, stop))
    destination_start = tuple(src - begin for src, begin in zip(source_start, start))
    destination_stop = tuple(
        dst + (src_end - src_begin)
        for dst, src_begin, src_end in zip(destination_start, source_start, source_stop)
    )
    result = np.zeros(roi, dtype=array.dtype)
    source_slices = tuple(slice(a, b) for a, b in zip(source_start, source_stop))
    destination_slices = tuple(slice(a, b) for a, b in zip(destination_start, destination_stop))
    result[destination_slices] = array[source_slices]
    return result, start


def component_candidates(mask: np.ndarray, voxel_mm3: float, min_mm3: float, max_components: int) -> list[dict[str, object]]:
    component_map, count = ndimage.label(mask > 0, structure=CONNECTIVITY_26)
    if not count:
        return []
    sizes = np.bincount(component_map.reshape(-1))
    candidates = []
    for component_id in range(1, len(sizes)):
        voxels = int(sizes[component_id])
        volume_mm3 = voxels * voxel_mm3
        if volume_mm3 < min_mm3:
            continue
        coordinates = np.argwhere(component_map == component_id)
        center = tuple(int(value) for value in np.rint(coordinates.mean(axis=0)))
        candidates.append(
            {
                "component_id": component_id,
                "voxels": voxels,
                "volume_mm3": volume_mm3,
                "center": center,
            }
        )
    candidates.sort(key=lambda item: (-float(item["volume_mm3"]), int(item["component_id"])))
    return candidates[:max_components] if max_components else candidates


def deterministic_preprocess(spacing: tuple[float, float, float], roi: tuple[int, int, int]):
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "prediction", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "prediction", "label"]),
            transforms.Orientationd(keys=["image", "prediction", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "prediction", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest", "nearest"),
            ),
            transforms.ScaleIntensityRanged(
                keys="image", a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(
                keys=["image", "prediction", "label"], source_key="image"
            ),
            transforms.SpatialPadd(
                keys=["image", "prediction", "label"], spatial_size=roi, mode="constant"
            ),
        ]
    )


def build_crops(args: argparse.Namespace) -> None:
    payload = load_healthy_json(args.healthy_json)
    images_dir = args.output / "imagesTr"
    labels_dir = args.output / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = deterministic_preprocess(args.spacing, args.roi)
    cropped_items = []
    case_reports = []
    for case_index, item in enumerate(payload["training"], start=1):
        image_path = Path(item["image"]).resolve()
        pred_path = args.predictions / prediction_name(image_path.name)
        if not pred_path.is_file():
            raise FileNotFoundError(f"missing healthy prediction: {pred_path}")
        label_path = Path(item["label"]).resolve()
        transformed = preprocessing(
            {"image": str(image_path), "prediction": str(pred_path), "label": str(label_path)}
        )
        image = np.asarray(transformed["image"][0], dtype=np.float32)
        prediction = np.asarray(transformed["prediction"][0]) > 0
        if np.any(np.asarray(transformed["label"])):
            raise ValueError(f"hard-negative source is not label-confirmed healthy: {item['name']}")
        affine = np.asarray(transformed["image"].affine, dtype=np.float64)
        voxel_mm3 = float(np.prod(args.spacing))
        candidates = component_candidates(
            prediction, voxel_mm3, args.min_mm3, args.max_components_per_case
        )
        report = {"name": item["name"], "prediction": str(pred_path.resolve()), "components": []}
        for rank, candidate in enumerate(candidates, start=1):
            cropped_image, start = centered_crop(image, candidate["center"], args.roi)
            # Reconstruct clipped HU so the standard GF loader's fixed window
            # maps the saved crop exactly back to the normalized tensor.
            cropped_hu = cropped_image * 425.0 - 175.0
            crop_affine = affine.copy()
            crop_affine[:3, 3] = (affine @ np.asarray([*start, 1.0]))[:3]
            crop_name = f"HN_{item['name']}_c{rank:02d}"
            output_image = images_dir / f"{crop_name}_0000.nii.gz"
            output_label = labels_dir / f"{crop_name}.nii.gz"
            nib.save(nib.Nifti1Image(cropped_hu.astype(np.float32), crop_affine), output_image)
            nib.save(nib.Nifti1Image(np.zeros(args.roi, dtype=np.uint8), crop_affine), output_label)
            cropped_items.append(
                {
                    "image": str(output_image.resolve()),
                    "label": str(output_label.resolve()),
                    "name": crop_name,
                }
            )
            report["components"].append(
                {
                    **candidate,
                    "center": list(candidate["center"]),
                    "crop_start": list(start),
                    "image": str(output_image.resolve()),
                }
            )
        case_reports.append(report)
        print(f"processed {case_index}/{len(payload['training'])}: {len(candidates)} crops", flush=True)

    if not cropped_items:
        raise RuntimeError("no hard-negative components met the configured threshold")
    combined_training = [dict(item) for item in payload["training"]] + cropped_items
    derived = {
        "description": "GF-Screen proxy-healthy cases plus champion-mined hard-negative crops; no validation labels used",
        "training": combined_training,
        "validation": payload["validation"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    original_mass = len(payload["training"]) * args.reference_healthy_repeat
    recommended_repeat = max(1, round(original_mass / len(combined_training)))
    manifest = {
        "mode": "build",
        "healthy_json": file_identity(args.healthy_json.resolve()),
        "predictions": str(args.predictions.resolve()),
        "output_json": file_identity(args.output_json.resolve()),
        "configuration": {
            "spacing": list(args.spacing),
            "roi": list(args.roi),
            "min_mm3": args.min_mm3,
            "max_components_per_case": args.max_components_per_case,
            "connectivity": 26,
            "reference_healthy_repeat": args.reference_healthy_repeat,
        },
        "original_cases": len(payload["training"]),
        "hard_negative_crops": len(cropped_items),
        "combined_cases": len(combined_training),
        "recommended_healthy_repeat": recommended_repeat,
        "case_reports": case_reports,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "hard_negative_crops": len(cropped_items),
                "combined_cases": len(combined_training),
                "recommended_healthy_repeat": recommended_repeat,
                "output_json": str(args.output_json.resolve()),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--healthy-json", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--healthy-json", type=Path, required=True)
    build.add_argument("--predictions", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--output-json", type=Path, required=True)
    build.add_argument("--min-mm3", type=float, default=1000.0)
    build.add_argument("--max-components-per-case", type=int, default=4)
    build.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 3.0))
    build.add_argument("--roi", type=int, nargs=3, default=(96, 96, 64))
    build.add_argument("--reference-healthy-repeat", type=int, default=100)
    args = parser.parse_args()
    if args.command == "build":
        args.spacing = tuple(float(value) for value in args.spacing)
        args.roi = tuple(int(value) for value in args.roi)
        if args.min_mm3 <= 0 or args.max_components_per_case < 0:
            parser.error("volume threshold must be positive and max-components non-negative")
        if any(value <= 0 for value in (*args.spacing, *args.roi)):
            parser.error("spacing and ROI values must be positive")
        if args.reference_healthy_repeat < 1:
            parser.error("reference healthy repeat must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "stage":
        stage_images(args.healthy_json, args.output)
    else:
        build_crops(args)


if __name__ == "__main__":
    main()
