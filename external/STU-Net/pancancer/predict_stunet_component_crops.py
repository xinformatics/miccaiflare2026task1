#!/usr/bin/env python3
"""Public-only native component-crop approximation of the STU confirmer.

Unlike sparse sliding-window inference, this route crops the raw CT before
nnU-Net preprocessing. It therefore changes model context and is not exact;
fixed-public quality and runtime must both be measured before deployment.
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from analyze_stunet_gf_ensemble import confirmed_anchor_components
from predict_stunet_anchor_sparse import probabilities_to_nifti_order, quantized_threshold_mask
from predict_stunet_safe import (
    atomic_json,
    expected_output_name,
    nnUNetPredictor,
    safe_initialize,
    sha256_file,
    validate_dataset_role_path,
)
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)


P70 = 0.70
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def boxes_overlap(a: tuple[slice, ...], b: tuple[slice, ...]) -> bool:
    return all(x.start < y.stop and y.start < x.stop for x, y in zip(a, b))


def union_boxes(a: tuple[slice, ...], b: tuple[slice, ...]) -> tuple[slice, ...]:
    return tuple(slice(min(x.start, y.start), max(x.stop, y.stop)) for x, y in zip(a, b))


def component_crop_boxes(
    anchor: np.ndarray,
    spacing: tuple[float, float, float],
    margin_mm: float,
) -> list[tuple[slice, slice, slice]]:
    """Return clipped, overlap-merged native xyz component boxes."""

    anchor = np.asarray(anchor, dtype=bool)
    spacing = tuple(float(value) for value in spacing)
    if anchor.ndim != 3:
        raise ValueError("anchor must be three-dimensional")
    if len(spacing) != 3 or min(spacing) <= 0 or not np.isfinite(spacing).all():
        raise ValueError("invalid spacing")
    if margin_mm < 0 or not np.isfinite(margin_mm):
        raise ValueError("invalid component margin")
    labels, count = ndimage.label(anchor, structure=CONNECTIVITY_26)
    if count == 0:
        return []
    margin = np.ceil(float(margin_mm) / np.asarray(spacing)).astype(int)
    boxes: list[tuple[slice, slice, slice]] = []
    for found in ndimage.find_objects(labels):
        if found is None:
            continue
        expanded = tuple(
            slice(max(0, axis.start - int(pad)), min(size, axis.stop + int(pad)))
            for axis, pad, size in zip(found, margin, anchor.shape)
        )
        boxes.append(expanded)
    changed = True
    while changed:
        changed = False
        merged: list[tuple[slice, slice, slice]] = []
        while boxes:
            current = boxes.pop()
            for index, other in enumerate(boxes):
                if boxes_overlap(current, other):
                    boxes[index] = union_boxes(current, other)
                    changed = True
                    break
            else:
                merged.append(current)
        boxes = merged
    return sorted(boxes, key=lambda box: tuple(axis.start for axis in box))


def cropped_affine(affine: np.ndarray, box: tuple[slice, ...]) -> np.ndarray:
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = [axis.start for axis in box]
    return np.asarray(affine, dtype=np.float64) @ translation


def save_crop(
    path: Path,
    array: np.ndarray,
    reference: nib.Nifti1Image,
    box: tuple[slice, slice, slice],
) -> None:
    header = reference.header.copy()
    header.set_data_shape(array.shape)
    image = nib.Nifti1Image(array, cropped_affine(reference.affine, box), header)
    nib.save(image, str(path))


def decompress_nifti_gzip(
    source: Path,
    destination: Path,
    workers: int,
) -> None:
    """Materialize a gzip NIfTI with parallel decompression for mmap slicing."""

    pigz = shutil.which("pigz")
    if pigz is None:
        raise RuntimeError("pigz is required for bounded-time compressed CT loading")
    if not 1 <= workers <= 8:
        raise ValueError("I/O workers must be in [1, 8]")
    with destination.open("wb") as stream:
        subprocess.run(
            [pigz, "-p", str(workers), "-dc", str(source)],
            stdout=stream,
            check=True,
        )


def load_binary_like(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
        raise ValueError(f"geometry mismatch: {path}")
    values = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(values), (0, 1)).all():
        raise ValueError(f"nonbinary mask: {path}")
    return values.astype(bool, copy=False)


def save_binary(path: Path, mask: np.ndarray, reference: nib.Nifti1Image) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header), str(temporary))
    os.replace(temporary, path)


def main() -> None:
    wall_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--part-id", required=True, type=int)
    parser.add_argument("--num-parts", required=True, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--io-workers", default=8, type=int)
    parser.add_argument("--margin-mm", default=48.0, type=float)
    parser.add_argument("--scratch", default=Path("/tmp"), type=Path)
    parser.add_argument("--dataset-role", choices=("public",), required=True)
    parser.add_argument("--terminal", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.workers < 1
        or not 1 <= args.io_workers <= 8
        or args.num_parts < 1
        or not 0 <= args.part_id < args.num_parts
    ):
        raise ValueError("invalid worker or shard contract")
    validate_dataset_role_path(args.images, args.dataset_role)
    inventory = sorted(args.images.glob("*_0000.nii.gz"))
    selected = inventory[args.part_id :: args.num_parts]
    if not selected:
        raise ValueError("empty component-crop shard")
    expected_names = {expected_output_name(path) for path in inventory}
    if {path.name for path in args.anchors.glob("*.nii.gz")} != expected_names:
        raise ValueError("anchor inventory mismatch")
    checkpoint = args.model_dir / "fold_all" / args.checkpoint
    for path in (args.model_dir / "dataset.json", args.model_dir / "plans.json", checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha = sha256_file(checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.workers)
    torch.set_num_interop_threads(1)

    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError(f"CUDA GPU {args.gpu} unavailable")
    torch.cuda.set_device(args.gpu)
    device_keyword = (
        "perform_everything_on_device"
        if "perform_everything_on_device" in inspect.signature(nnUNetPredictor).parameters
        else "perform_everything_on_gpu"
    )
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device("cuda", args.gpu),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
        **{device_keyword: True},
    )
    torch.cuda.reset_peak_memory_stats(args.gpu)
    initialization_started = time.perf_counter()
    safe_initialize(predictor, args.model_dir, args.checkpoint)
    initialization_seconds = time.perf_counter() - initialization_started
    initialization_peak_allocated = torch.cuda.max_memory_allocated(args.gpu) / 2**20
    initialization_peak_reserved = torch.cuda.max_memory_reserved(args.gpu) / 2**20

    rows = []
    for image_path in selected:
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(args.gpu)
        name = expected_output_name(image_path)
        reference = nib.load(str(image_path))
        anchor = load_binary_like(args.anchors / name, reference)
        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
        boxes = component_crop_boxes(anchor, spacing, args.margin_mm)
        proposal = np.zeros(anchor.shape, dtype=bool)
        crop_tiles = 0
        io_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="stunet-component-crops-", dir=args.scratch) as temporary:
            temporary = Path(temporary)
            if image_path.name.endswith(".nii.gz"):
                uncompressed = temporary / "source.nii"
                decompress_nifti_gzip(image_path, uncompressed, args.io_workers)
                source_image = nib.load(str(uncompressed), mmap=True)
            else:
                source_image = reference
            io_seconds = time.perf_counter() - io_started
            for crop_id, box in enumerate(boxes):
                image_crop = np.ascontiguousarray(np.asanyarray(source_image.dataobj[box]))
                anchor_crop = np.ascontiguousarray(anchor[box].astype(np.uint8))
                crop_image_path = temporary / f"crop_{crop_id:03d}_0000.nii.gz"
                crop_anchor_path = temporary / f"crop_{crop_id:03d}.nii.gz"
                save_crop(crop_image_path, image_crop, reference, box)
                save_crop(crop_anchor_path, anchor_crop, reference, box)
                preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
                data, _, properties = preprocessor.run_case(
                    [str(crop_image_path)],
                    str(crop_anchor_path),
                    predictor.plans_manager,
                    predictor.configuration_manager,
                    predictor.dataset_json,
                )
                logits = predictor.predict_logits_from_preprocessed_data(
                    torch.from_numpy(np.ascontiguousarray(data)).float()
                )
                _, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
                    logits,
                    predictor.plans_manager,
                    predictor.configuration_manager,
                    predictor.label_manager,
                    properties,
                    return_probabilities=True,
                    num_threads_torch=args.workers,
                )
                probabilities = probabilities_to_nifti_order(probabilities, image_crop.shape)
                proposal[box] |= quantized_threshold_mask(probabilities[1], P70)
                crop_tiles += 1
                del data, logits, probabilities
        final = confirmed_anchor_components(anchor, proposal, np.zeros_like(anchor), "stu")
        if np.any(final & ~anchor):
            raise RuntimeError(f"cropped STU confirmer escaped anchor for {name}")
        save_binary(args.output / name, final, reference)
        torch.cuda.synchronize(args.gpu)
        rows.append(
            {
                "case": name,
                "anchor_components": int(ndimage.label(anchor, structure=CONNECTIVITY_26)[1]),
                "crop_count": len(boxes),
                "crop_forward_passes": crop_tiles,
                "compressed_input_materialization_seconds": io_seconds,
                "crop_voxels": int(sum(np.prod([axis.stop - axis.start for axis in box]) for box in boxes)),
                "native_voxels": int(np.prod(anchor.shape)),
                "seconds": time.perf_counter() - started,
                "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(args.gpu) / 2**20,
                "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved(args.gpu) / 2**20,
            }
        )
        del proposal, final, anchor
        torch.cuda.empty_cache()

    if sha256_file(checkpoint) != checkpoint_sha:
        raise RuntimeError("checkpoint changed during component-crop inference")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "experiment": "native_component_crop_stunet_p70_approximation",
        "deployment_authorized": False,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha},
        "margin_mm": args.margin_mm,
        "compressed_input_loader": "pigz_to_temporary_uncompressed_nifti_then_mmap",
        "io_workers": args.io_workers,
        "p70_threshold": P70,
        "case_count": len(rows),
        "model_initialization_seconds": initialization_seconds,
        "initialization_peak_gpu_allocated_mb": initialization_peak_allocated,
        "initialization_peak_gpu_reserved_mb": initialization_peak_reserved,
        "wall_seconds": time.perf_counter() - wall_started,
        "peak_gpu_allocated_mb": max([initialization_peak_allocated, *(row["peak_gpu_allocated_mb"] for row in rows)]),
        "peak_gpu_reserved_mb": max([initialization_peak_reserved, *(row["peak_gpu_reserved_mb"] for row in rows)]),
        "cases": rows,
        "dataset_role": "public",
        "public_images_read": True,
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.terminal, payload)


if __name__ == "__main__":
    main()
