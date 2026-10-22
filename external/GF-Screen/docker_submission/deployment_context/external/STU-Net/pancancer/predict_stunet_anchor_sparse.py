#!/usr/bin/env python3
"""Experimental public-only sparse STU p70 anchor confirmer.

This predictor evaluates only full-grid nnU-Net windows that can contribute to
GF anchor voxels (plus a conservative resampling halo). It is intentionally not
authorized for hidden/test use until its final masks are proven identical to
the full-volume epoch-160 p70 deployment on all 50 fixed-public cases.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from scipy import ndimage

from predict_stunet_safe import (
    atomic_json,
    expected_output_name,
    nnUNetPredictor,
    safe_initialize,
    sha256_file,
    validate_dataset_role_path,
)
from runtime_confirmation_policy import confirmation_state
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)


P70 = 0.70
DEPENDENCY_HALO = 2


def confirmed_anchor_components(
    anchor: np.ndarray,
    stunet: np.ndarray,
    freetumor: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Keep complete 26-connected anchor components confirmed by supports."""

    labels, count = ndimage.label(anchor.astype(bool), structure=np.ones((3, 3, 3)))
    if count == 0:
        return np.zeros_like(anchor, dtype=bool)
    stunet_ids = np.zeros(count + 1, dtype=bool)
    freetumor_ids = np.zeros(count + 1, dtype=bool)
    stunet_ids[np.unique(labels[anchor & stunet])] = True
    freetumor_ids[np.unique(labels[anchor & freetumor])] = True
    if mode == "stu":
        keep = stunet_ids
    elif mode == "ftu":
        keep = freetumor_ids
    elif mode == "either":
        keep = stunet_ids | freetumor_ids
    elif mode == "both":
        keep = stunet_ids & freetumor_ids
    else:
        raise ValueError(f"invalid anchor confirmation mode: {mode}")
    keep[0] = False
    return keep[labels]


def quantized_threshold_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the deployed uint16 probability contract before thresholding."""

    probability = np.asarray(probability, dtype=np.float32)
    if not 0.0 <= threshold <= 1.0 or not np.isfinite(probability).all():
        raise ValueError("invalid probability or threshold")
    quantized = np.rint(np.clip(probability, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return quantized.astype(np.float32) / 65535.0 > threshold


def probabilities_to_nifti_order(
    probabilities: np.ndarray, reference_shape: tuple[int, int, int]
) -> np.ndarray:
    """Normalize nnU-Net's channel/z/y/x export to channel/x/y/z order."""

    probabilities = np.asarray(probabilities)
    reference_shape = tuple(int(value) for value in reference_shape)
    if probabilities.shape == (2, *reference_shape):
        return probabilities
    if probabilities.shape == (2, *reference_shape[::-1]):
        return probabilities.transpose(0, 3, 2, 1)
    raise RuntimeError(
        "restored probability/NIfTI shape mismatch: "
        f"{probabilities.shape} versus {(2, *reference_shape)}"
    )


def dependency_from_preprocessed_anchor(
    anchor_preprocessed: np.ndarray, original_anchor_nonempty: bool
) -> np.ndarray:
    """Build the conservative dependency halo and reject erased anchors."""

    anchor_preprocessed = np.asarray(anchor_preprocessed) > 0
    if anchor_preprocessed.ndim != 3:
        raise ValueError("preprocessed anchor must be three-dimensional")
    if original_anchor_nonempty and not anchor_preprocessed.any():
        raise RuntimeError("preprocessing erased a nonempty GF anchor")
    return ndimage.binary_dilation(
        anchor_preprocessed,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=DEPENDENCY_HALO,
    )


def select_dependency_slicers(
    slicers: list[tuple[slice, ...]], dependency_mask: np.ndarray
) -> list[tuple[slice, ...]]:
    """Keep every full-grid window that contributes to a dependency voxel."""

    dependency_mask = np.asarray(dependency_mask, dtype=bool)
    if dependency_mask.ndim != 3:
        raise ValueError("dependency mask must be three-dimensional")
    selected = [slicer for slicer in slicers if dependency_mask[slicer[1:]].any()]
    if dependency_mask.any() and not selected:
        raise RuntimeError("nonempty dependency mask has no covering window")
    return selected


def dependency_coverage_is_exact(
    full_slicers: list[tuple[slice, ...]],
    selected_slicers: list[tuple[slice, ...]],
    dependency_mask: np.ndarray,
) -> bool:
    """Prove selected/full window multiplicity agrees at dependency voxels."""

    dependency_mask = np.asarray(dependency_mask, dtype=bool)
    full = np.zeros(dependency_mask.shape, dtype=np.uint16)
    sparse = np.zeros_like(full)
    for slicer in full_slicers:
        full[slicer[1:]] += 1
    for slicer in selected_slicers:
        sparse[slicer[1:]] += 1
    return bool(
        np.array_equal(full[dependency_mask], sparse[dependency_mask])
        and np.all(full[dependency_mask] > 0)
    )


class AnchorSparsePredictor(nnUNetPredictor):
    """nnU-Net predictor with an exact target-dependency window subset."""

    dependency_mask: np.ndarray | None = None
    full_tile_count: int = 0
    selected_tile_count: int = 0

    def _internal_get_sliding_window_slicers(self, image_size):
        full = super()._internal_get_sliding_window_slicers(image_size)
        if self.dependency_mask is None:
            self.full_tile_count = self.selected_tile_count = len(full)
            return full
        padded, _ = pad_nd_image(
            self.dependency_mask[None].astype(np.uint8, copy=False),
            self.configuration_manager.patch_size,
            "constant",
            # acvl-utils 0.2.2 dispatches NumPy arrays to np.pad, whose
            # constant keyword is ``constant_values`` (not torch's ``value``).
            {"constant_values": 0},
            True,
            None,
        )
        padded = np.asarray(padded[0], dtype=bool)
        if tuple(padded.shape) != tuple(image_size):
            raise RuntimeError(
                f"padded dependency/image mismatch: {padded.shape} != {image_size}"
            )
        selected = select_dependency_slicers(full, padded)
        if not dependency_coverage_is_exact(full, selected, padded):
            raise RuntimeError("sparse window dependency coverage is not exact")
        self.full_tile_count = len(full)
        self.selected_tile_count = len(selected)
        return selected

    def predict_sliding_window_return_logits(self, input_image):
        logits = super().predict_sliding_window_return_logits(input_image)
        # The parent divides unvisited voxels by zero. Those voxels are outside
        # the dependency halo and cannot affect p70 inside the anchor; neutral
        # logits make them background at the strict >0.70 deployment threshold.
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)


def load_binary_like(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape != reference.shape or not np.allclose(
        image.affine, reference.affine, atol=1e-4
    ):
        raise ValueError(f"geometry mismatch: {path}")
    values = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(values), [0, 1]).all():
        raise ValueError(f"nonbinary mask: {path}")
    return values.astype(bool, copy=False)


def save_binary(path: Path, mask: np.ndarray, reference: nib.Nifti1Image) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header),
        str(temporary),
    )
    os.replace(temporary, path)


def main() -> None:
    wall_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--freetumor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--part-id", required=True, type=int)
    parser.add_argument("--num-parts", required=True, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--dataset-role", choices=("public",), required=True)
    parser.add_argument("--terminal", required=True, type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.num_parts < 1 or not 0 <= args.part_id < args.num_parts:
        raise ValueError("invalid worker or shard contract")
    validate_dataset_role_path(args.images, args.dataset_role)
    inventory = sorted(args.images.glob("*_0000.nii.gz"))
    selected_images = inventory[args.part_id :: args.num_parts]
    if not selected_images:
        raise ValueError("empty sparse-inference shard")
    expected_names = {expected_output_name(path) for path in inventory}
    for role, directory in (("anchor", args.anchors), ("FreeTumor", args.freetumor)):
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected_names:
            raise ValueError(
                f"{role} inventory mismatch: missing={sorted(expected_names-observed)} "
                f"extra={sorted(observed-expected_names)}"
            )
    checkpoint = args.model_dir / "fold_all" / args.checkpoint
    for path in (args.model_dir / "dataset.json", args.model_dir / "plans.json", checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha_before = sha256_file(checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.workers)
    torch.set_num_interop_threads(1)

    pending = []
    rows = []
    for image_path in selected_images:
        name = expected_output_name(image_path)
        reference = nib.load(str(image_path))
        anchor_path = args.anchors / name
        free_path = args.freetumor / name
        anchor = load_binary_like(anchor_path, reference)
        freetumor = load_binary_like(free_path, reference)
        state = confirmation_state(anchor, freetumor)
        if not state["requires_stunet"]:
            save_binary(args.output / name, anchor, reference)
            rows.append(
                {
                    "case": name,
                    **state,
                    "stunet_skipped": True,
                    "full_tiles": 0,
                    "selected_tiles": 0,
                    "seconds": 0.0,
                }
            )
        else:
            pending.append((image_path, name, reference, anchor, freetumor, state))

    predictor = None
    model_initialization_seconds = 0.0
    if pending:
        if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
            raise RuntimeError(f"CUDA GPU {args.gpu} unavailable")
        torch.cuda.set_device(args.gpu)
        device_keyword = (
            "perform_everything_on_device"
            if "perform_everything_on_device"
            in inspect.signature(AnchorSparsePredictor).parameters
            else "perform_everything_on_gpu"
        )
        predictor = AnchorSparsePredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            device=torch.device("cuda", args.gpu),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
            **{device_keyword: True},
        )
        initialization_started = time.perf_counter()
        safe_initialize(predictor, args.model_dir, args.checkpoint)
        model_initialization_seconds = time.perf_counter() - initialization_started

    for image_path, name, reference, anchor, freetumor, state in pending:
        started = time.perf_counter()
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
        data, anchor_preprocessed, properties = preprocessor.run_case(
            [str(image_path)],
            str(args.anchors / name),
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        dependency = dependency_from_preprocessed_anchor(
            anchor_preprocessed[0], bool(anchor.any())
        )
        predictor.dependency_mask = dependency
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
        probabilities = probabilities_to_nifti_order(probabilities, reference.shape)
        proposal = quantized_threshold_mask(probabilities[1], P70)
        final = confirmed_anchor_components(anchor, proposal, freetumor, "either")
        if np.any(final & ~anchor):
            raise RuntimeError(f"sparse confirmer escaped anchor for {name}")
        save_binary(args.output / name, final, reference)
        torch.cuda.synchronize(args.gpu)
        rows.append(
            {
                "case": name,
                **state,
                "stunet_skipped": False,
                "full_tiles": predictor.full_tile_count,
                "selected_tiles": predictor.selected_tile_count,
                "seconds": time.perf_counter() - started,
            }
        )
        predictor.dependency_mask = None
        del data, anchor_preprocessed, dependency, logits, probabilities, proposal, final
        torch.cuda.empty_cache()

    checkpoint_sha_after = sha256_file(checkpoint)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("checkpoint changed during sparse inference")
    rows.sort(key=lambda row: row["case"])
    expected_outputs = {expected_output_name(path) for path in selected_images}
    observed_outputs = {path.name for path in args.output.glob("*.nii.gz")}
    if not expected_outputs.issubset(observed_outputs):
        raise RuntimeError("incomplete sparse output shard")
    terminal = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "experiment": "anchor_sparse_exact_dependency_p70",
        "deployment_authorized": False,
        "authorization_requirement": (
            "all 50 fixed-public final masks must equal the full-volume "
            "component-epoch-160 p70 deployment"
        ),
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha_after},
        "p70_threshold": P70,
        "probability_contract": "round_clip_float_to_uint16_then_strict_greater_than_p70",
        "tile_step_size": 0.5,
        "dependency_halo_preprocessed_voxels": DEPENDENCY_HALO,
        "case_count": len(rows),
        "cases": rows,
        "stunet_skipped_cases": sum(row["stunet_skipped"] for row in rows),
        "model_initialization_seconds": model_initialization_seconds,
        "wall_seconds": time.perf_counter() - wall_started,
        "full_tiles": sum(row["full_tiles"] for row in rows),
        "selected_tiles": sum(row["selected_tiles"] for row in rows),
        "dataset_role": "public",
        "public_images_read": True,
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.terminal, terminal)


if __name__ == "__main__":
    main()
