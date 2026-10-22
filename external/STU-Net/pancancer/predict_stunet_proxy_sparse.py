#!/usr/bin/env python3
"""Sparse STU confirmer using the FreeTumor-free proxy route."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)


ROOT = Path(__file__).resolve().parents[3]
DOCKER_ROOT = ROOT / "external/GF-Screen/docker_submission"
for source in (ROOT / "external/GF-Screen", DOCKER_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from predict_stunet_anchor_sparse import (  # noqa: E402
    AnchorSparsePredictor,
    dependency_from_preprocessed_anchor,
    load_binary_like,
    probabilities_to_nifti_order,
    save_binary,
)
from predict_stunet_safe import (  # noqa: E402
    atomic_json,
    expected_output_name,
    safe_initialize,
    sha256_file,
    validate_dataset_role_path,
)
from runtime_winner_mask_recipe import (  # noqa: E402
    confirm_with_stunet_p70_or_existing_support_proxy,
    existing_support_proxy_mask,
)


def main() -> None:
    wall_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--bbox-raw", required=True, type=Path)
    parser.add_argument("--prompt-whole", required=True, type=Path)
    parser.add_argument("--prompt-tumor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--part-id", required=True, type=int)
    parser.add_argument("--num-parts", required=True, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--dataset-role", choices=("public", "test"), required=True)
    parser.add_argument("--terminal", required=True, type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.num_parts < 1 or not 0 <= args.part_id < args.num_parts:
        raise ValueError("invalid worker or shard contract")
    validate_dataset_role_path(args.images, args.dataset_role)
    inventory = sorted(args.images.glob("*_0000.nii.gz"))
    selected_images = inventory[args.part_id :: args.num_parts]
    if not selected_images:
        raise ValueError("empty proxy-sparse shard")
    expected_names = {expected_output_name(path) for path in inventory}
    for role, directory in (
        ("anchor", args.anchors),
        ("bbox_raw", args.bbox_raw),
        ("prompt_whole", args.prompt_whole),
        ("prompt_tumor", args.prompt_tumor),
    ):
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected_names:
            raise ValueError(f"{role} inventory mismatch")
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
        anchor = load_binary_like(args.anchors / name, reference)
        bbox = load_binary_like(args.bbox_raw / name, reference)
        whole = load_binary_like(args.prompt_whole / name, reference)
        tumor = load_binary_like(args.prompt_tumor / name, reference)
        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
        proxy = existing_support_proxy_mask(
            anchor=anchor,
            prompt_whole_body_lesion=whole,
            prompt_tumor=tumor,
            bbox_raw=bbox,
            spacing=spacing,
        )
        unresolved = anchor & ~proxy
        if not unresolved.any():
            save_binary(args.output / name, anchor, reference)
            rows.append(
                {
                    "case": name,
                    "anchor_voxels": int(anchor.sum()),
                    "proxy_voxels": int(proxy.sum()),
                    "unresolved_voxels": 0,
                    "stunet_skipped": True,
                    "full_tiles": 0,
                    "selected_tiles": 0,
                    "seconds": 0.0,
                    "peak_gpu_allocated_mb": 0.0,
                    "peak_gpu_reserved_mb": 0.0,
                }
            )
        else:
            pending.append(
                (image_path, name, reference, anchor, bbox, whole, tumor, proxy, unresolved)
            )

    predictor = None
    model_initialization_seconds = 0.0
    initialization_peak_allocated_mb = 0.0
    initialization_peak_reserved_mb = 0.0
    if pending:
        if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
            raise RuntimeError(f"CUDA GPU {args.gpu} unavailable")
        torch.cuda.set_device(args.gpu)
        device_keyword = (
            "perform_everything_on_device"
            if "perform_everything_on_device" in inspect.signature(AnchorSparsePredictor).parameters
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
        torch.cuda.synchronize(args.gpu)
        torch.cuda.reset_peak_memory_stats(args.gpu)
        started = time.perf_counter()
        safe_initialize(predictor, args.model_dir, args.checkpoint)
        torch.cuda.synchronize(args.gpu)
        model_initialization_seconds = time.perf_counter() - started
        initialization_peak_allocated_mb = torch.cuda.max_memory_allocated(args.gpu) / (
            1024**2
        )
        initialization_peak_reserved_mb = torch.cuda.max_memory_reserved(args.gpu) / (
            1024**2
        )

    for image_path, name, reference, anchor, bbox, whole, tumor, proxy, unresolved in pending:
        torch.cuda.synchronize(args.gpu)
        torch.cuda.reset_peak_memory_stats(args.gpu)
        started = time.perf_counter()
        stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name
        dependency_path = args.output / f".{stem}.proxy_dependency.nii.gz"
        save_binary(dependency_path, unresolved, reference)
        try:
            preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
            data, dependency_preprocessed, properties = preprocessor.run_case(
                [str(image_path)],
                str(dependency_path),
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.dataset_json,
            )
        finally:
            dependency_path.unlink(missing_ok=True)
        dependency = dependency_from_preprocessed_anchor(
            dependency_preprocessed[0], True
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
        probability_uint16 = np.rint(
            np.clip(probabilities[1], 0.0, 1.0) * 65535.0
        ).astype(np.uint16)
        final, stats = confirm_with_stunet_p70_or_existing_support_proxy(
            anchor=anchor,
            prompt_whole_body_lesion=whole,
            prompt_tumor=tumor,
            bbox_raw=bbox,
            stunet_probability_uint16=probability_uint16,
            spacing=tuple(float(value) for value in reference.header.get_zooms()[:3]),
        )
        if np.any(final & ~anchor) or np.any(proxy & ~final):
            raise RuntimeError(f"proxy-sparse confirmer violated component contract: {name}")
        save_binary(args.output / name, final, reference)
        torch.cuda.synchronize(args.gpu)
        rows.append(
            {
                "case": name,
                "anchor_voxels": int(anchor.sum()),
                "proxy_voxels": int(proxy.sum()),
                "unresolved_voxels": int(unresolved.sum()),
                "stunet_skipped": False,
                "full_tiles": predictor.full_tile_count,
                "selected_tiles": predictor.selected_tile_count,
                "proxy_confirmed_components": stats.proxy_confirmed_components,
                "stunet_confirmed_components": stats.stunet_confirmed_components,
                "seconds": time.perf_counter() - started,
                "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(args.gpu)
                / (1024**2),
                "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved(args.gpu)
                / (1024**2),
            }
        )
        predictor.dependency_mask = None
        del data, dependency_preprocessed, dependency, logits, probabilities, probability_uint16, final
        torch.cuda.empty_cache()

    checkpoint_sha_after = sha256_file(checkpoint)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("checkpoint changed during proxy-sparse inference")
    rows.sort(key=lambda row: row["case"])
    expected_outputs = {expected_output_name(path) for path in selected_images}
    observed_outputs = {path.name for path in args.output.glob("*.nii.gz")}
    if not expected_outputs.issubset(observed_outputs):
        raise RuntimeError("incomplete proxy-sparse output shard")
    atomic_json(
        args.terminal,
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "COMPLETE",
            "experiment": "freetumor_free_existing_support_proxy_sparse_stunet_p70",
            "deployment_authorized": False,
            "authorization_requirement": "all 50 public masks must equal the frozen full-volume proxy deployment and pass runtime",
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha_after},
            "p70_threshold": 0.70,
            "tile_step_size": 0.5,
            "case_count": len(rows),
            "cases": rows,
            "stunet_skipped_cases": sum(row["stunet_skipped"] for row in rows),
            "model_initialization_seconds": model_initialization_seconds,
            "initialization_peak_gpu_allocated_mb": initialization_peak_allocated_mb,
            "initialization_peak_gpu_reserved_mb": initialization_peak_reserved_mb,
            "peak_gpu_allocated_mb": max(
                [initialization_peak_allocated_mb]
                + [row["peak_gpu_allocated_mb"] for row in rows]
            ),
            "peak_gpu_reserved_mb": max(
                [initialization_peak_reserved_mb]
                + [row["peak_gpu_reserved_mb"] for row in rows]
            ),
            "wall_seconds": time.perf_counter() - wall_started,
            "full_tiles": sum(row["full_tiles"] for row in rows),
            "selected_tiles": sum(row["selected_tiles"] for row in rows),
            "dataset_role": args.dataset_role,
            "public_images_read": args.dataset_role == "public",
            "public_labels_read": False,
            "hidden_read": args.dataset_role == "test",
            "submission_created": False,
            "uploaded": False,
        },
    )


if __name__ == "__main__":
    main()
