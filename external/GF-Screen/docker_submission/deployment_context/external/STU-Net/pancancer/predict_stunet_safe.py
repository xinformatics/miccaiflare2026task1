#!/usr/bin/env python3
"""Hash-bound, shardable STU-Net inference without site-package edits."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def _install_torch113_optimized_module_shim() -> None:
    """Provide nnU-Net's compile-type sentinel on pre-Dynamo PyTorch.

    nnU-Net 2.4 imports ``torch._dynamo.OptimizedModule`` even when compilation
    is explicitly disabled. The challenge image's torch 1.13 build has no
    ``torch._dynamo`` package. A sentinel type is sufficient because deployed
    networks are ordinary ``nn.Module`` objects and ``nnUNet_compile=false``.
    """

    try:
        from torch._dynamo import OptimizedModule  # noqa: F401
    except ImportError:
        module = types.ModuleType("torch._dynamo")

        class OptimizedModule(torch.nn.Module):
            pass

        module.OptimizedModule = OptimizedModule
        sys.modules["torch._dynamo"] = module


_install_torch113_optimized_module_shim()


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from STUNetTrainerPanCancer import (  # noqa: E402
    STUNetTrainerPanCancer,
    STUNetTrainerPanCancerComponentBox,
    STUNetTrainerPanCancerComponentBoxE160Balanced,
)
from STUNetTrainerPanCancerCompleteHybrid import (  # noqa: E402
    STUNetTrainerPanCancerComponentBoxE160CompleteHybrid,
)
from STUNetTrainerPanCancerSourceBalanced import (  # noqa: E402
    STUNetTrainerPanCancerComponentBoxE160SourceStratified,
)
from compact_probability_export import export_prediction_from_logits_compact  # noqa: E402

import nnunetv2.inference.predict_from_raw_data as prediction_module  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402


TRAINER_NAME = "STUNetTrainerPanCancer"
COMPONENT_BOX_TRAINER_NAME = "STUNetTrainerPanCancerComponentBox"
COMPONENT_BOX_E160_BALANCED_TRAINER_NAME = (
    "STUNetTrainerPanCancerComponentBoxE160Balanced"
)
COMPONENT_BOX_E160_COMPLETE_HYBRID_TRAINER_NAME = (
    "STUNetTrainerPanCancerComponentBoxE160CompleteHybrid"
)
COMPONENT_BOX_E160_SOURCE_STRATIFIED_TRAINER_NAME = (
    "STUNetTrainerPanCancerComponentBoxE160SourceStratified"
)
TRAINER_CLASSES = {
    TRAINER_NAME: STUNetTrainerPanCancer,
    COMPONENT_BOX_TRAINER_NAME: STUNetTrainerPanCancerComponentBox,
    COMPONENT_BOX_E160_BALANCED_TRAINER_NAME: (
        STUNetTrainerPanCancerComponentBoxE160Balanced
    ),
    COMPONENT_BOX_E160_COMPLETE_HYBRID_TRAINER_NAME: (
        STUNetTrainerPanCancerComponentBoxE160CompleteHybrid
    ),
    COMPONENT_BOX_E160_SOURCE_STRATIFIED_TRAINER_NAME: (
        STUNetTrainerPanCancerComponentBoxE160SourceStratified
    ),
}
_ORIGINAL_RESOLVER = prediction_module.recursive_find_python_class


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_trainer(folder, class_name, current_module):
    if class_name in TRAINER_CLASSES:
        return TRAINER_CLASSES[class_name]
    return _ORIGINAL_RESOLVER(folder, class_name, current_module)


def safe_initialize(
    predictor: nnUNetPredictor,
    model_dir: Path,
    checkpoint_name: str,
) -> None:
    """Restrict the locally produced checkpoint load to tensor-safe globals."""

    prediction_module.recursive_find_python_class = resolve_trainer
    original_load = torch.load
    try:
        supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
        supports_safe_globals = hasattr(torch.serialization, "safe_globals")
        if supports_weights_only and supports_safe_globals:
            numpy_core = getattr(np, "_core", np.core)
            numpy_globals = [
                numpy_core.multiarray.scalar,
                np.dtype,
                type(np.dtype(np.float64)),
                type(np.dtype(np.float32)),
                type(np.dtype(np.int64)),
                type(np.dtype(np.bool_)),
            ]

            def weights_only_load(*args, **kwargs):
                kwargs["weights_only"] = True
                return original_load(*args, **kwargs)

            torch.load = weights_only_load
            # nnU-Net's logger stores NumPy scalar/dtype objects. Restrict the
            # allowlist to those exact safe value types while keeping tensor-
            # only checkpoint loading enabled on modern PyTorch.
            with torch.serialization.safe_globals(numpy_globals):
                predictor.initialize_from_trained_model_folder(
                    str(model_dir), use_folds=("all",), checkpoint_name=checkpoint_name
                )
        else:
            # The challenge image is pinned to torch 1.13, before
            # ``weights_only`` and ``safe_globals`` existed. The bundled
            # checkpoint is hash-bound before and after loading, so regular
            # torch.load is the compatible local-artifact path.
            predictor.initialize_from_trained_model_folder(
                str(model_dir), use_folds=("all",), checkpoint_name=checkpoint_name
            )
    finally:
        torch.load = original_load
        prediction_module.recursive_find_python_class = _ORIGINAL_RESOLVER


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def expected_output_name(image: Path) -> str:
    suffix = "_0000.nii.gz"
    if not image.name.endswith(suffix):
        raise ValueError(f"invalid challenge image name: {image.name}")
    return image.name[: -len(suffix)] + ".nii.gz"


def validate_dataset_role_path(images: Path, role: str) -> None:
    """Fail closed when an image directory does not match its declared role."""

    parts = {part.casefold() for part in images.resolve().parts}
    public = "validation-public-images" in parts
    hidden = "validation-hidden-images" in parts
    validation = "validation" in parts or public or hidden
    if role == "public" and not public:
        raise ValueError("public role requires Validation-Public-Images")
    if role == "hidden" and not hidden:
        raise ValueError("hidden role requires Validation-Hidden-Images")
    if role == "training_complete" and validation:
        raise ValueError("training_complete role forbids every validation path")


def validate_shard_outputs(
    inventory: list[Path], output: Path, part_id: int, num_parts: int
) -> list[str]:
    expected = [
        expected_output_name(path)
        for path in inventory[part_id::num_parts]
    ]
    observed = sorted(path.name for path in output.glob("*.nii.gz"))
    # The output directory is shared by disjoint ranks, so require this rank's
    # files to exist but allow the other rank to have published its files too.
    if not set(expected).issubset(observed):
        raise ValueError(
            f"STU-Net shard output mismatch: missing={sorted(set(expected)-set(observed))}"
        )
    for source in inventory[part_id::num_parts]:
        destination = output / expected_output_name(source)
        reference = nib.load(str(source))
        prediction = nib.load(str(destination))
        values = np.asanyarray(prediction.dataobj)
        if prediction.shape != reference.shape or not np.allclose(
            prediction.affine, reference.affine, atol=1e-4
        ):
            raise ValueError(f"STU-Net geometry mismatch: {source.name}")
        if not np.isin(np.unique(values), [0, 1]).all():
            raise ValueError(f"STU-Net output is nonbinary: {destination.name}")
    return expected


def compact_probability_exports(cases: list[str], output: Path) -> list[str]:
    """Keep one uint16 lesion channel in NIfTI axis order.

    nnU-Net exports two redundant float32 softmax channels in array (z,y,x)
    order. The compact file is deterministic, four times smaller before
    compression, and is the exact artifact used by both public selection and
    hidden replay.
    """

    compacted = []
    for name in cases:
        stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name
        raw = output / f"{stem}.npz"
        properties = output / f"{stem}.pkl"
        compact = output / f"{stem}.prob16.npz"
        reference = nib.load(str(output / name))
        if compact.is_file():
            with np.load(compact) as check:
                restored = check["lesion_probability_uint16"]
                if restored.dtype != np.uint16 or restored.shape != reference.shape:
                    raise ValueError(f"invalid existing compact probability: {name}")
            # A killed exporter can leave a corrupt raw archive beside an
            # already validated compact artifact. The compact file is the
            # deployed format and is authoritative within this checkpoint-
            # specific output directory; remove only the stale redundant files.
            if raw.is_file():
                raw.unlink()
            if properties.is_file():
                properties.unlink()
            compacted.append(compact.name)
            continue
        if not raw.is_file():
            raise ValueError(f"missing probability export: {raw}")
        with np.load(raw) as archive:
            probabilities = archive["probabilities"]
            if probabilities.ndim != 4 or probabilities.shape[0] != 2:
                raise ValueError(f"invalid probability shape: {raw}")
            lesion = np.asarray(probabilities[1], dtype=np.float32)
        if lesion.shape == reference.shape[::-1]:
            lesion = lesion.transpose(2, 1, 0)
        elif lesion.shape != reference.shape:
            raise ValueError(
                f"probability/NIfTI geometry mismatch: {name} "
                f"{lesion.shape} != {reference.shape}"
            )
        binary = np.asanyarray(reference.dataobj).astype(bool, copy=False)
        if not np.array_equal(lesion > 0.5, binary):
            raise ValueError(
                f"channel/orientation probability contract failed at 0.50: {name}"
            )
        quantized = np.rint(np.clip(lesion, 0.0, 1.0) * 65535.0).astype(np.uint16)
        temporary = compact.with_name(f".{compact.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, lesion_probability_uint16=quantized)
        with np.load(temporary) as check:
            restored = check["lesion_probability_uint16"]
            if restored.dtype != np.uint16 or restored.shape != reference.shape:
                raise ValueError(f"compact probability validation failed: {name}")
        os.replace(temporary, compact)
        raw.unlink()
        if properties.is_file():
            properties.unlink()
        compacted.append(compact.name)
    return compacted


def reusable_compact_outputs(cases: list[str], output: Path) -> bool:
    """Return true only when every mask and compact probability already exist."""

    return bool(cases) and all(
        (output / name).is_file()
        and (
            output
            / f"{name[:-len('.nii.gz')] if name.endswith('.nii.gz') else name}.prob16.npz"
        ).is_file()
        for name in cases
    )


def sequential_image_lists(
    inventory: list[Path], part_id: int, num_parts: int
) -> list[list[str]]:
    """Select the same shard as nnU-Net without its multiprocessing pools."""

    return [[str(path)] for path in inventory[part_id::num_parts]]


def validate_tile_step_size(value: float) -> float:
    """Validate nnU-Net's sliding-window step without changing its default."""

    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError("tile step size must be in (0, 1]")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--part-id", required=True, type=int)
    parser.add_argument("--num-parts", default=2, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--direct-compact-probabilities", action="store_true")
    parser.add_argument("--mirroring", action="store_true")
    parser.add_argument(
        "--tile-step-size",
        default=0.5,
        type=float,
        help="nnU-Net sliding-window step; 0.5 preserves the validated baseline",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="run preprocessing and export in-process to minimize host-memory peaks",
    )
    parser.add_argument(
        "--dataset-role",
        choices=("public", "hidden", "training_complete"),
        required=True,
    )
    parser.add_argument("--terminal", required=True, type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.num_parts < 1 or not 0 <= args.part_id < args.num_parts:
        raise ValueError("invalid workers or shard contract")
    args.tile_step_size = validate_tile_step_size(args.tile_step_size)
    if args.direct_compact_probabilities and not args.save_probabilities:
        raise ValueError("direct compact export requires --save-probabilities")
    validate_dataset_role_path(args.images, args.dataset_role)
    inventory = sorted(args.images.glob("*_0000.nii.gz"))
    if not inventory:
        raise FileNotFoundError(f"no challenge images in {args.images}")
    fold = args.model_dir / "fold_all"
    checkpoint = fold / args.checkpoint
    for path in (args.model_dir / "dataset.json", args.model_dir / "plans.json", checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha_before = sha256_file(checkpoint)
    if args.sequential:
        # GPU sliding-window inference does not benefit from a large CPU pool,
        # and full-volume probability resampling can otherwise create many
        # simultaneous allocator arenas on high-resolution CTs.
        torch.set_num_threads(args.workers)
        torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    expected_cases = [
        expected_output_name(path) for path in inventory[args.part_id :: args.num_parts]
    ]
    reused_existing_outputs = args.save_probabilities and reusable_compact_outputs(
        expected_cases, args.output
    )
    if reused_existing_outputs:
        cases = validate_shard_outputs(
            inventory, args.output, args.part_id, args.num_parts
        )
        probability_files = compact_probability_exports(cases, args.output)
        checkpoint_sha_after = checkpoint_sha_before
        if "STUNetTrainerPanCancerComponentBoxE160SourceStratified__" in args.model_dir.name:
            trainer_name = COMPONENT_BOX_E160_SOURCE_STRATIFIED_TRAINER_NAME
        elif "STUNetTrainerPanCancerComponentBoxE160CompleteHybrid__" in args.model_dir.name:
            trainer_name = COMPONENT_BOX_E160_COMPLETE_HYBRID_TRAINER_NAME
        elif "STUNetTrainerPanCancerComponentBoxE160Balanced__" in args.model_dir.name:
            trainer_name = COMPONENT_BOX_E160_BALANCED_TRAINER_NAME
        elif "STUNetTrainerPanCancerComponentBox__" in args.model_dir.name:
            trainer_name = COMPONENT_BOX_TRAINER_NAME
        elif "STUNetTrainerPanCancer__" in args.model_dir.name:
            trainer_name = TRAINER_NAME
        else:
            raise ValueError(
                f"cannot safely infer trainer from model directory: {args.model_dir}"
            )
    else:
        if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
            raise RuntimeError(f"CUDA GPU {args.gpu} unavailable")
        torch.cuda.set_device(args.gpu)
        predictor = nnUNetPredictor(
            tile_step_size=args.tile_step_size,
            use_gaussian=True,
            use_mirroring=args.mirroring,
            perform_everything_on_device=True,
            device=torch.device("cuda", args.gpu),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        safe_initialize(predictor, args.model_dir, args.checkpoint)
        trainer_name = predictor.trainer_name
        if trainer_name not in TRAINER_CLASSES:
            raise ValueError(f"unexpected trainer in checkpoint: {trainer_name}")
        original_export = prediction_module.export_prediction_from_logits
        selected_export = (
            export_prediction_from_logits_compact
            if args.direct_compact_probabilities
            else original_export
        )
        if args.sequential:
            def sequential_export(*export_args, **export_kwargs):
                export_kwargs["num_threads_torch"] = args.workers
                return selected_export(*export_args, **export_kwargs)

            prediction_module.export_prediction_from_logits = sequential_export
        elif args.direct_compact_probabilities:
            prediction_module.export_prediction_from_logits = selected_export
        try:
            if args.sequential:
                predictor.predict_from_files_sequential(
                    sequential_image_lists(inventory, args.part_id, args.num_parts),
                    str(args.output),
                    save_probabilities=args.save_probabilities,
                    overwrite=False,
                )
            else:
                predictor.predict_from_files(
                    str(args.images),
                    str(args.output),
                    save_probabilities=args.save_probabilities,
                    overwrite=False,
                    num_processes_preprocessing=args.workers,
                    num_processes_segmentation_export=args.workers,
                    num_parts=args.num_parts,
                    part_id=args.part_id,
                )
        finally:
            prediction_module.export_prediction_from_logits = original_export
        cases = validate_shard_outputs(
            inventory, args.output, args.part_id, args.num_parts
        )
        checkpoint_sha_after = sha256_file(checkpoint)
        if checkpoint_sha_after != checkpoint_sha_before:
            raise ValueError("STU-Net checkpoint changed during inference")
        probability_files = (
            compact_probability_exports(cases, args.output)
            if args.save_probabilities
            else []
        )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "model": (
            "STU-Net-B component-box pan-cancer"
            if trainer_name == COMPONENT_BOX_TRAINER_NAME
            else "STU-Net-B partial-box pan-cancer"
        ),
        "model_dir": str(args.model_dir.resolve()),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha_after,
        },
        "trainer": trainer_name,
        "gpu": args.gpu,
        "part_id": args.part_id,
        "num_parts": args.num_parts,
        "case_count": len(cases),
        "cases": cases,
        "save_probabilities": args.save_probabilities,
        "probability_format": (
            "uint16_lesion_probability_in_nifti_axis_order"
            if args.save_probabilities
            else None
        ),
        "probability_files": probability_files,
        "reused_existing_outputs": reused_existing_outputs,
        "direct_compact_probabilities": args.direct_compact_probabilities,
        "mirroring": args.mirroring,
        "tile_step_size": args.tile_step_size,
        "sequential": args.sequential,
        "export_threads": args.workers if args.sequential else None,
        "dataset_role": args.dataset_role,
        "training_images_read": args.dataset_role == "training_complete",
        "public_images_read": args.dataset_role == "public",
        "public_labels_read": False,
        "hidden_read": args.dataset_role == "hidden",
        "hidden_labels_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.terminal, payload)


if __name__ == "__main__":
    main()
