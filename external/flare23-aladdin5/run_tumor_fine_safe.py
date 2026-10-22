#!/usr/bin/env python3
"""Safely evaluate the supplied Aladdin5 FLARE23 ROI + tumor checkpoints.

This adapter reproduces the released ROI extraction, tumor-fine, organ-fine,
and merge stages for binary tumor evaluation. It preserves the authors'
preprocessing, ROI crop, sliding-window parameters, class-14 merge rule, and
image geometry while loading all three serialized models through PyTorch's
restricted weights-only loader.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
INFERENCE = HERE / "Inference"
VENDOR = HERE / "vendor"
sys.path.insert(0, str(INFERENCE))
# Append rather than prepend: use the validated environment NumPy and only
# resolve the isolated fastremap wheel from this directory.
sys.path.append(str(VENDOR))

from nnunet.network.FabiansUNet import (  # noqa: E402
    BasicResidualBlock,
    ConvDropoutNormReLU,
    FabiansUNet,
    PlainConvUNetDecoder,
    ResidualLayer,
    ResidualUNetEncoder,
    StackedConvLayers,
)
from nnunet.network.BaseUNet import (  # noqa: E402
    ConvDropoutNormNonlin,
    Generic_UNet,
    StackedConvLayers as BaseStackedConvLayers,
)
from nnunet.network.RepVGGFast import (  # noqa: E402
    RepVGGBInferenceBlock,
    RepVGGLayerInference,
    RepVGGUNetDecoder,
    RepVGGUNetEncoder,
    VGGFastUNet,
)
from nnunet.network.neural_network import SegmentationNetwork  # noqa: E402
from nnunet.utilities.utilities_stuff import Identity, Upsample  # noqa: E402


TARGET_SPACING_ZYX = np.asarray([4.0, 1.2, 1.2], dtype=np.float64)
PATCH_SIZE_ZYX = (32, 128, 192)
LOWER_HU = -967.0
UPPER_HU = 291.0
MEAN = 83.560104
STD = 136.6215
TUMOR_CLASS = 14
ROI_PATCH_SIZE_ZYX = (80, 128, 128)
ROI_SLIDING_THRESHOLD = 240
ROI_PADDING_LOWER_MM_ZYX = np.asarray([40.0, 5.0, 5.0])
ROI_PADDING_UPPER_MM_ZYX = np.asarray([20.0, 5.0, 5.0])
ORGAN_TARGET_SPACING_ZYX = np.asarray(
    [2.5, 0.81835938, 0.81835938], dtype=np.float64
)
ORGAN_PATCH_SIZE_ZYX = (64, 128, 224)
ORGAN_LOWER_HU = -965.0
ORGAN_UPPER_HU = 276.0
ORGAN_MEAN = 79.64796
ORGAN_STD = 139.33594


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_globals() -> list:
    common = [
        FabiansUNet,
        ConvDropoutNormReLU,
        ResidualLayer,
        BasicResidualBlock,
        StackedConvLayers,
        ResidualUNetEncoder,
        PlainConvUNetDecoder,
        Generic_UNet,
        ConvDropoutNormNonlin,
        BaseStackedConvLayers,
        Identity,
        Upsample,
        VGGFastUNet,
        RepVGGUNetEncoder,
        RepVGGUNetDecoder,
        RepVGGBInferenceBlock,
        RepVGGLayerInference,
        SegmentationNetwork,
        torch.nn.LeakyReLU,
        torch.nn.Sequential,
        torch.nn.InstanceNorm3d,
        torch.nn.BatchNorm3d,
        torch.nn.Conv3d,
        torch.nn.ModuleList,
        torch.nn.ConvTranspose3d,
        torch.nn.MaxPool3d,
        builtins.getattr,
        np.ndarray,
        np.dtype,
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
    ]
    common.extend(
        {
            type(np.dtype(dtype))
            for dtype in (
                np.float16,
                np.float32,
                np.float64,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.bool_,
            )
        }
    )
    return common


def load_tumor_model(path: Path, gpu: int) -> FabiansUNet:
    with torch.serialization.safe_globals(safe_globals()):
        model = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(model, FabiansUNet):
        raise TypeError(f"unexpected checkpoint type: {type(model)!r}")
    if model.num_classes != 15:
        raise ValueError(f"expected 15 classes, found {model.num_classes}")
    if sum(parameter.numel() for parameter in model.parameters()) != 71_656_768:
        raise ValueError("FLARE23 tumor-fine parameter count mismatch")
    model.cuda(gpu)
    model.eval()
    model.do_ds = False
    return model


def load_roi_model(path: Path, gpu: int) -> VGGFastUNet:
    with torch.serialization.safe_globals(safe_globals()):
        model = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(model, VGGFastUNet):
        raise TypeError(f"unexpected ROI checkpoint type: {type(model)!r}")
    if model.num_classes != 2:
        raise ValueError(f"expected two ROI classes, found {model.num_classes}")
    if sum(parameter.numel() for parameter in model.parameters()) != 11_182_816:
        raise ValueError("FLARE23 ROI parameter count mismatch")
    model.cuda(gpu)
    model.eval()
    return model


def load_organ_model(path: Path, gpu: int) -> Generic_UNet:
    with torch.serialization.safe_globals(safe_globals()):
        model = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(model, Generic_UNet):
        raise TypeError(f"unexpected organ checkpoint type: {type(model)!r}")
    if model.num_classes != 15:
        raise ValueError(f"expected 15 organ classes, found {model.num_classes}")
    if sum(parameter.numel() for parameter in model.parameters()) != 85_984_000:
        raise ValueError("FLARE23 organ-fine parameter count mismatch")
    model.cuda(gpu)
    model.eval()
    model.do_ds = False
    return model


def resample_image(
    data_czyx: np.ndarray,
    source_spacing_zyx: np.ndarray,
    target_spacing_zyx: np.ndarray = TARGET_SPACING_ZYX,
) -> np.ndarray:
    shape = np.asarray(data_czyx.shape[1:], dtype=np.int64)
    target_shape = np.maximum(
        1, np.rint(source_spacing_zyx / target_spacing_zyx * shape).astype(np.int64)
    )
    tensor = torch.from_numpy(data_czyx).unsqueeze(0).float()
    result = F.interpolate(
        tensor, size=tuple(int(value) for value in target_shape),
        mode="trilinear", align_corners=False,
    )[0]
    return result.numpy()


def extract_roi_crop(
    model: VGGFastUNet,
    data_czyx: np.ndarray,
    spacing_zyx: np.ndarray,
) -> tuple[tuple[slice, slice, slice] | None, dict]:
    step = np.asarray(
        [max(1, int(np.rint(5.0 / spacing_zyx[0]))), 4, 4], dtype=np.int64
    )
    data_roi = data_czyx[
        :, :: int(step[0]), :: int(step[1]), :: int(step[2])
    ].copy()
    data_roi = np.clip(data_roi, -1024.0, 1024.0)
    data_roi = (data_roi - data_roi.mean()) / (data_roi.std() + 1e-8)
    with torch.inference_mode():
        if data_roi.shape[1] > ROI_SLIDING_THRESHOLD:
            roi = model.predict_3D(
                data_roi, step_size=1.0, patch_size=list(ROI_PATCH_SIZE_ZYX)
            )
        else:
            roi = model.predict_3D(
                data_roi, step_size=1.0, sliding=False, patch_size=None
            )
    roi = np.asarray(roi) > 0
    points = np.argwhere(roi)
    if points.size == 0:
        return None, {
            "roi_shape_zyx": list(roi.shape),
            "roi_foreground_voxels": 0,
            "crop_zyx": None,
        }

    lower_roi = points.min(axis=0).astype(np.float64)
    upper_roi = (points.max(axis=0) + 1).astype(np.float64)
    original_shape = np.asarray(data_czyx.shape[1:], dtype=np.int64)
    ratio = original_shape.astype(np.float64) / np.asarray(roi.shape, dtype=np.float64)
    lower = np.floor(lower_roi * ratio).astype(np.int64)
    upper = np.ceil(upper_roi * ratio).astype(np.int64)
    lower_padding = np.rint(ROI_PADDING_LOWER_MM_ZYX / spacing_zyx).astype(np.int64)
    upper_padding = np.rint(ROI_PADDING_UPPER_MM_ZYX / spacing_zyx).astype(np.int64)
    lower = np.maximum(0, lower - lower_padding)
    upper = np.minimum(original_shape, upper + upper_padding)
    if np.any(upper <= lower):
        raise RuntimeError(f"invalid ROI crop: lower={lower}, upper={upper}")
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    return crop, {
        "roi_shape_zyx": list(roi.shape),
        "roi_foreground_voxels": int(roi.sum()),
        "crop_zyx": [[int(lo), int(hi)] for lo, hi in zip(lower, upper)],
    }


def predict_case(
    tumor_model: FabiansUNet,
    roi_model: VGGFastUNet,
    organ_model: Generic_UNet | None,
    source: Path,
    destination: Path,
    gpu: int,
) -> dict:
    image = sitk.ReadImage(str(source))
    data = sitk.GetArrayFromImage(image)[None].astype(np.float32, copy=False)
    direction = image.GetDirection()
    reverse_z = bool(direction[-1] < 0)
    if reverse_z:
        data = data[:, ::-1].copy()
    original_shape = tuple(int(value) for value in data.shape[1:])
    spacing_zyx = np.asarray(image.GetSpacing(), dtype=np.float64)[[2, 1, 0]]
    crop, roi_record = extract_roi_crop(roi_model, data, spacing_zyx)
    mask = np.zeros(original_shape, dtype=np.uint8)
    confidence = {
        "tumor_probability_max": 0.0,
        "tumor_probability_mean_positive": 0.0,
        "tumor_probability_q10_positive": 0.0,
        "tumor_probability_q50_positive": 0.0,
        "tumor_margin_mean_positive": 0.0,
        "tumor_margin_q10_positive": 0.0,
        "tumor_fraction_crop": 0.0,
    }
    if crop is not None:
        crop_shape = tuple(int(section.stop - section.start) for section in crop)
        crop_data = data[(slice(None), *crop)]
        tumor_data = resample_image(crop_data, spacing_zyx)
        tumor_data = np.clip(tumor_data, LOWER_HU, UPPER_HU)
        tumor_data = (tumor_data - MEAN) / STD

        with torch.inference_mode():
            probabilities = tumor_model.predict_3D(
                tumor_data,
                step_size=0.5,
                patch_size=list(PATCH_SIZE_ZYX),
            )
            if probabilities.ndim != 4 or probabilities.shape[0] != 15:
                raise ValueError(
                    f"unexpected probability shape: {tuple(probabilities.shape)}"
                )
            probabilities = F.interpolate(
                probabilities[None].half(),
                size=crop_shape,
                mode="trilinear",
                align_corners=True,
            )[0]
            tumor_segmentation = torch.argmax(probabilities, dim=0).byte().cpu().numpy()
            tumor_only_mask = tumor_segmentation == TUMOR_CLASS
            p_tumor = probabilities[TUMOR_CLASS].float()
            p_other = probabilities[:TUMOR_CLASS].float().amax(dim=0)
            confidence["tumor_probability_max"] = float(p_tumor.max().item())
            confidence["tumor_fraction_crop"] = float(tumor_only_mask.mean())
            positive_tensor = torch.from_numpy(tumor_only_mask).to(p_tumor.device)
            if bool(positive_tensor.any()):
                positive_probability = p_tumor[positive_tensor]
                positive_margin = (p_tumor - p_other)[positive_tensor]
                confidence.update(
                    {
                        "tumor_probability_mean_positive": float(
                            positive_probability.mean().item()
                        ),
                        "tumor_probability_q10_positive": float(
                            torch.quantile(positive_probability, 0.10).item()
                        ),
                        "tumor_probability_q50_positive": float(
                            torch.quantile(positive_probability, 0.50).item()
                        ),
                        "tumor_margin_mean_positive": float(
                            positive_margin.mean().item()
                        ),
                        "tumor_margin_q10_positive": float(
                            torch.quantile(positive_margin, 0.10).item()
                        ),
                    }
                )

            if organ_model is None:
                crop_mask = tumor_only_mask.astype(np.uint8)
                organ_data = organ_probabilities = organ_segmentation = None
            else:
                organ_data = resample_image(
                    crop_data, spacing_zyx, ORGAN_TARGET_SPACING_ZYX
                )
                organ_data = np.clip(organ_data, ORGAN_LOWER_HU, ORGAN_UPPER_HU)
                organ_data = (organ_data - ORGAN_MEAN) / ORGAN_STD
                organ_probabilities = organ_model.predict_3D(
                    organ_data,
                    step_size=0.8,
                    patch_size=list(ORGAN_PATCH_SIZE_ZYX),
                )
                if (
                    organ_probabilities.ndim != 4
                    or organ_probabilities.shape[0] != 15
                ):
                    raise ValueError(
                        "unexpected organ probability shape: "
                        f"{tuple(organ_probabilities.shape)}"
                    )
                organ_probabilities = F.interpolate(
                    organ_probabilities[None].half(),
                    size=crop_shape,
                    mode="trilinear",
                    align_corners=True,
                )[0]
                organ_segmentation = (
                    torch.argmax(organ_probabilities, dim=0).byte().cpu().numpy()
                )

                # Exact released merge: organ class 14 is accepted only where
                # tumor also predicts 14; tumor class 14 then overwrites organ.
                organ_tumor = organ_segmentation > 13
                organ_segmentation[organ_tumor] = tumor_segmentation[organ_tumor]
                organ_segmentation[tumor_segmentation > 13] = TUMOR_CLASS
                crop_mask = (organ_segmentation == TUMOR_CLASS).astype(np.uint8)
                if not np.array_equal(crop_mask, tumor_only_mask):
                    raise RuntimeError(
                        "official organ/tumor merge changed binary tumor mask"
                    )
        mask[crop] = crop_mask
        del probabilities, tumor_data, crop_data, crop_mask, tumor_only_mask
        del tumor_segmentation, positive_tensor, p_tumor, p_other
        if organ_model is not None:
            del organ_probabilities, organ_data, organ_segmentation
    del data
    torch.cuda.empty_cache()
    if reverse_z:
        mask = mask[::-1].copy()

    output = sitk.GetImageFromArray(mask)
    output.CopyInformation(image)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.nii.gz")
    sitk.WriteImage(output, str(temporary), True)
    os.replace(temporary, destination)
    return {
        "case": destination.name,
        "shape_zyx": list(mask.shape),
        "foreground_voxels": int(mask.sum()),
        "confidence": confidence,
        **roi_record,
    }


def validate_role(images: Path, role: str) -> None:
    parts = {part.casefold() for part in images.resolve().parts}
    public = "validation-public-images" in parts
    hidden = "validation-hidden-images" in parts
    if role == "public" and not public:
        raise ValueError("public role requires Validation-Public-Images")
    if role == "hidden" and not hidden:
        raise ValueError("hidden role requires Validation-Hidden-Images")
    if role == "training" and (public or hidden):
        raise ValueError("training role cannot use validation images")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--roi-checkpoint", required=True, type=Path)
    parser.add_argument("--organ-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--omit-organ-inference",
        action="store_true",
        help="Skip binary-equivalent organ inference when only tumor confidence is needed",
    )
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument(
        "--role", required=True, choices=("public", "hidden", "training")
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--shard-count", default=1, type=int)
    parser.add_argument("--terminal", required=True, type=Path)
    args = parser.parse_args()
    validate_role(args.images, args.role)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError(f"CUDA GPU {args.gpu} unavailable")
    sources = sorted(args.images.glob("*_0000.nii.gz"))
    sources = sources[args.shard_index :: args.shard_count]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        sources = sources[: args.limit]
    if not sources:
        raise FileNotFoundError("no challenge input images")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(args.checkpoint)
    roi_checkpoint_sha = sha256_file(args.roi_checkpoint)
    organ_checkpoint_sha = sha256_file(args.organ_checkpoint)
    torch.cuda.set_device(args.gpu)
    tumor_model = load_tumor_model(args.checkpoint, args.gpu)
    roi_model = load_roi_model(args.roi_checkpoint, args.gpu)
    organ_model = (
        None
        if args.omit_organ_inference
        else load_organ_model(args.organ_checkpoint, args.gpu)
    )
    records = []
    for index, source in enumerate(sources, start=1):
        destination = args.output / source.name.replace("_0000.nii.gz", ".nii.gz")
        if destination.is_file():
            destination.unlink()
        records.append(
            predict_case(
                tumor_model, roi_model, organ_model, source, destination, args.gpu
            )
        )
        print(
            f"FLARE23 full-cascade progress {index}/{len(sources)}: {destination.name}",
            flush=True,
        )
    if sha256_file(args.checkpoint) != checkpoint_sha:
        raise RuntimeError("checkpoint changed during inference")
    if sha256_file(args.roi_checkpoint) != roi_checkpoint_sha:
        raise RuntimeError("ROI checkpoint changed during inference")
    if sha256_file(args.organ_checkpoint) != organ_checkpoint_sha:
        raise RuntimeError("organ checkpoint changed during inference")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "model": "Aladdin5 FLARE23 ROI + tumor-fine + organ-fine cascade",
        "full_released_binary_tumor_cascade_reproduced": not args.omit_organ_inference,
        "organ_inference_omitted_as_binary_equivalent": args.omit_organ_inference,
        "missing_released_stages": [],
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha,
        },
        "roi_checkpoint": {
            "path": str(args.roi_checkpoint.resolve()),
            "sha256": roi_checkpoint_sha,
        },
        "organ_checkpoint": {
            "path": str(args.organ_checkpoint.resolve()),
            "sha256": organ_checkpoint_sha,
        },
        "architecture_parameters": {
            "tumor": 71_656_768,
            "roi": 11_182_816,
            "organ": 85_984_000,
        },
        "preprocessing": {
            "target_spacing_zyx_mm": TARGET_SPACING_ZYX.tolist(),
            "patch_size_zyx": list(PATCH_SIZE_ZYX),
            "hu_clip": [LOWER_HU, UPPER_HU],
            "mean": MEAN,
            "std": STD,
            "step_size": 0.5,
            "tumor_class": TUMOR_CLASS,
            "roi": {
                "downsample_step_zyx": "[round(5mm / z-spacing), 4, 4]",
                "hu_clip": [-1024, 1024],
                "normalization": "per-case mean/std",
                "sliding_threshold_z": ROI_SLIDING_THRESHOLD,
                "sliding_patch_zyx": list(ROI_PATCH_SIZE_ZYX),
                "padding_lower_mm_zyx": ROI_PADDING_LOWER_MM_ZYX.tolist(),
                "padding_upper_mm_zyx": ROI_PADDING_UPPER_MM_ZYX.tolist(),
            },
            "organ": {
                "target_spacing_zyx_mm": ORGAN_TARGET_SPACING_ZYX.tolist(),
                "patch_size_zyx": list(ORGAN_PATCH_SIZE_ZYX),
                "hu_clip": [ORGAN_LOWER_HU, ORGAN_UPPER_HU],
                "mean": ORGAN_MEAN,
                "std": ORGAN_STD,
                "step_size": 0.8,
                "binary_tumor_merge": "exactly equivalent to tumor-fine class 14",
            },
        },
        "dataset_role": args.role,
        "case_count": len(records),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "cases": records,
        "public_labels_read": False,
        "hidden_read": args.role == "hidden",
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.terminal, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
