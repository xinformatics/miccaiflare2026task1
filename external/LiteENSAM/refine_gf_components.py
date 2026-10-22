#!/usr/bin/env python3
"""Refine automatic GF-Screen lesion proposals with box-prompt Lite-ENSAM.

The released Lite-ENSAM wrapper is written for the prompted FLARE subtask and
expects RECIST arrays plus optional TorchRL dependencies.  This adapter uses
the released image encoder/decoder directly.  Every connected GF component is
converted to the same longest-diameter prompt used during Lite-ENSAM training,
then decoded independently so distant proposals cannot force destructive
whole-volume downsampling.

The script writes three label-free, reproducible candidates (Lite only, union,
and intersection) and a component agreement table.  Public labels are never
read here; scoring and selection remain separate operations.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.model.registry import model_registry  # noqa: E402
from src.transform.transform import VolumeTransforms  # noqa: E402
from src.utils.decode import decoder_forward  # noqa: E402


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR / "pretrained_models/lite_ensam_flare25.pth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--a-min", type=float, default=-175.0)
    parser.add_argument("--a-max", type=float, default=250.0)
    parser.add_argument("--min-proposal-mm3", type=float, default=0.0)
    parser.add_argument("--size-threshold", type=int, default=128**3)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--continue-prediction", action="store_true")
    return parser.parse_args()


def longest_diameter_prompt(component: np.ndarray) -> np.ndarray:
    """Return the training-compatible [2, 3] (z, y, x) diameter prompt."""
    if component.ndim != 3 or not component.any():
        raise ValueError("component must be a non-empty 3D mask")
    z = int(component.sum(axis=(1, 2)).argmax())
    points = np.argwhere(component[z])
    if len(points) == 1:
        point = np.asarray([z, *points[0]], dtype=np.float32)
        return np.stack([point, point])
    try:
        hull_points = points[ConvexHull(points).vertices]
    except QhullError:
        hull_points = points
    squared_distance = (
        (hull_points[:, None, :] - hull_points[None, :, :]) ** 2
    ).sum(axis=-1)
    first, second = np.unravel_index(
        int(squared_distance.argmax()), squared_distance.shape
    )
    return np.asarray(
        [[z, *hull_points[first]], [z, *hull_points[second]]],
        dtype=np.float32,
    )


def window_to_uint8(image_hu: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max <= a_min:
        raise ValueError("HU window must be finite with a_max > a_min")
    scaled = np.clip((image_hu.astype(np.float32) - a_min) / (a_max - a_min), 0, 1)
    return np.rint(255.0 * scaled).astype(np.uint8)


def dice_pair(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(first.sum()) + int(second.sum())
    return 1.0 if denominator == 0 else 2.0 * int(np.logical_and(first, second).sum()) / denominator


def decode_component(
    model: torch.nn.Module,
    image_uint8: np.ndarray,
    component: np.ndarray,
    device: torch.device,
    size_threshold: int,
) -> np.ndarray:
    prompt = torch.from_numpy(longest_diameter_prompt(component)).unsqueeze(0).to(device)
    transform = VolumeTransforms(size_threshold)
    # The upstream transform prints every crop shape; keep batch logs compact.
    with contextlib.redirect_stdout(io.StringIO()):
        volume, transformed_prompt, _ = transform.forward(
            torch.from_numpy(image_uint8).float().to(device), prompt
        )
    device_type = device.type
    autocast = (
        torch.autocast(device_type=device_type)
        if device_type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        embeddings = model.segresnet(volume.unsqueeze(0).unsqueeze(0))
        logits, _ = decoder_forward(model, embeddings, boxes=transformed_prompt)
    logits = transform.backward(logits.squeeze(1).float()).squeeze(0)
    return logits.gt(0).cpu().numpy()


def proposal_name_to_image_name(name: str) -> str:
    if not name.endswith(".nii.gz"):
        raise ValueError(f"unexpected proposal filename: {name}")
    return name.removesuffix(".nii.gz") + "_0000.nii.gz"


def write_mask(mask: np.ndarray, reference: sitk.Image, destination: Path) -> None:
    output = sitk.GetImageFromArray(mask.astype(np.uint8, copy=False))
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(destination), useCompression=True)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_registry["ahus_model_rope_mixed"]()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device)

    output_dirs = {
        mode: args.output_root / mode for mode in ("lite", "union", "intersection")
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_root / "components.csv"
    fieldnames = [
        "case",
        "component",
        "proposal_voxels",
        "proposal_mm3",
        "lite_voxels",
        "intersection_voxels",
        "union_voxels",
        "pair_dice",
        "lite_to_proposal_volume",
        "seconds",
    ]
    existing_rows: list[dict[str, str]] = []
    if args.continue_prediction and audit_path.exists():
        with audit_path.open(newline="") as stream:
            existing_rows = list(csv.DictReader(stream))

    proposals = sorted(args.proposals.glob("*.nii.gz"))
    if args.max_cases:
        proposals = proposals[: args.max_cases]
    if not proposals:
        raise ValueError(f"no NIfTI proposals found in {args.proposals}")

    rows = existing_rows
    for case_index, proposal_path in enumerate(proposals, start=1):
        destinations = {mode: path / proposal_path.name for mode, path in output_dirs.items()}
        if args.continue_prediction and all(path.is_file() for path in destinations.values()):
            print(f"[{case_index}/{len(proposals)}] skip {proposal_path.name}", flush=True)
            continue
        image_path = args.images / proposal_name_to_image_name(proposal_path.name)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_itk = sitk.ReadImage(str(image_path))
        proposal_itk = sitk.ReadImage(str(proposal_path))
        if image_itk.GetSize() != proposal_itk.GetSize():
            raise ValueError(f"geometry size mismatch for {proposal_path.name}")
        image_hu = sitk.GetArrayFromImage(image_itk)
        proposal = sitk.GetArrayFromImage(proposal_itk) > 0
        image_uint8 = window_to_uint8(image_hu, args.a_min, args.a_max)
        voxel_mm3 = float(np.prod(image_itk.GetSpacing()))
        component_map, component_count = ndimage.label(proposal, structure=CONNECTIVITY_26)
        lite = np.zeros_like(proposal)
        kept = 0
        case_start = time.time()
        for component_id in range(1, component_count + 1):
            component = component_map == component_id
            proposal_voxels = int(component.sum())
            proposal_mm3 = proposal_voxels * voxel_mm3
            if proposal_mm3 < args.min_proposal_mm3:
                continue
            kept += 1
            started = time.time()
            refined = decode_component(
                model, image_uint8, component, device, args.size_threshold
            )
            lite |= refined
            intersection_voxels = int(np.logical_and(component, refined).sum())
            rows.append(
                {
                    "case": proposal_path.name,
                    "component": str(component_id),
                    "proposal_voxels": str(proposal_voxels),
                    "proposal_mm3": f"{proposal_mm3:.6f}",
                    "lite_voxels": str(int(refined.sum())),
                    "intersection_voxels": str(intersection_voxels),
                    "union_voxels": str(int(np.logical_or(component, refined).sum())),
                    "pair_dice": f"{dice_pair(component, refined):.8f}",
                    "lite_to_proposal_volume": f"{int(refined.sum()) / max(1, proposal_voxels):.8f}",
                    "seconds": f"{time.time() - started:.4f}",
                }
            )
        masks = {
            "lite": lite,
            "union": np.logical_or(proposal, lite),
            "intersection": np.logical_and(proposal, lite),
        }
        for mode, mask in masks.items():
            write_mask(mask, proposal_itk, destinations[mode])
        with audit_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"[{case_index}/{len(proposals)}] {proposal_path.name}: "
            f"components={kept} seconds={time.time() - case_start:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
