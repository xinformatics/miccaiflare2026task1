#!/usr/bin/env python3
"""Challenge inference entry point for the GF-Screen test container."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

APP_ROOT = Path(__file__).resolve().parent
if not (APP_ROOT / "models").is_dir():
    APP_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT))

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)
from scipy import ndimage

from models.GF_Screen_RL import GFScreen_RL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/workspace/inputs"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/outputs"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/model.pt"))
    parser.add_argument("--glance-threshold", type=float, default=0.5)
    parser.add_argument("--lesion-threshold", type=float, default=0.5)
    parser.add_argument("--min-component-mm3", type=float, default=1000.0)
    parser.add_argument("--focus-batch-size", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.75,
                        help="in-plane overlap ratio")
    parser.add_argument("--overlap-z", type=float, default=0.75,
                        help="through-plane overlap ratio")
    parser.add_argument("--large-volume-mvox", type=float, default=30.0,
                        help="use the fast grid above this pre-crop resampled size")
    parser.add_argument("--large-overlap", type=float, default=0.5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    return parser.parse_args()


def output_name(input_name: str) -> str:
    if input_name.endswith("_0000.nii.gz"):
        return input_name[: -len("_0000.nii.gz")] + ".nii.gz"
    return input_name


def filter_components(mask: np.ndarray, voxel_mm3: float, minimum_mm3: float) -> np.ndarray:
    if minimum_mm3 <= 0 or not mask.any():
        return mask
    components, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes_mm3 = np.bincount(components.ravel()) * voxel_mm3
    keep = sizes_mm3 >= minimum_mm3
    keep[0] = False
    return keep[components]


def load_state(path: Path):
    # The packaged checkpoint contains only tensors and is compatible with old
    # torch versions used for CUDA 11.6 challenge hardware.
    checkpoint = torch.load(str(path), map_location="cpu")
    return checkpoint.get("state_dict", checkpoint)


def make_model(args: argparse.Namespace) -> GFScreen_RL:
    model_args = SimpleNamespace(
        glance_backbone="resnet18",
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_checkpoint=False,
        roi_x=96,
        roi_y=96,
        roi_z=64,
        batch_size=1,
        sw_batch_size=1,
        focus_batch_size=args.focus_batch_size,
        glance_threshold=args.glance_threshold,
    )
    model = GFScreen_RL(model_args)
    state = load_state(args.checkpoint)
    model.load_state_dict(state, strict=True)
    del state
    return model.eval().cuda()


def crop_coordinates(
    shape: tuple[int, int, int], overlap_ratios: tuple[float, float, float]
) -> list[tuple[int, int, int]]:
    """Generate GF-Screen's deterministic 96x96x64 inference grid."""
    if any(not 0.0 <= value < 1.0 for value in overlap_ratios):
        raise ValueError(f"overlaps must be in [0, 1), got {overlap_ratios}")
    crop = (96, 96, 64)
    overlap = tuple(int(value * ratio) for value, ratio in zip(crop, overlap_ratios))
    stride = tuple(value - overlap_value for value, overlap_value in zip(crop, overlap))
    counts = tuple(
        (size - overlap_value + stride_value - 1) // stride_value
        for size, overlap_value, stride_value in zip(shape, overlap, stride)
    )
    coordinates = []
    for i in range(counts[0]):
        for j in range(counts[1]):
            for k in range(counts[2]):
                start = [i * stride[0], j * stride[1], k * stride[2]]
                for axis in range(3):
                    if start[axis] + crop[axis] > shape[axis]:
                        start[axis] = shape[axis] - crop[axis]
                coordinates.append(tuple(start))
    return coordinates


def crop_batch(volume: torch.Tensor, coordinates, crop=(96, 96, 64)) -> torch.Tensor:
    patches = []
    for x, y, z in coordinates:
        patches.append(volume[:, x:x + crop[0], y:y + crop[1], z:z + crop[2]])
    return torch.stack(patches, dim=0)


def stream_predict(
    model: GFScreen_RL,
    volume: torch.Tensor,
    glance_threshold: float,
    lesion_threshold: float,
    focus_batch_size: int,
    overlap_ratios: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """Run the same GF grid while avoiding all-crop/all-logit GPU materialization."""
    spatial_shape = tuple(int(value) for value in volume.shape[1:])
    coordinates = crop_coordinates(spatial_shape, overlap_ratios)
    selected = []
    glance_batch_size = 8
    for start in range(0, len(coordinates), glance_batch_size):
        current = coordinates[start:start + glance_batch_size]
        patches = crop_batch(volume, current).cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            actions = model.glance_model.get_actions(patches)
        keep = (actions.softmax(1)[:, 1] > glance_threshold).cpu().tolist()
        selected.extend(coord for coord, use in zip(current, keep) if use)
        del patches, actions

    compression = len(selected) / len(coordinates) if coordinates else 0.0
    probability_sum = np.zeros(spatial_shape, dtype=np.float32)
    coverage = np.zeros(spatial_shape, dtype=np.uint16)
    for start in range(0, len(selected), focus_batch_size):
        current = selected[start:start + focus_batch_size]
        patches = crop_batch(volume, current).cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            foreground = model.segmentation_model(patches).softmax(1)[:, 1]
        foreground = foreground.float().cpu().numpy()
        for patch, (x, y, z) in zip(foreground, current):
            probability_sum[x:x + 96, y:y + 96, z:z + 64] += patch
            coverage[x:x + 96, y:y + 96, z:z + 64] += 1
        del patches, foreground

    return (
        (coverage > 0) & (probability_sum >= lesion_threshold * coverage),
        compression,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Task 1 inference")
    inputs = sorted(path for path in args.input.rglob("*.nii.gz") if path.is_file())
    if not inputs:
        raise RuntimeError(f"No .nii.gz inputs found below {args.input}")
    names = [output_name(path.name) for path in inputs]
    if len(set(names)) != len(names):
        raise RuntimeError("Input basenames are not unique after removing the _0000 suffix")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise RuntimeError(f"Invalid shard {args.shard_id}/{args.num_shards}")
    inputs = inputs[args.shard_id::args.num_shards]
    args.output.mkdir(parents=True, exist_ok=True)

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 3.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175.0, a_max=250.0,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            SpatialPadd(keys=["image"], spatial_size=(96, 96, 64), mode="constant"),
        ]
    )
    inverse = Compose(
        [
            EnsureTyped(keys=["pred"]),
            Invertd(
                keys=["pred"], transform=transforms, orig_keys=["image"],
                meta_keys=["pred_meta_dict"], orig_meta_keys=["image_meta_dict"],
                meta_key_postfix="meta_dict", nearest_interp=True, to_tensor=True,
            ),
        ]
    )
    dataset = Dataset(
        data=[{"image": str(path), "source": str(path)} for path in inputs],
        transform=transforms,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    startup = time.perf_counter()
    model = make_model(args)
    torch.backends.cudnn.benchmark = True
    print(f"Loaded checkpoint in {time.perf_counter() - startup:.2f}s", flush=True)

    loader_iterator = iter(loader)
    with torch.inference_mode():
        for index in range(1, len(dataset) + 1):
            started = time.perf_counter()
            batch = next(loader_iterator)
            source = Path(batch["source"][0])
            destination = args.output / output_name(source.name)
            reference = nib.load(str(source))
            resampled_shape = np.ceil(
                np.asarray(reference.shape)
                * np.asarray(reference.header.get_zooms()[:3])
                / np.asarray((1.0, 1.0, 3.0))
            )
            resampled_mvox = float(np.prod(resampled_shape) / 1e6)
            if resampled_mvox > args.large_volume_mvox:
                overlaps = (args.large_overlap,) * 3
                grid_name = "large-fast"
            else:
                overlaps = (args.overlap, args.overlap, args.overlap_z)
                grid_name = "full"
            mask, compression = stream_predict(
                model,
                batch["image"][0],
                args.glance_threshold,
                args.lesion_threshold,
                args.focus_batch_size,
                overlaps,
            )
            prediction = torch.from_numpy(mask.astype(np.uint8)).unsqueeze(0).unsqueeze(0)
            inverse_batch = dict(batch)
            inverse_batch["pred"] = prediction
            restored = inverse(decollate_batch(inverse_batch)[0])["pred"]
            mask = restored.detach().cpu().numpy().squeeze().astype(bool, copy=False)

            if mask.shape != reference.shape:
                raise RuntimeError(
                    f"Geometry restoration failed for {source.name}: {mask.shape} != {reference.shape}"
                )
            voxel_mm3 = float(abs(np.linalg.det(reference.affine[:3, :3])))
            mask = filter_components(mask, voxel_mm3, args.min_component_mm3)
            header = reference.header.copy()
            header.set_data_dtype(np.uint8)
            temporary = destination.with_name("." + destination.name + ".tmp.nii.gz")
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header), str(temporary))
            os.replace(str(temporary), str(destination))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            print(
                f"[{index:03d}/{len(inputs):03d}] {source.name} -> {destination.name}; "
                f"{elapsed:.2f}s, grid={grid_name}, source={resampled_mvox:.1f}M, "
                f"gate={float(compression):.3f}, voxels={int(mask.sum())}",
                flush=True,
            )
            del prediction, restored
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
