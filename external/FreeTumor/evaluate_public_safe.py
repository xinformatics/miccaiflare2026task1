#!/usr/bin/env python3
"""Shardable challenge-image inference for the published FreeTumor model.

The upstream evaluator uses unrestricted ``torch.load`` and a MONAI constructor
signature removed by the installed release. This wrapper preserves its exact
1.5-mm preprocessing and tumor-channel mapping while restricting checkpoint
deserialization and recording a hash-bound provenance terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.core.multiarray as numpy_multiarray
import torch
from monai import transforms
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from torch.serialization import safe_globals


EXPECTED_CHECKPOINT_SHA256 = (
    "c039619094c8b9b38a06bd62d8b5c8f088507cfa39419120ec8beaba255c05a0"
)
TUMOR_LABELS = (2, 4, 6)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_output_name(image_name: str) -> str:
    suffix = "_0000.nii.gz"
    if not image_name.endswith(suffix):
        raise ValueError(f"unexpected challenge image name: {image_name}")
    return image_name[: -len(suffix)] + ".nii.gz"


def shard_inventory(images: Path, shard_index: int, num_shards: int) -> list[Path]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard index/count")
    inventory = sorted(images.glob("*_0000.nii.gz"))
    if not inventory:
        raise FileNotFoundError(f"no challenge images in {images}")
    return inventory[shard_index::num_shards]


def tumor_mask_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5 or logits.shape[1] != 7:
        raise ValueError("FreeTumor logits must have shape [B, 7, X, Y, Z]")
    labels = logits.argmax(dim=1)
    mask = torch.zeros_like(labels, dtype=torch.uint8)
    for label in TUMOR_LABELS:
        mask |= labels.eq(label)
    return mask


def load_checkpoint_safely(path: Path) -> dict:
    observed = sha256_file(path)
    if observed != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"FreeTumor checkpoint SHA256 mismatch: {observed}")
    # The legacy checkpoint contains only the two globals reported by
    # get_unsafe_globals_in_checkpoint. Bind the historical module name
    # explicitly; unrestricted pickle execution remains disabled.
    allowed = [
        (numpy_multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
        type(np.dtype(np.float64)),
    ]
    with safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise ValueError("FreeTumor checkpoint lacks a state_dict")
    return checkpoint


def build_model(checkpoint: dict) -> SwinUNETR:
    kwargs = {
        "in_channels": 1,
        "out_channels": 7,
        "feature_size": 48,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "dropout_path_rate": 0.0,
        "use_checkpoint": True,
        "use_v2": True,
    }
    try:
        model = SwinUNETR(img_size=(96, 96, 96), **kwargs)
    except TypeError:
        model = SwinUNETR(**kwargs)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=-175.0,
                a_max=250.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.SpatialPadd(
                keys=["image"], spatial_size=(96, 96, 96), mode="constant"
            ),
        ]
    )


def save_original_geometry(
    item: dict, inverse: transforms.Compose, source: Path, destination: Path
) -> None:
    inverted = inverse(item)
    prediction = inverted["pred"]
    array = prediction[0].detach().cpu().numpy().astype(np.uint8, copy=False)
    reference = nib.load(str(source))
    if array.shape != reference.shape:
        raise ValueError(
            f"inverse shape mismatch for {source.name}: {array.shape} != {reference.shape}"
        )
    if array.max(initial=0) > 1:
        raise ValueError(f"nonbinary prediction for {source.name}")
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp.nii.gz"
    )
    nib.save(nib.Nifti1Image(array, reference.affine, header), str(temporary))
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--num-shards", default=2, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--sw-batch-size", default=4, type=int)
    parser.add_argument("--overlap", default=0.75, type=float)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument(
        "--dataset-role",
        choices=("public", "hidden"),
        default="public",
        help="records truthful provenance; labels are never accepted or read",
    )
    args = parser.parse_args()
    if args.workers < 0 or args.sw_batch_size < 1:
        raise ValueError("workers must be nonnegative and sw-batch-size positive")
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("overlap must lie in [0, 1)")
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError(f"CUDA GPU {args.gpu} is unavailable")

    inventory = shard_inventory(args.images, args.shard_index, args.num_shards)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_rows = [
        {"image": str(path.resolve()), "source": str(path.resolve())}
        for path in inventory
    ]
    preprocessing = build_transforms()
    loader = DataLoader(
        Dataset(dataset_rows, preprocessing),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    inverse = transforms.Compose(
        [
            transforms.Invertd(
                keys=["pred"],
                transform=preprocessing,
                orig_keys=["image"],
                nearest_interp=True,
                to_tensor=True,
            )
        ]
    )

    torch.cuda.set_device(args.gpu)
    checkpoint = load_checkpoint_safely(args.checkpoint)
    model = build_model(checkpoint).to(args.gpu).eval()
    inferer = partial(
        sliding_window_inference,
        roi_size=(96, 96, 96),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.overlap,
    )
    torch.backends.cudnn.benchmark = True

    written = []
    with torch.no_grad():
        for batch in loader:
            source = Path(batch["source"][0])
            destination = args.output / case_output_name(source.name)
            image = batch["image"].to(args.gpu)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = inferer(image)
            batch["pred"] = tumor_mask_from_logits(logits).unsqueeze(1)
            item = decollate_batch(batch)[0]
            save_original_geometry(item, inverse, source, destination)
            written.append(destination)
            print(f"[{len(written)}/{len(inventory)}] {destination.name}", flush=True)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "model": "FreeTumor published abdominal baseline",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "embedded_epoch": int(checkpoint.get("epoch", -1)),
            "embedded_best_acc": float(checkpoint.get("best_acc", 0.0)),
        },
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "case_count": len(written),
        "cases": [path.name for path in written],
        "preprocessing": {
            "spacing_mm": [1.5, 1.5, 1.5],
            "hu_window": [-175.0, 250.0],
            "roi": [96, 96, 96],
            "overlap": args.overlap,
            "tumor_channels": list(TUMOR_LABELS),
        },
        "dataset_role": args.dataset_role,
        "public_images_read": args.dataset_role == "public",
        "public_labels_read": False,
        "hidden_read": args.dataset_role == "hidden",
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.terminal, payload)


if __name__ == "__main__":
    main()
