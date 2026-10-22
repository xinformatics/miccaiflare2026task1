#!/usr/bin/env python3
"""Extract frozen Glance CT-patch embeddings for candidate components.

The component identifiers come from an existing component CSV and therefore
remain in the candidate mask's native geometry.  Candidate centroids are
mapped through physical RAS coordinates into a 1x1x3-mm CT volume before a
96x96x64 patch is cropped.  This matches the physical field of view and input
normalization used to train GF-Screen's Glance ResNet-18.

This utility never uses labels.  It can consequently be reused for public
diagnostics, training-proposal mining, and hidden/test inference without
changing feature semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cc3d
import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import torch

from models.glance import Glance_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def centered_crop(
    array: np.ndarray,
    center: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Return a constant-padded crop with exactly ``shape`` voxels."""
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D array, found {array.shape}")
    if any(value <= 0 for value in shape):
        raise ValueError("crop dimensions must be positive")
    start = tuple(int(c) - int(size) // 2 for c, size in zip(center, shape))
    stop = tuple(begin + size for begin, size in zip(start, shape))
    source_start = tuple(max(0, begin) for begin in start)
    source_stop = tuple(min(limit, end) for limit, end in zip(array.shape, stop))
    destination_start = tuple(
        source - begin for source, begin in zip(source_start, start)
    )
    destination_stop = tuple(
        target + source_end - source_begin
        for target, source_begin, source_end in zip(
            destination_start, source_start, source_stop
        )
    )
    output = np.zeros(shape, dtype=np.float32)
    output[
        tuple(slice(a, b) for a, b in zip(destination_start, destination_stop))
    ] = array[tuple(slice(a, b) for a, b in zip(source_start, source_stop))]
    return output


def component_centers(mask: np.ndarray, requested: set[int]) -> dict[int, np.ndarray]:
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    maximum = int(labels.max())
    missing = sorted(value for value in requested if value <= 0 or value > maximum)
    if missing:
        raise ValueError(f"component IDs absent from candidate mask: {missing[:5]}")
    statistics = cc3d.statistics(labels)
    return {
        component_id: np.asarray(statistics["centroids"][component_id], dtype=np.float64)
        for component_id in sorted(requested)
    }


def normalize_checkpoint_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("glance_model."):
            result[key[len("glance_model.") :]] = value
    if not result:
        raise ValueError("checkpoint contains no glance_model parameters")
    return result


def load_glance(checkpoint: Path, device: torch.device) -> Glance_model:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("checkpoint state_dict is not a mapping")
    model = Glance_model(backbone="resnet18")
    missing, _unexpected = model.load_state_dict(
        normalize_checkpoint_state(state), strict=False
    )
    required_missing = [
        key for key in missing if key.startswith("backbone.") or key.startswith("layer.")
    ]
    if required_missing:
        raise ValueError(f"checkpoint is missing Glance parameters: {required_missing[:5]}")
    return model.to(device).eval()


def read_component_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"case", "component_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"component CSV must contain {sorted(required)}")
    keys = [(row["case"], int(row["component_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("component CSV contains duplicate case/component keys")
    return rows


def extract_case(
    case: str,
    component_ids: list[int],
    images: Path,
    candidates: Path,
    model: Glance_model,
    device: torch.device,
    batch_size: int,
    crop_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    candidate_image = nib.load(str(candidates / f"{case}.nii.gz"))
    candidate = np.asanyarray(candidate_image.dataobj).astype(bool)
    centers = component_centers(candidate, set(component_ids))

    ct_image = nib.load(str(images / f"{case}_0000.nii.gz"))
    ct_image = resample_to_output(
        nib.as_closest_canonical(ct_image), voxel_sizes=(1.0, 1.0, 3.0), order=1
    )
    ct = np.nan_to_num(
        np.asanyarray(ct_image.dataobj).astype(np.float32),
        nan=-175.0,
        posinf=250.0,
        neginf=-175.0,
    )
    ct = np.clip((ct + 175.0) / 425.0, 0.0, 1.0)
    inverse_ct_affine = np.linalg.inv(ct_image.affine)

    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    batch: list[np.ndarray] = []

    def flush() -> None:
        if not batch:
            return
        tensor = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            encoded = model.backbone(tensor)
            action = model.layer(encoded)
        features.append(encoded.float().cpu().numpy())
        logits.append(action.float().cpu().numpy())
        batch.clear()

    for component_id in component_ids:
        native = np.asarray([*centers[component_id], 1.0], dtype=np.float64)
        world = candidate_image.affine @ native
        ct_center = inverse_ct_affine @ world
        center = tuple(int(value) for value in np.rint(ct_center[:3]))
        batch.append(centered_crop(ct, center, crop_shape))
        if len(batch) >= batch_size:
            flush()
    flush()
    return np.concatenate(features, axis=0), np.concatenate(logits, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-shape", type=int, nargs=3, default=(96, 96, 64))
    args = parser.parse_args()
    if args.batch_size < 1 or any(value < 16 for value in args.crop_shape):
        parser.error("batch size must be positive and crop dimensions at least 16")

    rows = read_component_rows(args.components)
    cases: dict[str, list[int]] = {}
    for row in rows:
        cases.setdefault(row["case"], []).append(int(row["component_id"]))
    for case in cases:
        for path in (
            args.images / f"{case}_0000.nii.gz",
            args.candidate / f"{case}.nii.gz",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = load_glance(args.checkpoint, device)
    by_key: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for index, (case, component_ids) in enumerate(sorted(cases.items()), 1):
        feature, logit = extract_case(
            case,
            component_ids,
            args.images,
            args.candidate,
            model,
            device,
            args.batch_size,
            tuple(args.crop_shape),
        )
        for component_id, one_feature, one_logit in zip(
            component_ids, feature, logit
        ):
            by_key[(case, component_id)] = (one_feature, one_logit)
        print(f"[{index:02d}/{len(cases)}] {case}: {len(component_ids)}", flush=True)

    ordered = [by_key[(row["case"], int(row["component_id"]))] for row in rows]
    embeddings = np.stack([item[0] for item in ordered]).astype(np.float16)
    logits = np.stack([item[1] for item in ordered]).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        cases=np.asarray([row["case"] for row in rows]),
        component_ids=np.asarray([int(row["component_id"]) for row in rows]),
        embeddings=embeddings,
        logits=logits,
    )
    manifest = {
        "warning": "Label-free frozen feature extraction; no classifier was fitted.",
        "components": len(rows),
        "cases": len(cases),
        "embedding_shape": list(embeddings.shape),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "component_csv": str(args.components.resolve()),
        "component_csv_sha256": sha256(args.components),
        "candidate": str(args.candidate.resolve()),
        "images": str(args.images.resolve()),
        "spacing_mm": [1.0, 1.0, 3.0],
        "crop_shape": list(args.crop_shape),
        "intensity_window_hu": [-175.0, 250.0],
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
