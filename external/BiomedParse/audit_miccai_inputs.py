#!/usr/bin/env python3
"""Fail-closed audit for NIfTI-adapted BiomedParse inference inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_case(task: tuple[str, str, tuple[str, ...]]) -> dict:
    image_text, input_text, prompts = task
    image_path, input_path = Path(image_text), Path(input_text)
    reference = sitk.ReadImage(str(image_path))
    expected_shape = tuple(reversed(reference.GetSize()))
    expected_spacing = np.asarray(reference.GetSpacing(), dtype=np.float32)[::-1]
    with np.load(input_path, allow_pickle=True) as archive:
        if set(archive.files) != {"imgs", "spacing", "text_prompts"}:
            raise ValueError(f"unexpected fields: {input_path} {archive.files}")
        image = archive["imgs"]
        spacing = archive["spacing"]
        prompt_map = archive["text_prompts"].item()
    expected_prompts = {str(index + 1): value for index, value in enumerate(prompts)}
    expected_prompts["instance_label"] = 0
    if image.dtype != np.uint8 or image.shape != expected_shape:
        raise ValueError(f"image dtype/shape mismatch: {input_path}")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or not np.allclose(
        spacing, expected_spacing, rtol=0.0, atol=1e-5
    ):
        raise ValueError(f"spacing mismatch: {input_path}")
    if prompt_map != expected_prompts:
        raise ValueError(f"prompt mismatch: {input_path} {prompt_map}")
    return {
        "name": input_path.name,
        "bytes": input_path.stat().st_size,
        "sha256": sha256_file(input_path),
        "shape": list(image.shape),
    }


def audit_inputs(
    manifest_path: Path,
    cohort: str,
    images: Path,
    inputs: Path,
    prompts: tuple[str, ...],
    expected_cases: int,
    workers: int,
) -> dict:
    if not prompts or workers < 1 or expected_cases < 1:
        raise ValueError("prompts, workers, and expected cases must be positive")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [item for item in payload.get("items", []) if item.get("cohort") == cohort]
    if len(selected) != expected_cases or any(
        item.get("complete_label") is not True for item in selected
    ):
        raise ValueError(
            f"cohort manifest mismatch: cohort={cohort} cases={len(selected)}"
        )
    expected_images = {Path(str(item["image"])).name for item in selected}
    observed_images = {path.name for path in images.glob("*.nii.gz")}
    expected_inputs = {
        name[:-7] + ".npz" if name.endswith(".nii.gz") else Path(name).stem + ".npz"
        for name in expected_images
    }
    observed_inputs = {path.name for path in inputs.glob("*.npz")}
    if observed_images != expected_images or observed_inputs != expected_inputs:
        raise ValueError(
            f"coverage mismatch: images={len(observed_images)}/{len(expected_images)} "
            f"inputs={len(observed_inputs)}/{len(expected_inputs)}"
        )
    tasks = []
    for item in sorted(selected, key=lambda row: str(row["name"])):
        image_name = Path(str(item["image"])).name
        input_name = image_name[:-7] + ".npz"
        tasks.append((str(images / image_name), str(inputs / input_name), prompts))
    if workers == 1:
        records = [audit_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(audit_case, tasks))
    digest = hashlib.sha256()
    total_bytes = 0
    for record in sorted(records, key=lambda row: row["name"]):
        total_bytes += int(record["bytes"])
        digest.update(record["name"].encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "cohort": cohort,
        "cases": len(records),
        "prompts": list(prompts),
        "window_hu": [-160.0, 240.0],
        "exact_file_coverage": True,
        "all_inputs_uint8_3d": True,
        "all_shapes_and_spacings_match": True,
        "all_prompt_maps_match": True,
        "labels_read": False,
        "bytes": total_bytes,
        "sha256_inventory": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_inputs(
        args.manifest, args.cohort, args.images, args.inputs, tuple(args.prompt),
        args.expected_cases, args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
