#!/usr/bin/env python3
"""Apply a qualified gate to traceable GF components of an immutable anchor."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np

from subtractive_gate_features import FEATURES, extract_component_features, feature_matrix


_BUNDLE: dict[str, object] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image(images: Path, mask_name: str) -> Path:
    case = mask_name.removesuffix(".nii.gz")
    candidates = (images / mask_name, images / f"{case}_0000.nii.gz")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no CT image for {mask_name} in {images}")


def initialize(model_path: str) -> None:
    global _BUNDLE
    _BUNDLE = joblib.load(model_path)


def one(task: tuple[str, str, str, str, str]) -> dict[str, object]:
    raw_text, anchor_text, images_text, output_text, name = task
    if _BUNDLE is None:
        raise RuntimeError("gate model was not initialized")
    raw_path = Path(raw_text) / name
    anchor_path = Path(anchor_text) / name
    images = Path(images_text)
    output = Path(output_text)
    raw_image = nib.load(str(raw_path))
    anchor_image = nib.load(str(anchor_path))
    ct_path = resolve_image(images, name)
    ct_image = nib.load(str(ct_path))
    if (
        raw_image.shape != anchor_image.shape
        or raw_image.shape != ct_image.shape
        or not np.allclose(raw_image.affine, anchor_image.affine, atol=1e-3)
        or not np.allclose(raw_image.affine, ct_image.affine, atol=1e-3)
    ):
        raise ValueError(f"geometry mismatch: {name}")
    raw = np.asarray(raw_image.dataobj) > 0
    anchor = np.asarray(anchor_image.dataobj) > 0
    ct = np.asarray(ct_image.dataobj)
    rows, component_map = extract_component_features(
        raw,
        ct,
        raw_image.affine,
        minimum_component_mm3=float(_BUNDLE["minimum_component_mm3"]),
    )
    deleted: list[int] = []
    probabilities: list[float] = []
    if rows:
        probability = _BUNDLE["model"].predict_proba(feature_matrix(rows))[:, 1]
        probabilities = [float(value) for value in probability]
        deleted = [
            int(row["comp"])
            for row, value in zip(rows, probability)
            if float(value) >= float(_BUNDLE["threshold"])
            and float(row["mm3"])
            > float(_BUNDLE["minimum_deletion_mm3_exclusive"])
        ]
    deletion_mask = np.isin(component_map, deleted)
    candidate = anchor & ~deletion_mask
    if np.any(candidate & ~anchor):
        raise AssertionError("subtractive gate added foreground")
    destination = output / name
    temporary = destination.with_name(f".{name}.{os.getpid()}.nii.gz")
    result_image = nib.Nifti1Image(
        candidate.astype(np.uint8), anchor_image.affine, anchor_image.header
    )
    result_image.set_data_dtype(np.uint8)
    nib.save(result_image, str(temporary))
    os.replace(temporary, destination)
    return {
        "case": name.removesuffix(".nii.gz"),
        "raw_components": len(rows),
        "deleted_components": len(deleted),
        "deleted_anchor_voxels": int(np.count_nonzero(anchor & deletion_mask)),
        "anchor_voxels": int(anchor.sum()),
        "candidate_voxels": int(candidate.sum()),
        "maximum_fp_probability": max(probabilities, default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-gf", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fit-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.expected_count < 1 or args.workers < 1:
        parser.error("invalid expected count or workers")

    marker = json.loads(args.fit_marker.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "COMPLETE"
        or marker.get("model") != str(args.model.resolve())
        or marker.get("model_sha256") != sha256(args.model)
        or float(marker.get("minimum_deletion_mm3_exclusive", -1)) != 2000.0
        or marker.get("public_or_hidden_data_read") is not False
    ):
        raise SystemExit("invalid qualified final-fit marker")
    bundle = joblib.load(args.model)
    if (
        tuple(bundle.get("features", ())) != FEATURES
        or float(bundle.get("minimum_deletion_mm3_exclusive", -1)) != 2000.0
    ):
        raise SystemExit("invalid gate model bundle")

    names = sorted(path.name for path in args.anchor.glob("*.nii.gz"))
    raw_names = {path.name for path in args.raw_gf.glob("*.nii.gz")}
    if len(names) != args.expected_count or set(names) != raw_names:
        raise ValueError(
            f"anchor/raw coverage mismatch: anchor={len(names)} raw={len(raw_names)}"
        )
    for name in names:
        resolve_image(args.images, name)
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(args.raw_gf), str(args.anchor), str(args.images), str(args.output), name)
        for name in names
    ]
    with Pool(args.workers, initializer=initialize, initargs=(str(args.model),)) as pool:
        rows = []
        for index, row in enumerate(pool.imap(one, tasks, chunksize=1), 1):
            rows.append(row)
            if index % 10 == 0:
                print(f"[{index}/{len(tasks)}] traceable gate", flush=True)
    observed = {path.name for path in args.output.glob("*.nii.gz")}
    if observed != set(names):
        raise RuntimeError("incomplete subtractive-gate outputs")
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "cases": len(rows),
        "raw_gf": str(args.raw_gf.resolve()),
        "anchor": str(args.anchor.resolve()),
        "images": str(args.images.resolve()),
        "output": str(args.output.resolve()),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "threshold": float(bundle["threshold"]),
        "minimum_component_mm3": float(bundle["minimum_component_mm3"]),
        "minimum_deletion_mm3_exclusive": float(
            bundle["minimum_deletion_mm3_exclusive"]
        ),
        "deleted_components": sum(int(row["deleted_components"]) for row in rows),
        "changed_cases": sum(int(row["deleted_components"]) > 0 for row in rows),
        "deleted_anchor_voxels": sum(int(row["deleted_anchor_voxels"]) for row in rows),
        "case_summaries": rows,
        "labels_read": False,
        "public_or_hidden_data_read": args.expected_count in (50, 100),
        "submission_created": False,
    }
    marker_path = args.output / "GATE_COMPLETE.json"
    temporary = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, marker_path)
    print(json.dumps({key: payload[key] for key in payload if key != "case_summaries"}, indent=2))


if __name__ == "__main__":
    main()
