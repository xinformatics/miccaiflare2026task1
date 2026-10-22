#!/usr/bin/env python3
"""Generate label-blind high-sensitivity pulmonary blob proposals from CT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def approximate_lung_region(ct: np.ndarray, dilation_voxels: int) -> np.ndarray:
    body = np.zeros_like(ct, dtype=bool)
    structure = ndi.generate_binary_structure(2, 1)
    for index in range(ct.shape[2]):
        tissue = ndi.binary_closing(ct[:, :, index] > -500.0, structure=structure, iterations=2)
        tissue = _largest_component(tissue)
        body[:, :, index] = ndi.binary_fill_holes(tissue)
    lung_air = body & (ct < -350.0)
    labels, count = ndi.label(lung_air)
    if count:
        sizes = np.bincount(labels.ravel())
        keep = [
            int(value)
            for value in np.argsort(sizes[1:])[::-1][:4] + 1
            if sizes[int(value)] >= 200
        ]
        lung_air = np.isin(labels, keep)
    else:
        lung_air = np.zeros_like(body)
    return ndi.binary_dilation(lung_air, iterations=dilation_voxels) & body


def blob_candidates(
    ct: np.ndarray,
    spacing: np.ndarray,
    target_spacing: float,
    sigmas_mm: tuple[float, ...],
    response_threshold: float,
    nms_distance_mm: float,
    maximum_candidates: int,
) -> list[tuple[float, np.ndarray]]:
    zoom = spacing / float(target_spacing)
    sampled = ndi.zoom(ct.astype(np.float32, copy=False), zoom, order=1, mode="nearest", prefilter=False)
    lung = approximate_lung_region(sampled, max(1, round(10.0 / target_spacing)))
    bright = np.clip((sampled + 1000.0) / 1200.0, 0.0, 1.0)
    candidates: list[tuple[float, np.ndarray]] = []
    for sigma_mm in sigmas_mm:
        sigma = sigma_mm / target_spacing
        response = -ndi.gaussian_laplace(bright, sigma=sigma, mode="nearest") * sigma**2
        maxima = response == ndi.maximum_filter(response, size=3, mode="nearest")
        maxima &= lung & (sampled > -850.0) & (response >= response_threshold)
        coordinates = np.argwhere(maxima)
        for coordinate in coordinates:
            candidates.append((float(response[tuple(coordinate)]), coordinate.astype(np.float64)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[float, np.ndarray]] = []
    minimum_squared = (nms_distance_mm / target_spacing) ** 2
    for score, coordinate in candidates:
        if all(float(np.sum((coordinate - prior[1]) ** 2)) >= minimum_squared for prior in selected):
            selected.append((score, coordinate))
            if len(selected) >= maximum_candidates:
                break
    # Convert downsampled voxel coordinates back to original voxel axes.
    return [(score, coordinate / zoom) for score, coordinate in selected]


def paint_sphere(mask: np.ndarray, center: np.ndarray, spacing: np.ndarray, radius_mm: float) -> None:
    radius_voxels = np.ceil(radius_mm / spacing).astype(int)
    rounded = np.rint(center).astype(int)
    lower = np.maximum(rounded - radius_voxels, 0)
    upper = np.minimum(rounded + radius_voxels + 1, np.asarray(mask.shape))
    grids = np.ogrid[tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))]
    distance = np.zeros(tuple((upper - lower).tolist()), dtype=np.float32)
    for axis, grid in enumerate(grids):
        distance += ((grid - float(center[axis])) * float(spacing[axis])) ** 2
    mask[tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))] |= distance <= radius_mm**2


def _run_case(task: dict[str, object]) -> dict[str, object]:
    image_path = Path(str(task["image"]))
    prediction_name = str(task["prediction_name"])
    output_root = Path(str(task["output_root"]))
    top_k = tuple(int(value) for value in task["top_k"])
    image = nib.load(str(image_path))
    ct = np.asanyarray(image.dataobj, dtype=np.float32)
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    candidates = blob_candidates(
        ct,
        spacing,
        float(task["target_spacing"]),
        tuple(float(value) for value in task["sigmas_mm"]),
        float(task["response_threshold"]),
        float(task["nms_distance_mm"]),
        max(top_k),
    )
    for count in top_k:
        mask = np.zeros(image.shape, dtype=bool)
        for _, center in candidates[:count]:
            paint_sphere(mask, center, spacing, float(task["sphere_radius_mm"]))
        destination = output_root / f"top{count:03d}" / prediction_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.nii.gz")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), image.affine, header), str(temporary))
        os.replace(temporary, destination)
    return {
        "case": prediction_name,
        "candidates": len(candidates),
        "maximum_score": candidates[0][0] if candidates else None,
        "image_sha256": sha256_file(image_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", default="50,100,200")
    parser.add_argument("--target-spacing", type=float, default=2.5)
    parser.add_argument("--sigmas-mm", default="2.5,5.0,7.5")
    parser.add_argument("--response-threshold", type=float, default=0.025)
    parser.add_argument("--nms-distance-mm", type=float, default=6.0)
    parser.add_argument("--sphere-radius-mm", type=float, default=4.0)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    top_k = tuple(sorted({int(value) for value in args.top_k.split(",")}))
    sigmas_mm = tuple(float(value) for value in args.sigmas_mm.split(","))
    if not top_k or min(top_k) < 1 or args.workers < 1:
        raise ValueError("invalid candidate counts or worker count")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = [item for item in manifest["items"] if str(item.get("cohort")) == args.cohort]
    if len(items) != args.expected_count or any(item.get("complete_label") is not True for item in items):
        raise ValueError(f"cohort coverage/completeness mismatch: {len(items)}/{args.expected_count}")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "image": item["image"], "prediction_name": item["prediction_name"],
            "output_root": str(args.output), "top_k": top_k,
            "target_spacing": args.target_spacing, "sigmas_mm": sigmas_mm,
            "response_threshold": args.response_threshold,
            "nms_distance_mm": args.nms_distance_mm,
            "sphere_radius_mm": args.sphere_radius_mm,
        }
        for item in items
    ]
    if args.workers == 1:
        rows = [_run_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_run_case, tasks))
    expected = {str(item["prediction_name"]) for item in items}
    for count in top_k:
        observed = {path.name for path in (args.output / f"top{count:03d}").glob("*.nii.gz")}
        if observed != expected:
            raise ValueError(f"output coverage mismatch top{count}: {len(observed)}/{len(expected)}")
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "multiscale_scale_normalized_3d_LoG_with_heuristic_lung_mask",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "cohort": args.cohort,
        "cases": len(items),
        "parameters": {
            "top_k": list(top_k), "target_spacing_mm": args.target_spacing,
            "sigmas_mm": list(sigmas_mm), "response_threshold": args.response_threshold,
            "nms_distance_mm": args.nms_distance_mm, "sphere_radius_mm": args.sphere_radius_mm,
        },
        "candidate_count": {
            "minimum": min(int(row["candidates"]) for row in rows),
            "mean": float(np.mean([int(row["candidates"]) for row in rows])),
            "maximum": max(int(row["candidates"]) for row in rows),
        },
        "labels_read": False,
        "public_or_hidden_data_read": False,
        "rows": rows,
    }
    marker = args.output / "PROPOSALS_COMPLETE.json"
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
