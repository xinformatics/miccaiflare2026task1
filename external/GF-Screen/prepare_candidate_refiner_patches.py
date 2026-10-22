#!/usr/bin/env python3
"""Build label-safe, prompt-conditioned 3-D candidate patches.

The input proposal masks must have been produced without consulting the
labels. Labels are used only here to assign candidate-level targets and a
voxel refinement target. Complete-label cases may contribute positives and
negatives. Explicit positive-only cases may contribute overlapping positives
only; unmatched proposals remain unknown and are never materialized as
negatives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage

from candidate_bbox_tiles import (
    component_world_bounds,
    in_crop_target,
    tile_centers_world,
)
from proposal_component_instances import source_separated_instances


PATCH_VERSION = "candidate-refiner-v4-iso15-preserved-seed-target"
TILED_PATCH_VERSION = "candidate-refiner-v5-bbox24-tiled-all-volume"


def strip_nii(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def parse_triplet(value: str, cast) -> tuple:
    values = tuple(cast(item) for item in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return values


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def prediction_path(directory: Path, prediction_name: str) -> Path:
    path = directory / prediction_name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def same_grid(reference: nib.Nifti1Image, other: nib.Nifti1Image) -> bool:
    return reference.shape[:3] == other.shape[:3] and np.allclose(
        reference.affine, other.affine, atol=1e-3, rtol=0
    )


def crop_affine(
    center_world: np.ndarray,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag(np.asarray(spacing, dtype=np.float64))
    affine[:3, 3] = center_world - np.asarray(spacing) * (np.asarray(shape) - 1.0) / 2.0
    return affine


def slice_affine(
    source_affine: np.ndarray, slices: tuple[slice, slice, slice]
) -> np.ndarray:
    """Return the world affine for a spatially sliced source array."""
    starts = np.asarray(
        [0 if value.start is None else int(value.start) for value in slices],
        dtype=np.float64,
    )
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = starts
    return np.asarray(source_affine, dtype=np.float64) @ translation


def pad_slices(
    slices: tuple[slice, slice, slice],
    shape: tuple[int, int, int],
    padding: int = 1,
) -> tuple[slice, slice, slice]:
    """Pad a component bbox so linear interpolation matches a full zero field."""
    if padding < 0:
        raise ValueError("padding must be nonnegative")
    return tuple(
        slice(
            max(0, int(value.start) - padding),
            min(int(size), int(value.stop) + padding),
        )
        for value, size in zip(slices, shape)
    )


def resample_array(
    array: np.ndarray,
    source_affine: np.ndarray,
    target_affine: np.ndarray,
    shape: tuple[int, int, int],
    order: int,
    cval: float,
) -> np.ndarray:
    mapping = np.linalg.inv(source_affine) @ target_affine
    return ndimage.affine_transform(
        np.asarray(array),
        matrix=mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=shape,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=order > 1,
    )


def resample_seed_prompt(
    component_support: np.ndarray,
    source_affine: np.ndarray,
    target_affine: np.ndarray,
    shape: tuple[int, int, int],
    source_count: int,
) -> tuple[np.ndarray, bool]:
    """Resample a small component without allowing its prompt to disappear."""
    if source_count < 1:
        raise ValueError("source_count must be positive")
    strength = float(np.max(component_support)) / source_count
    seed = resample_array(
        component_support.astype(np.float32) / source_count,
        source_affine,
        target_affine,
        shape,
        1,
        0.0,
    ).astype(np.float32)
    fallback = not bool(np.any(seed > 1e-6))
    if fallback:
        # An even-sized crop is centred between two voxels per axis. Preserve
        # the physical centroid as a 2x2x2 point prompt for sub-voxel seeds.
        indices = [slice((size - 1) // 2, size // 2 + 1) for size in shape]
        seed[tuple(indices)] = max(strength, 1.0 / source_count)
    return seed, fallback


def resample_binary_target(
    component_target: np.ndarray,
    source_affine: np.ndarray,
    target_affine: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, bool]:
    """Nearest-resample a target while preserving an in-crop sub-voxel lesion."""
    target = resample_array(
        component_target.astype(np.uint8),
        source_affine,
        target_affine,
        shape,
        0,
        0.0,
    ) > 0
    fallback = bool(component_target.any()) and not bool(target.any())
    if fallback:
        source_coordinates = np.argwhere(component_target)
        center_world = nib.affines.apply_affine(
            source_affine, source_coordinates.mean(axis=0)
        )
        target_coordinate = nib.affines.apply_affine(
            np.linalg.inv(target_affine), center_world
        )
        target_index = np.rint(target_coordinate).astype(int)
        if np.any(target_index < 0) or np.any(target_index >= np.asarray(shape)):
            raise ValueError(
                "positive target lies outside its candidate-centred crop: "
                f"target_coordinate={target_coordinate.tolist()} shape={shape}"
            )
        target[tuple(target_index)] = True
    return target, fallback


def component_records(
    union: np.ndarray,
    support_count: np.ndarray,
    target: np.ndarray,
    spacing: np.ndarray,
    min_mm3: float,
    max_mm3: float,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    labels, count = cc3d.connected_components(
        union.astype(np.uint8), connectivity=26, return_N=True
    )
    gt_labels = cc3d.connected_components(target.astype(np.uint8), connectivity=26)
    distance = ndimage.distance_transform_edt(~target, sampling=spacing)
    statistics = cc3d.statistics(labels)
    records: list[dict[str, object]] = []
    voxel_mm3 = float(np.prod(spacing))
    for component_id in range(1, count + 1):
        voxels = int(statistics["voxel_counts"][component_id])
        volume = voxels * voxel_mm3
        if not min_mm3 <= volume <= max_mm3:
            continue
        bbox = pad_slices(statistics["bounding_boxes"][component_id], labels.shape)
        local_component = labels[bbox] == component_id
        overlap_ids = np.unique(gt_labels[bbox][local_component])
        overlap_ids = overlap_ids[overlap_ids > 0]
        intersection = int(np.count_nonzero(target[bbox][local_component]))
        support_values = support_count[bbox][local_component]
        local_distance = distance[bbox][local_component]
        records.append(
            {
                "component_id": component_id,
                "bbox": bbox,
                "centroid_voxel": statistics["centroids"][component_id],
                "target_component_ids": overlap_ids.astype(np.int32).tolist(),
                "voxels": voxels,
                "volume_mm3": volume,
                "positive": bool(overlap_ids.size),
                "intersection_voxels": intersection,
                "seed_overlap_fraction": intersection / max(voxels, 1),
                "distance_to_target_mm": 0.0 if overlap_ids.size else float(local_distance.min()),
                "mean_source_support": float(support_values.mean()),
                "max_source_support": int(support_values.max()),
            }
        )
    return records, labels, gt_labels


def source_separated_component_records(
    source_masks: list[np.ndarray],
    support_count: np.ndarray,
    target: np.ndarray,
    spacing: np.ndarray,
    min_mm3: float,
    max_mm3: float,
) -> tuple[list[dict[str, object]], list[np.ndarray], np.ndarray]:
    """Build label targets for independently connected proposal instances."""
    component_maps, instances = source_separated_instances(source_masks)
    gt_labels = cc3d.connected_components(target.astype(np.uint8), connectivity=26)
    distance = ndimage.distance_transform_edt(~target, sampling=spacing)
    voxel_mm3 = float(np.prod(spacing))
    records: list[dict[str, object]] = []
    for instance in instances:
        voxels = int(instance["voxels"])
        volume = voxels * voxel_mm3
        if not min_mm3 <= volume <= max_mm3:
            continue
        source_index = int(instance["source_index"])
        source_component_id = int(instance["source_component_id"])
        bbox = pad_slices(tuple(instance["bbox"]), target.shape)
        local_component = (
            component_maps[source_index][bbox] == source_component_id
        )
        overlap_ids = np.unique(gt_labels[bbox][local_component])
        overlap_ids = overlap_ids[overlap_ids > 0]
        intersection = int(np.count_nonzero(target[bbox][local_component]))
        support_values = support_count[bbox][local_component]
        local_distance = distance[bbox][local_component]
        records.append(
            {
                "component_id": int(instance["component_id"]),
                "source_index": source_index,
                "source_component_id": source_component_id,
                "equivalent_components": instance["equivalent_components"],
                "bbox": bbox,
                "centroid_voxel": instance["centroid_voxel"],
                "target_component_ids": overlap_ids.astype(np.int32).tolist(),
                "voxels": voxels,
                "volume_mm3": volume,
                "positive": bool(overlap_ids.size),
                "intersection_voxels": intersection,
                "seed_overlap_fraction": intersection / max(voxels, 1),
                "distance_to_target_mm": (
                    0.0 if overlap_ids.size else float(local_distance.min())
                ),
                "mean_source_support": float(support_values.mean()),
                "max_source_support": int(support_values.max()),
            }
        )
    return records, component_maps, gt_labels


def pack_local_component(local: np.ndarray) -> tuple[tuple[int, int, int], bytes]:
    """Pack one bbox-local component without retaining a full label volume."""
    array = np.asarray(local, dtype=bool)
    if array.ndim != 3:
        raise ValueError(f"local component must be 3-D, got {array.shape}")
    shape = tuple(int(value) for value in array.shape)
    packed = np.packbits(array.reshape(-1), bitorder="little").tobytes()
    return shape, packed


def unpack_local_component(record: dict[str, object]) -> np.ndarray:
    """Reconstruct a compact bbox-local component stored on a record."""
    shape = tuple(int(value) for value in record["_component_shape"])
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"invalid compact component shape: {shape}")
    count = int(np.prod(shape, dtype=np.int64))
    packed = np.frombuffer(record["_component_bits"], dtype=np.uint8)
    required = (count + 7) // 8
    if packed.size != required:
        raise ValueError(
            f"compact component byte mismatch: {packed.size} != {required}"
        )
    return np.unpackbits(
        packed, bitorder="little", count=count
    ).reshape(shape).astype(bool, copy=False)


def source_separated_component_records_compact(
    source_masks: Iterable[np.ndarray],
    support_count: np.ndarray,
    target: np.ndarray,
    spacing: np.ndarray,
    min_mm3: float,
    max_mm3: float,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Match source-separated records while retaining no full component maps.

    The legacy implementation keeps one full-volume ``uint32`` label map for
    every proposal source.  With twelve sources this dominates host memory.
    This implementation preserves the same source/component order and exact
    duplicate policy, but stores each unique component as packed bbox-local
    bits and releases a source label map before processing the next source.
    """
    shape = tuple(int(value) for value in np.asarray(target).shape)
    if len(shape) != 3:
        raise ValueError(f"target must be 3-D, got {shape}")
    if tuple(np.asarray(support_count).shape) != shape:
        raise ValueError("support-count shape mismatch")
    gt_labels = cc3d.connected_components(
        np.asarray(target, dtype=np.uint8), connectivity=26
    )
    distance = ndimage.distance_transform_edt(~np.asarray(target, dtype=bool), sampling=spacing)
    voxel_mm3 = float(np.prod(spacing))
    instances: list[dict[str, object]] = []
    fingerprints: dict[bytes, list[int]] = {}
    records: list[dict[str, object]] = []

    source_count = 0
    for source_index, source_mask in enumerate(source_masks):
        if tuple(np.asarray(source_mask).shape) != shape:
            raise ValueError(
                f"proposal source {source_index} shape mismatch: "
                f"{tuple(np.asarray(source_mask).shape)} != {shape}"
            )
        source_count += 1
        component_map, count = cc3d.connected_components(
            np.asarray(source_mask, dtype=np.uint8),
            connectivity=26,
            return_N=True,
        )
        statistics = cc3d.statistics(component_map)
        for source_component_id in range(1, count + 1):
            tight_bbox = tuple(statistics["bounding_boxes"][source_component_id])
            tight_local = component_map[tight_bbox] == source_component_id
            fingerprint = hashlib.sha256()
            bounds = np.asarray(
                [(int(axis.start), int(axis.stop)) for axis in tight_bbox],
                dtype=np.int64,
            )
            tight_shape, tight_bits = pack_local_component(tight_local)
            fingerprint.update(bounds.tobytes())
            fingerprint.update(np.asarray(tight_shape, dtype=np.int64).tobytes())
            fingerprint.update(tight_bits)
            digest = fingerprint.digest()

            duplicate_index = None
            for candidate_index in fingerprints.get(digest, []):
                candidate = instances[candidate_index]
                if (
                    tuple(candidate["bbox"]) == tight_bbox
                    and tuple(candidate["_tight_shape"]) == tight_shape
                    and candidate["_tight_bits"] == tight_bits
                ):
                    duplicate_index = candidate_index
                    break
            identity = {
                "source_index": source_index,
                "source_component_id": source_component_id,
            }
            if duplicate_index is not None:
                instances[duplicate_index]["equivalent_components"].append(identity)
                continue

            equivalent_components = [identity]
            instance = {
                "component_id": len(instances) + 1,
                "source_index": source_index,
                "source_component_id": source_component_id,
                "equivalent_components": equivalent_components,
                "bbox": tight_bbox,
                "_tight_shape": tight_shape,
                "_tight_bits": tight_bits,
            }
            instances.append(instance)
            fingerprints.setdefault(digest, []).append(len(instances) - 1)

            voxels = int(statistics["voxel_counts"][source_component_id])
            volume = voxels * voxel_mm3
            if not min_mm3 <= volume <= max_mm3:
                continue
            bbox = pad_slices(tight_bbox, shape)
            local_component = component_map[bbox] == source_component_id
            component_shape, component_bits = pack_local_component(local_component)
            overlap_ids = np.unique(gt_labels[bbox][local_component])
            overlap_ids = overlap_ids[overlap_ids > 0]
            intersection = int(np.count_nonzero(target[bbox][local_component]))
            support_values = support_count[bbox][local_component]
            local_distance = distance[bbox][local_component]
            records.append(
                {
                    "component_id": int(instance["component_id"]),
                    "source_index": source_index,
                    "source_component_id": source_component_id,
                    "equivalent_components": equivalent_components,
                    "bbox": bbox,
                    "centroid_voxel": np.asarray(
                        statistics["centroids"][source_component_id],
                        dtype=np.float64,
                    ),
                    "target_component_ids": overlap_ids.astype(np.int32).tolist(),
                    "voxels": voxels,
                    "volume_mm3": volume,
                    "positive": bool(overlap_ids.size),
                    "intersection_voxels": intersection,
                    "seed_overlap_fraction": intersection / max(voxels, 1),
                    "distance_to_target_mm": (
                        0.0 if overlap_ids.size else float(local_distance.min())
                    ),
                    "mean_source_support": float(support_values.mean()),
                    "max_source_support": int(support_values.max()),
                    "_component_shape": component_shape,
                    "_component_bits": component_bits,
                }
            )
        del component_map

    if source_count == 0:
        raise ValueError("at least one proposal source is required")

    return records, gt_labels


def deterministic_negative_subset(
    records: list[dict[str, object]],
    case: str,
    maximum: int,
    minimum_per_source: int = 0,
) -> list[dict[str, object]]:
    if minimum_per_source < 0:
        raise ValueError("minimum_per_source must be nonnegative")
    positives = [record for record in records if record["positive"]]
    negatives = [record for record in records if not record["positive"]]
    if maximum >= 0 and len(negatives) > maximum:
        # Split the bounded budget between difficult lesion-adjacent/large
        # negatives and a deterministic diversity sample. Pure hard-negative
        # ranking misses the many remote false-positive phenotypes seen in
        # empty or low-burden deployment cases.
        def digest(record: dict[str, object]) -> str:
            return hashlib.sha256(
                f"{case}:{record['component_id']}".encode("utf-8")
            ).hexdigest()

        def hard_rank(record: dict[str, object]) -> tuple[float, float, str]:
            digest = hashlib.sha256(
                f"{case}:{record['component_id']}".encode("utf-8")
            ).hexdigest()
            return (
                float(record["distance_to_target_mm"]),
                -float(record["volume_mm3"]),
                digest,
            )

        reserved: list[dict[str, object]] = []
        if minimum_per_source:
            by_source: dict[int, list[dict[str, object]]] = {}
            for record in negatives:
                if "source_index" not in record:
                    raise ValueError(
                        "minimum_per_source requires source-separated records"
                    )
                by_source.setdefault(int(record["source_index"]), []).append(record)
            required = sum(min(minimum_per_source, len(rows)) for rows in by_source.values())
            if required > maximum:
                raise ValueError(
                    f"source minimum requires {required} negatives but cap is {maximum}"
                )
            for source in sorted(by_source):
                rows = by_source[source]
                hard_one = sorted(rows, key=hard_rank)[: min(1, minimum_per_source)]
                selected_ids = {int(record["component_id"]) for record in hard_one}
                diversity = sorted(
                    [record for record in rows if int(record["component_id"]) not in selected_ids],
                    key=digest,
                )[: max(0, minimum_per_source - len(hard_one))]
                reserved.extend(hard_one + diversity)

        reserved_ids = {int(record["component_id"]) for record in reserved}
        pool = [
            record for record in negatives
            if int(record["component_id"]) not in reserved_ids
        ]
        remaining_budget = maximum - len(reserved)
        hard_count = min(max(1, remaining_budget // 2), remaining_budget)
        hard = sorted(pool, key=hard_rank)[:hard_count]
        hard_ids = reserved_ids | {int(record["component_id"]) for record in hard}
        remaining = [
            record for record in negatives if int(record["component_id"]) not in hard_ids
        ]
        diverse = sorted(remaining, key=digest)[: remaining_budget - len(hard)]
        negatives = reserved + hard + diverse
    return sorted(
        positives + negatives,
        key=lambda record: int(record["component_id"]),
    )


def atomic_npz(
    path: Path,
    *,
    compressed: bool = True,
    durable: bool = True,
    **arrays: np.ndarray,
) -> None:
    """Atomically persist arrays, optionally avoiding CPU-heavy DEFLATE.

    Prepared training patches retain the historical compressed default. Large
    inference sweeps can opt into ZIP_STORED: probability tensors are dense,
    compress poorly, and are consumed only through ``numpy.load``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        writer = np.savez_compressed if compressed else np.savez
        writer(stream, **arrays)
        stream.flush()
        if durable:
            os.fsync(stream.fileno())
    os.replace(temporary, path)


def task_fingerprint(task: dict[str, object]) -> str:
    """Hash every case-level input and recipe field except resume control."""
    payload = {key: value for key, value in task.items() if key != "resume"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def case_rows_sidecar(output: Path, case: str) -> Path:
    digest = hashlib.sha256(case.encode("utf-8")).hexdigest()[:16]
    return output / "case_rows" / f"{digest}.json"


def load_resumable_case(
    task: dict[str, object], output: Path, case: str, cohort: str
) -> dict[str, object] | None:
    """Return a completed case only when recipe and every atomic patch match."""
    sidecar = case_rows_sidecar(output, case)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid resumable case sidecar: {sidecar}") from error
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported resumable case schema: {sidecar}")
    if payload.get("task_sha256") != task_fingerprint(task):
        return None
    if payload.get("case") != case or payload.get("cohort") != cohort:
        raise RuntimeError(f"resumable case identity mismatch: {sidecar}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"resumable case has no rows: {sidecar}")
    patch_root = (output / "patches").resolve()
    for row in rows:
        if not isinstance(row, dict) or row.get("case") != case:
            raise RuntimeError(f"invalid resumable row: {sidecar}")
        path = Path(str(row.get("path", "")))
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            return None
        if resolved.parent != patch_root or resolved.suffix != ".npz":
            raise RuntimeError(f"resumable patch path escaped output: {path}")
        if resolved.stat().st_size <= 0:
            return None
    return {"case": case, "cohort": cohort, "rows": rows, "resumed": True}


def write_resumable_case(
    task: dict[str, object], output: Path, result: dict[str, object]
) -> None:
    """Commit case rows only after every referenced patch was atomically saved."""
    sidecar = case_rows_sidecar(output, str(result["case"]))
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "task_sha256": task_fingerprint(task),
        "case": result["case"],
        "cohort": result["cohort"],
        "rows": result["rows"],
    }
    temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, sidecar)


def process_case(task: dict[str, object]) -> dict[str, object]:
    item = task["item"]
    assert isinstance(item, dict)
    case = str(item["name"])
    positive_only = item.get("positive_only_label") is True
    if item.get("complete_label") is not True and not positive_only:
        raise ValueError(
            f"case is neither complete-label nor explicit positive-only: {case}"
        )
    if positive_only and item.get("complete_label") is True:
        raise ValueError(f"case has conflicting label contracts: {case}")
    image_path = Path(str(item["image"]))
    label_path = Path(str(item["label"]))
    proposal_dirs = [Path(path) for path in task["proposal_dirs"]]
    source_names = list(task["source_names"])
    output = Path(str(task["output"]))
    resume = bool(task.get("resume", False))
    if resume:
        resumed = load_resumable_case(
            task, output, case, str(item["cohort"])
        )
        if resumed is not None:
            return resumed
    shape = tuple(task["shape"])
    crop_spacing = tuple(task["crop_spacing"])
    bbox_margin_mm = float(task.get("bbox_margin_mm", 0.0))
    tile_stride_mm = float(task.get("tile_stride_mm", 0.0))
    max_negative_tiles = int(task.get("max_negative_tiles", 1))
    component_policy = str(task.get("component_policy", "union"))
    compact_source_components = bool(task.get("compact_source_components", False))
    if component_policy not in ("union", "source_separate"):
        raise ValueError(f"invalid component policy: {component_policy}")
    if compact_source_components and component_policy != "source_separate":
        raise ValueError(
            "compact source components require component_policy=source_separate"
        )

    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    if not same_grid(image_nii, label_nii):
        raise ValueError(f"image/label geometry mismatch: {case}")
    ct = np.asanyarray(image_nii.dataobj).astype(np.float32)
    target = np.asanyarray(label_nii.dataobj) > 0
    proposal_paths = []
    source_masks = [] if not compact_source_components else None
    support_count = np.zeros(image_nii.shape[:3], dtype=np.uint8)
    for directory in proposal_dirs:
        path = prediction_path(directory, str(item["prediction_name"]))
        proposal_nii = nib.load(
            str(path)
        )
        if not same_grid(image_nii, proposal_nii):
            raise ValueError(f"proposal geometry mismatch: {case} {directory}")
        source_mask = np.asanyarray(proposal_nii.dataobj) > 0
        np.add(support_count, source_mask, out=support_count, casting="unsafe")
        proposal_paths.append(path)
        if source_masks is not None:
            source_masks.append(source_mask)
    native_spacing = np.linalg.norm(image_nii.affine[:3, :3], axis=0)
    if component_policy == "union":
        assert source_masks is not None
        union = support_count > 0
        records, component_map, target_component_map = component_records(
            union,
            support_count,
            target,
            native_spacing,
            float(task["min_mm3"]),
            float(task["max_mm3"]),
        )
        component_maps = None
    else:
        if compact_source_components:
            def streamed_source_masks():
                for path in proposal_paths:
                    proposal_nii = nib.load(str(path))
                    if not same_grid(image_nii, proposal_nii):
                        raise ValueError(
                            f"proposal geometry changed during compact pass: {case} {path}"
                        )
                    yield np.asanyarray(proposal_nii.dataobj) > 0

            records, target_component_map = (
                source_separated_component_records_compact(
                    streamed_source_masks(),
                    support_count,
                    target,
                    native_spacing,
                    float(task["min_mm3"]),
                    float(task["max_mm3"]),
                )
            )
            component_maps = None
        else:
            assert source_masks is not None
            records, component_maps, target_component_map = source_separated_component_records(
                source_masks,
                support_count,
                target,
                native_spacing,
                float(task["min_mm3"]),
                float(task["max_mm3"]),
            )
        component_map = None
    if positive_only:
        records = [record for record in records if bool(record["positive"])]
    else:
        records = deterministic_negative_subset(
            records,
            case,
            int(task["max_negatives"]),
            int(task.get("min_negatives_per_source", 0)),
        )

    rows = []
    for record in records:
        component_id = int(record["component_id"])
        bbox = tuple(record["bbox"])
        if component_policy == "union":
            assert component_map is not None
            component_local = component_map[bbox] == component_id
        else:
            if compact_source_components:
                component_local = unpack_local_component(record)
            else:
                assert component_maps is not None
                component_local = (
                    component_maps[int(record["source_index"])][bbox]
                    == int(record["source_component_id"])
                )
        bbox_start = np.asarray([value.start for value in bbox], dtype=np.int64)
        component_coordinates = np.argwhere(component_local) + bbox_start
        center_world = nib.affines.apply_affine(
            image_nii.affine, np.asarray(record["centroid_voxel"])
        )
        if tile_stride_mm > 0:
            low_world, high_world = component_world_bounds(
                component_coordinates, image_nii.affine, bbox_margin_mm
            )
            tile_centers = tile_centers_world(
                low_world, high_world, shape, crop_spacing, tile_stride_mm
            )
        else:
            tile_centers = [center_world]
        if not bool(record["positive"]) and max_negative_tiles >= 0:
            tile_centers = sorted(
                tile_centers,
                key=lambda value: float(np.linalg.norm(value - center_world)),
            )[:max_negative_tiles]

        if component_policy == "union":
            component_support = np.where(component_local, support_count[bbox], 0)
            prompt_source_count = len(source_masks)
        else:
            # Each candidate is one source's complete component.  A binary
            # prompt preserves the one-source training amplitude and prevents
            # unrelated overlapping sources from changing its morphology.
            component_support = component_local.astype(np.uint8)
            prompt_source_count = 1
        component_affine = slice_affine(image_nii.affine, bbox)
        target_ids = np.asarray(record["target_component_ids"], dtype=np.int32)
        component_target = (
            np.isin(target_component_map, target_ids)
            if target_ids.size
            else None
        )
        for tile_index, tile_center_world in enumerate(tile_centers):
            target_affine = crop_affine(tile_center_world, shape, crop_spacing)
            ct_crop = resample_array(
                ct, image_nii.affine, target_affine, shape, 1, -1024.0
            )
            soft = np.clip(ct_crop, -175.0, 250.0)
            soft = (soft + 175.0) / 425.0
            lung = np.clip(ct_crop, -1000.0, 400.0)
            lung = (lung + 1000.0) / 1400.0
            seed_support, seed_prompt_fallback = resample_seed_prompt(
                component_support,
                component_affine,
                target_affine,
                shape,
                prompt_source_count,
            )
            if component_target is None:
                refine_target = np.zeros(shape, dtype=bool)
                target_resample_fallback = False
            elif tile_stride_mm > 0:
                nearest = resample_array(
                    component_target.astype(np.uint8),
                    image_nii.affine,
                    target_affine,
                    shape,
                    0,
                    0.0,
                ) > 0
                refine_target, target_resample_fallback = in_crop_target(
                    component_target,
                    image_nii.affine,
                    target_affine,
                    shape,
                    nearest,
                )
            else:
                refine_target, target_resample_fallback = resample_binary_target(
                    component_target,
                    image_nii.affine,
                    target_affine,
                    shape,
                )
            patch = np.stack((soft, lung, seed_support), axis=0).astype(np.float16)
            suffix = f"__t{tile_index:03d}" if tile_stride_mm > 0 else ""
            patch_id = f"{case}__c{component_id:04d}{suffix}"
            path = output / "patches" / f"{patch_id}.npz"
            tile_positive = int(refine_target.any())
            if positive_only and not tile_positive:
                # A tile from a target-bearing proposal may fall outside the
                # one annotated/pseudo-positive lesion. Its background remains
                # unknown under a positive-only contract and must not become a
                # classification or segmentation negative.
                continue
            atomic_npz(
                path,
                image=patch,
                target=refine_target[None].astype(np.uint8),
                label=np.asarray(tile_positive, dtype=np.uint8),
                target_affine=target_affine,
                source_affine=np.asarray(image_nii.affine, dtype=np.float64),
            )
            row = {
                "patch_id": patch_id,
                "case": case,
                "cohort": item["cohort"],
                "training_only": int(bool(item.get("training_only", False))),
                "label_contract": (
                    str(item.get("label_contract", "positive_only"))
                    if positive_only
                    else str(item.get("label_contract", "complete"))
                ),
                "component_id": component_id,
                    "tile_index": tile_index,
                    "candidate_positive": int(record["positive"]),
                    "label": tile_positive,
                    "seed_voxels": record["voxels"],
                    "seed_mm3": round(float(record["volume_mm3"]), 6),
                    "intersection_voxels": record["intersection_voxels"],
                    "seed_overlap_fraction": round(float(record["seed_overlap_fraction"]), 8),
                    "distance_to_target_mm": round(float(record["distance_to_target_mm"]), 6),
                    "mean_source_support": round(float(record["mean_source_support"]) / len(source_names), 8),
                    "max_source_support": round(float(record["max_source_support"]) / len(source_names), 8),
                    "seed_prompt_fallback": int(seed_prompt_fallback),
                    "target_resample_fallback": int(target_resample_fallback),
                    "sources": ",".join(source_names),
                "path": str(path.resolve()),
            }
            if component_policy == "source_separate":
                equivalent = list(record["equivalent_components"])
                row.update(
                    {
                        "component_policy": component_policy,
                        "source_index": int(record["source_index"]),
                        "source_component_id": int(record["source_component_id"]),
                        "source_name": source_names[int(record["source_index"])],
                        "equivalent_components": json.dumps(
                            equivalent, sort_keys=True, separators=(",", ":")
                        ),
                    }
                )
            rows.append(row)
    result = {"case": case, "cohort": item["cohort"], "rows": rows}
    if resume:
        write_resumable_case(task, output, result)
    return result


def parse_proposals(values: Iterable[str]) -> tuple[list[str], list[Path]]:
    names, paths = [], []
    for value in values:
        if "=" not in value:
            raise ValueError(f"proposal must be NAME=DIRECTORY: {value}")
        name, directory = value.split("=", 1)
        if not name or name in names:
            raise ValueError(f"invalid or duplicate proposal name: {name!r}")
        path = Path(directory).resolve()
        if not path.is_dir():
            raise NotADirectoryError(path)
        names.append(name)
        paths.append(path)
    if not paths:
        raise ValueError("at least one proposal source is required")
    return names, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proposal", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="64,64,64")
    parser.add_argument("--spacing", default="1.5,1.5,1.5")
    parser.add_argument("--min-mm3", type=float, default=5.0)
    parser.add_argument("--max-mm3", type=float, default=20000.0)
    parser.add_argument("--max-negatives-per-case", type=int, default=64)
    parser.add_argument("--min-negatives-per-source", type=int, default=0)
    parser.add_argument("--bbox-margin-mm", type=float, default=0.0)
    parser.add_argument(
        "--tile-stride-mm",
        type=float,
        default=0.0,
        help="Enable bounding-box tiling when positive; zero preserves one-centroid behavior.",
    )
    parser.add_argument("--max-negative-tiles", type=int, default=1)
    parser.add_argument(
        "--component-policy",
        choices=("union", "source_separate"),
        default="union",
        help="Connect the source union (legacy) or each proposal source independently.",
    )
    parser.add_argument(
        "--compact-source-components",
        action="store_true",
        help=(
            "For source_separate only, retain bbox-local packed component masks "
            "instead of one full-volume uint32 map per source."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse atomic per-case row sidecars whose complete task recipe matches.",
    )
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.min_mm3 < 0
        or args.max_mm3 <= args.min_mm3
        or args.bbox_margin_mm < 0
        or args.tile_stride_mm < 0
        or args.max_negative_tiles < 0
        or args.min_negatives_per_source < 0
        or (args.compact_source_components and args.component_policy != "source_separate")
    ):
        parser.error("invalid worker or component-volume limits")
    shape = parse_triplet(args.shape, int)
    spacing = parse_triplet(args.spacing, float)
    if any(value < 8 for value in shape) or any(value <= 0 for value in spacing):
        parser.error("invalid patch shape or spacing")
    source_names, proposal_dirs = parse_proposals(args.proposal)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("manifest contains no items")
    for item in items:
        complete = item.get("complete_label") is True
        positive_only = item.get("positive_only_label") is True
        if complete == positive_only:
            raise ValueError(
                "every mining case must declare exactly one of complete_label "
                "or positive_only_label"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "item": item,
            "proposal_dirs": [str(path) for path in proposal_dirs],
            "source_names": source_names,
            "output": str(args.output.resolve()),
            "shape": shape,
            "crop_spacing": spacing,
            "min_mm3": args.min_mm3,
            "max_mm3": args.max_mm3,
            "max_negatives": args.max_negatives_per_case,
            "min_negatives_per_source": args.min_negatives_per_source,
            "bbox_margin_mm": args.bbox_margin_mm,
            "tile_stride_mm": args.tile_stride_mm,
            "max_negative_tiles": args.max_negative_tiles,
            "component_policy": args.component_policy,
            "compact_source_components": args.compact_source_components,
            "resume": args.resume,
        }
        for item in items
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(process_case, tasks), 1):
            results.append(result)
            print(
                f"[{index:03d}/{len(tasks):03d}] {result['case']} "
                f"patches={len(result['rows'])}",
                flush=True,
            )
    rows = [row for result in results for row in result["rows"]]
    if not rows or not any(int(row["label"]) for row in rows):
        raise RuntimeError("mining produced no usable positive candidates")
    index_path = args.output / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "version": TILED_PATCH_VERSION if args.tile_stride_mm > 0 else PATCH_VERSION,
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "cases": len(results),
        "patches": len(rows),
        "positive": sum(int(row["label"]) for row in rows),
        "negative": sum(1 - int(row["label"]) for row in rows),
        "shape": list(shape),
        "spacing": list(spacing),
        "component_mm3": [args.min_mm3, args.max_mm3],
        "max_negatives_per_case": args.max_negatives_per_case,
        "min_negatives_per_source": args.min_negatives_per_source,
        "bbox_margin_mm": args.bbox_margin_mm,
        "tile_stride_mm": args.tile_stride_mm,
        "max_negative_tiles": args.max_negative_tiles,
        "component_policy": args.component_policy,
        "compact_source_components": args.compact_source_components,
        "resumable_case_sidecars": args.resume,
        "label_contract_counts": {
            contract: sum(row["label_contract"] == contract for row in rows)
            for contract in sorted({str(row["label_contract"]) for row in rows})
        },
        "training_only_patches": sum(int(row["training_only"]) for row in rows),
        "manifest": file_identity(args.manifest),
        "proposal_sources": [
            {"name": name, "directory": str(path)}
            for name, path in zip(source_names, proposal_dirs)
        ],
        "index": file_identity(index_path),
        "cohort_counts": {
            cohort: sum(row["cohort"] == cohort for row in rows)
            for cohort in sorted({str(row["cohort"]) for row in rows})
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "COMPLETE.txt").write_text(summary["created_at"] + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
