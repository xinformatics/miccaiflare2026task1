#!/usr/bin/env python3
"""Fail-closed audit for label-free candidate-refiner inference patches."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def audit_patch(task: tuple[str, tuple[int, int, int]]) -> dict[str, int]:
    path_text, shape = task
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"image", "patch_affine"}:
            raise ValueError(f"unexpected archive keys: {path} {archive.files}")
        image = archive["image"]
        affine = archive["patch_affine"]
        if image.shape != (3, *shape) or affine.shape != (4, 4):
            raise ValueError(f"shape mismatch: {path} {image.shape} {affine.shape}")
        if not np.isfinite(image).all() or not np.isfinite(affine).all():
            raise ValueError(f"non-finite cache values: {path}")
        if float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise ValueError(f"image outside normalized range: {path}")
        seed_voxels = int(np.count_nonzero(image[2] > 1e-6))
        if seed_voxels == 0:
            raise ValueError(f"empty candidate prompt: {path}")
    return {"seed_voxels": seed_voxels}


def audit_prediction(
    task: tuple[str, tuple[int, int, int]]
    | tuple[str, tuple[int, int, int], bool],
) -> dict[str, float]:
    path_text, shape = task[:2]
    expected_compressed = task[2] if len(task) == 3 else None
    path = Path(path_text)
    if expected_compressed is not None:
        expected = zipfile.ZIP_DEFLATED if expected_compressed else zipfile.ZIP_STORED
        with zipfile.ZipFile(path) as stored:
            members = [item for item in stored.infolist() if not item.is_dir()]
            if not members or any(item.compress_type != expected for item in members):
                raise ValueError(f"prediction cache-mode mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "classification_probability",
            "segmentation_probability",
        }:
            raise ValueError(f"unexpected prediction keys: {path} {archive.files}")
        classification = archive["classification_probability"]
        segmentation = archive["segmentation_probability"]
        if classification.shape != () or segmentation.shape != (1, *shape):
            raise ValueError(
                f"prediction shape mismatch: {path} "
                f"{classification.shape} {segmentation.shape}"
            )
        if not np.isfinite(classification).all() or not np.isfinite(segmentation).all():
            raise ValueError(f"non-finite prediction values: {path}")
        minimum = min(float(classification), float(segmentation.min()))
        maximum = max(float(classification), float(segmentation.max()))
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(f"prediction outside probability range: {path}")
    return {"minimum": minimum, "maximum": maximum}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_inventory(payload: dict[str, object], key: str) -> int:
    raw_paths = payload.get(key)
    raw_hashes = payload.get(f"{key[:-1]}_sha256")
    if not isinstance(raw_paths, list) or not isinstance(raw_hashes, list):
        raise ValueError(f"invalid inference marker checkpoint inventory: {key}")
    if len(raw_paths) != len(raw_hashes):
        raise ValueError(f"checkpoint/hash count mismatch: {key}")
    for path_text, expected in zip(raw_paths, raw_hashes):
        path = Path(str(path_text))
        if not path.is_file() or sha256(path) != str(expected):
            raise ValueError(f"checkpoint identity mismatch: {path}")
    return len(raw_paths)


def validate_inference_plan(
    work: Path, inference: dict[str, object], required: bool
) -> bool:
    plan_text = inference.get("inference_plan")
    plan_hash = str(inference.get("inference_plan_sha256", ""))
    fingerprint = str(inference.get("inference_fingerprint", ""))
    if not plan_text and not plan_hash and not fingerprint:
        if required:
            raise ValueError("inference marker is missing its hash-bound plan")
        return False
    plan = Path(str(plan_text))
    expected = (work / "INFERENCE_PLAN.json").resolve()
    if (
        plan.resolve() != expected
        or not plan.is_file()
        or len(plan_hash) != 64
        or sha256(plan) != plan_hash
        or len(fingerprint) != 64
    ):
        raise ValueError("inference-plan identity mismatch")
    payload = json.loads(plan.read_text(encoding="utf-8"))
    if (
        payload.get("inference_fingerprint") != fingerprint
        or not isinstance(payload.get("identity"), dict)
        or payload["identity"].get("prediction_cache_compressed")
        is not inference.get("prediction_cache_compressed")
        or payload["identity"].get("prediction_cache_fsync")
        is not inference.get("prediction_cache_fsync")
    ):
        raise ValueError("inference plan and completion marker disagree")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--allow-external-patches",
        action="store_true",
        help="Allow manifest patch paths to reference an immutable cache outside work.",
    )
    parser.add_argument(
        "--require-predictions",
        action="store_true",
        help="Require and audit exact inference-marker/prediction coverage.",
    )
    parser.add_argument(
        "--require-inference-plan",
        action="store_true",
        help="Require the completion marker to hash-bind the cache-generation plan.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    manifest_path = args.work / "manifest.json"
    marker_path = args.work / "PREPARE_COMPLETE.txt"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError("inference cache is not marked complete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shape = tuple(int(value) for value in manifest.get("shape", ()))
    spacing = tuple(float(value) for value in manifest.get("spacing", ()))
    if shape != (64, 64, 64) or spacing != (1.5, 1.5, 1.5):
        raise ValueError(f"unexpected inference geometry: {shape} {spacing}")
    case_records = manifest.get("case_records")
    records = manifest.get("records")
    if (
        not isinstance(case_records, list)
        or not isinstance(records, list)
        or not case_records
        or not records
    ):
        raise ValueError("invalid inference manifest records")
    cases = [str(record["case"]) for record in case_records]
    patch_ids = [str(record["patch_id"]) for record in records]
    if (
        len(cases) != int(manifest.get("cases", -1))
        or len(cases) != len(set(cases))
        or len(records) != int(manifest.get("candidates", -1))
        or len(patch_ids) != len(set(patch_ids))
    ):
        raise ValueError("manifest count or uniqueness mismatch")
    case_set = set(cases)
    if any(str(record["case"]) not in case_set for record in records):
        raise ValueError("candidate references an unknown case")
    component_policy = str(manifest.get("component_policy", "union"))
    if component_policy not in ("union", "source_separate"):
        raise ValueError(f"invalid component policy: {component_policy}")
    if component_policy == "source_separate":
        source_names = list(manifest.get("source_names", []))
        if len(source_names) < 2:
            raise ValueError("source-separated cache requires multiple named sources")
        for record in records:
            source_index = int(record.get("source_index", -1))
            if (
                record.get("component_policy") != "source_separate"
                or not 0 <= source_index < len(source_names)
                or int(record.get("source_component_id", 0)) < 1
                or record.get("source_name") != source_names[source_index]
                or not isinstance(record.get("equivalent_components"), list)
            ):
                raise ValueError(
                    f"invalid source-separated component provenance: "
                    f"{record.get('patch_id')}"
                )
    indexed_paths = {Path(str(record["patch"])).resolve() for record in records}
    patch_directory = args.work / "patches"
    if patch_directory.is_dir():
        disk_paths = {path.resolve() for path in patch_directory.glob("*.npz")}
        if indexed_paths != disk_paths:
            raise ValueError(
                f"patch coverage mismatch missing={len(indexed_paths-disk_paths)} "
                f"extra={len(disk_paths-indexed_paths)}"
            )
    elif not args.allow_external_patches:
        raise FileNotFoundError(f"local patch directory is missing: {patch_directory}")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(audit_patch, ((str(path), shape) for path in indexed_paths)))

    prediction_results: list[dict[str, float]] = []
    if args.require_predictions:
        inference_marker = args.work / "INFERENCE_COMPLETE.json"
        if not inference_marker.is_file():
            raise FileNotFoundError("inference output is not marked complete")
        inference = json.loads(inference_marker.read_text(encoding="utf-8"))
        inference_plan_bound = validate_inference_plan(
            args.work, inference, args.require_inference_plan
        )
        if int(inference.get("candidates", -1)) != len(records):
            raise ValueError("inference marker candidate count mismatch")
        if validate_checkpoint_inventory(inference, "checkpoints") < 1:
            raise ValueError("inference marker has no classifier checkpoint")
        contour_count = validate_checkpoint_inventory(inference, "contour_checkpoints")
        contour_modes = inference.get("contour_input_modes")
        if not isinstance(contour_modes, list) or len(contour_modes) != contour_count:
            raise ValueError("contour checkpoint/input-mode count mismatch")
        expected_predictions = {
            (args.work / "predictions" / f"{patch_id}.npz").resolve()
            for patch_id in patch_ids
        }
        disk_predictions = {
            path.resolve() for path in (args.work / "predictions").glob("*.npz")
        }
        if expected_predictions != disk_predictions:
            raise ValueError(
                "prediction coverage mismatch "
                f"missing={len(expected_predictions-disk_predictions)} "
                f"extra={len(disk_predictions-expected_predictions)}"
            )
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            cache_compressed = inference.get("prediction_cache_compressed")
            if not isinstance(cache_compressed, bool):
                cache_compressed = None
            prediction_results = list(
                executor.map(
                    audit_prediction,
                    (
                        (str(path), shape, cache_compressed)
                        if cache_compressed is not None
                        else (str(path), shape)
                        for path in expected_predictions
                    ),
                )
            )
    else:
        inference_plan_bound = False
    report = {
        "work": str(args.work.resolve()),
        "cases": len(cases),
        "candidates": len(records),
        "shape": list(shape),
        "spacing": list(spacing),
        "only_disconnected": bool(manifest.get("only_disconnected")),
        "only_connected": bool(manifest.get("only_connected")),
        "component_policy": component_policy,
        "seed_prompt_fallbacks": sum(
            int(record.get("seed_prompt_fallback", 0)) for record in records
        ),
        "minimum_seed_voxels": min(result["seed_voxels"] for result in results),
        "exact_file_coverage": True,
        "all_prompts_nonempty": True,
        "all_values_finite_and_normalized": True,
        "external_patch_cache": not patch_directory.is_dir(),
        "predictions_required": args.require_predictions,
        "prediction_exact_file_coverage": bool(prediction_results),
        "prediction_values_finite_and_normalized": bool(prediction_results),
        "inference_plan_bound": inference_plan_bound,
        "prediction_cache_fsync": (
            inference.get("prediction_cache_fsync")
            if args.require_predictions
            else None
        ),
        "prediction_minimum": (
            min(result["minimum"] for result in prediction_results)
            if prediction_results
            else None
        ),
        "prediction_maximum": (
            max(result["maximum"] for result in prediction_results)
            if prediction_results
            else None
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
