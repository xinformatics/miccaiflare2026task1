#!/usr/bin/env python3
"""Resumable inference and deployment for the candidate-specific refiner.

This program never reads a ground-truth mask.  ``prepare`` extracts one
prompt-conditioned patch per disconnected proposal component, ``infer``
caches ensemble probabilities, and ``materialize`` can sweep conservative
classification/refinement gates without repeating GPU work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from candidate_bbox_tiles import component_world_bounds, tile_centers_world
from candidate_refiner_model import CandidateRefiner
from candidate_refiner_source_conditioning import append_source_condition_batch
from focus_classifier_fusion import (
    DEPLOYMENT_FEATURE_NAMES,
    apply_fusion as apply_focus_classifier_fusion,
    load_qualified_fusion,
)
from proposal_component_instances import instance_mask, source_separated_instances
from prepare_candidate_refiner_patches import (
    atomic_npz,
    crop_affine,
    pad_slices,
    parse_proposals,
    parse_triplet,
    resample_array,
    resample_seed_prompt,
    same_grid,
    slice_affine,
    strip_nii,
)


def image_for_case(images: Path, case: str) -> Path:
    choices = (images / f"{case}_0000.nii.gz", images / f"{case}.nii.gz")
    found = [path for path in choices if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one CT for {case}: {found}")
    return found[0]


def proposal_cases(directory: Path) -> set[str]:
    return {strip_nii(path.name) for path in directory.glob("*.nii.gz")}


def load_source_masks(
    proposal_dirs: list[Path], case: str, reference: nib.Nifti1Image
) -> tuple[list[np.ndarray], list[str]]:
    arrays, paths = [], []
    for directory in proposal_dirs:
        path = directory / f"{case}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        image = nib.load(str(path))
        if not same_grid(reference, image):
            raise ValueError(f"proposal geometry mismatch: {case} {directory}")
        arrays.append(np.asanyarray(image.dataobj) > 0)
        paths.append(str(path.resolve()))
    return arrays, paths


def load_sources(
    proposal_dirs: list[Path], case: str, reference: nib.Nifti1Image
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    arrays, paths = load_source_masks(proposal_dirs, case, reference)
    support = np.sum(arrays, axis=0, dtype=np.uint8)
    return support > 0, support, paths


def prepare(args: argparse.Namespace) -> None:
    source_names, proposal_dirs = parse_proposals(args.proposal)
    case_sets = [proposal_cases(path) for path in proposal_dirs]
    cases = sorted(set.intersection(*case_sets))
    if any(values != set(cases) for values in case_sets):
        raise ValueError("proposal sources do not contain identical case sets")
    case_limit = getattr(args, "case_limit", None)
    if case_limit:
        cases = cases[:case_limit]
    if not cases:
        raise RuntimeError("no proposal cases")
    shape = parse_triplet(args.shape, int)
    spacing = parse_triplet(args.spacing, float)
    work = args.work.resolve()
    (work / "patches").mkdir(parents=True, exist_ok=True)
    records = []
    case_records = []
    reused_patches = 0
    written_patches = 0
    component_policy = str(getattr(args, "component_policy", "union"))
    if component_policy not in ("union", "source_separate"):
        raise ValueError(f"invalid component policy: {component_policy}")
    for case_index, case in enumerate(cases, 1):
        image_path = image_for_case(args.images, case)
        image_nii = nib.load(str(image_path))
        ct = np.asanyarray(image_nii.dataobj).astype(np.float32)
        source_masks, source_paths = load_source_masks(
            proposal_dirs, case, image_nii
        )
        support = np.sum(source_masks, axis=0, dtype=np.uint8)
        union = support > 0
        base = None
        if args.base:
            base_path = args.base / f"{case}.nii.gz"
            if not base_path.is_file():
                raise FileNotFoundError(base_path)
            base_nii = nib.load(str(base_path))
            if not same_grid(image_nii, base_nii):
                raise ValueError(f"base geometry mismatch: {case}")
            base = np.asanyarray(base_nii.dataobj) > 0
        if component_policy == "union":
            component_map, count = cc3d.connected_components(
                union.astype(np.uint8), connectivity=26, return_N=True
            )
            statistics = cc3d.statistics(component_map)
            component_maps = None
            instances = [
                {
                    "component_id": component_id,
                    "source_index": -1,
                    "source_component_id": component_id,
                    "bbox": statistics["bounding_boxes"][component_id],
                    "centroid_voxel": statistics["centroids"][component_id],
                    "voxels": int(statistics["voxel_counts"][component_id]),
                }
                for component_id in range(1, count + 1)
            ]
        else:
            component_maps, instances = source_separated_instances(source_masks)
            component_map = None
        native_spacing = np.linalg.norm(image_nii.affine[:3, :3], axis=0)
        voxel_mm3 = float(np.prod(native_spacing))
        case_count = 0
        for instance in instances:
            component_id = int(instance["component_id"])
            voxels = int(instance["voxels"])
            volume = voxels * voxel_mm3
            if not args.min_mm3 <= volume <= args.max_mm3:
                continue
            bbox = pad_slices(
                tuple(instance["bbox"]), union.shape
            )
            if component_policy == "union":
                assert component_map is not None
                component_local = component_map[bbox] == component_id
            else:
                assert component_maps is not None
                component_local = (
                    component_maps[int(instance["source_index"])][bbox]
                    == int(instance["source_component_id"])
                )
            overlaps_base = base is not None and np.any(component_local & base[bbox])
            if args.only_disconnected and overlaps_base:
                continue
            if args.only_connected and not overlaps_base:
                continue
            bbox_start = np.asarray([value.start for value in bbox], dtype=np.int64)
            coordinates = np.argwhere(component_local) + bbox_start
            center_world = nib.affines.apply_affine(
                image_nii.affine, instance["centroid_voxel"]
            )
            if component_policy == "union":
                component_support = np.where(
                    component_local, support[bbox], 0
                ).astype(np.float32)
                prompt_source_count = len(proposal_dirs)
            else:
                component_support = component_local.astype(np.float32)
                prompt_source_count = 1
            component_affine = slice_affine(image_nii.affine, bbox)
            if args.tile_stride_mm > 0:
                low_world, high_world = component_world_bounds(
                    coordinates, image_nii.affine, args.bbox_margin_mm
                )
                tile_centers = tile_centers_world(
                    low_world, high_world, shape, spacing, args.tile_stride_mm
                )
            else:
                tile_centers = [center_world]
            values = support[bbox][component_local]
            for tile_index, tile_center_world in enumerate(tile_centers):
                patch_affine = crop_affine(tile_center_world, shape, spacing)
                seed, seed_prompt_fallback = resample_seed_prompt(
                    component_support,
                    component_affine,
                    patch_affine,
                    shape,
                    prompt_source_count,
                )
                suffix = f"__t{tile_index:03d}" if args.tile_stride_mm > 0 else ""
                patch_id = f"{case}__c{component_id:04d}{suffix}"
                patch_path = work / "patches" / f"{patch_id}.npz"
                reused = False
                if bool(getattr(args, "resume", False)) and patch_path.is_file():
                    try:
                        with np.load(patch_path, allow_pickle=False) as cached:
                            cached_image = cached["image"]
                            cached_affine = cached["patch_affine"]
                            reused = (
                                cached_image.shape == (3, *shape)
                                and cached_affine.shape == (4, 4)
                                and np.issubdtype(cached_image.dtype, np.floating)
                                and np.all(np.isfinite(cached_image))
                                and np.all(np.isfinite(cached_affine))
                                and np.allclose(
                                    cached_affine, patch_affine, rtol=0, atol=1e-8
                                )
                                and np.allclose(
                                    cached_image[2].astype(np.float32),
                                    seed,
                                    rtol=0,
                                    atol=5e-4,
                                )
                            )
                    except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile):
                        reused = False
                if reused:
                    reused_patches += 1
                else:
                    ct_crop = resample_array(
                        ct, image_nii.affine, patch_affine, shape, 1, -1024.0
                    )
                    soft = (np.clip(ct_crop, -175.0, 250.0) + 175.0) / 425.0
                    lung = (np.clip(ct_crop, -1000.0, 400.0) + 1000.0) / 1400.0
                    atomic_npz(
                        patch_path,
                        image=np.stack((soft, lung, seed), axis=0).astype(np.float16),
                        patch_affine=patch_affine.astype(np.float64),
                    )
                    written_patches += 1
                record = {
                        "patch_id": patch_id,
                        "case": case,
                        "component_id": component_id,
                        "tile_index": tile_index,
                        "seed_voxels": voxels,
                        "seed_mm3": volume,
                        "mean_source_support": float(values.mean()) / len(proposal_dirs),
                        "max_source_support": float(values.max()) / len(proposal_dirs),
                        "seed_prompt_fallback": int(seed_prompt_fallback),
                        "image": str(image_path.resolve()),
                        "source_proposals": source_paths,
                        "patch": str(patch_path.resolve()),
                    }
                if component_policy == "source_separate":
                    record.update(
                        {
                            "component_policy": component_policy,
                            "source_index": int(instance["source_index"]),
                            "source_component_id": int(
                                instance["source_component_id"]
                            ),
                            "source_name": source_names[int(instance["source_index"])],
                            "equivalent_components": instance[
                                "equivalent_components"
                            ],
                        }
                    )
                records.append(record)
                case_count += 1
        case_records.append(
            {
                "case": case,
                "image": str(image_path.resolve()),
                "source_proposals": source_paths,
            }
        )
        print(f"[{case_index:03d}/{len(cases):03d}] {case}: {case_count} candidates", flush=True)
    manifest = {
        "cases": len(cases),
        "candidates": len(records),
        "shape": list(shape),
        "spacing": list(spacing),
        "source_names": source_names,
        "proposal_dirs": [str(path) for path in proposal_dirs],
        "component_policy": component_policy,
        "base": str(args.base.resolve()) if args.base else None,
        "only_disconnected": bool(args.only_disconnected),
        "only_connected": bool(args.only_connected),
        "bbox_margin_mm": args.bbox_margin_mm,
        "tile_stride_mm": args.tile_stride_mm,
        "resume": bool(getattr(args, "resume", False)),
        "reused_patches": reused_patches,
        "written_patches": written_patches,
        "case_records": case_records,
        "records": records,
    }
    (work / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (work / "PREPARE_COMPLETE.txt").write_text(
        __import__("datetime").datetime.now().astimezone().isoformat() + "\n"
    )


class InferenceDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with np.load(record["patch"], allow_pickle=False) as archive:
            image = torch.from_numpy(archive["image"].astype(np.float32))
        return image, record["patch_id"], str(record.get("source_name", ""))


def load_model(path: Path, device: torch.device) -> CandidateRefiner:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    base_channels = int(config.get("base_channels", 16))
    source_condition_groups = tuple(config.get("source_condition_groups", []))
    prompt_channel = int(config.get("prompt_channel", 2))
    if prompt_channel != 2:
        raise ValueError(
            "candidate-refiner checkpoint prompt channel does not match "
            "the fixed (soft, lung, seed_support) input layout"
        )
    model = CandidateRefiner(
        in_channels=3 + len(source_condition_groups),
        base_channels=base_channels,
        prompt_masked_classifier=bool(config.get("prompt_masked_classifier", False)),
        prompt_channel=prompt_channel,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.candidate_refiner_source_condition_groups = source_condition_groups
    model.candidate_refiner_prompt_masked_classifier = bool(
        config.get("prompt_masked_classifier", False)
    )
    model.candidate_refiner_prompt_channel = prompt_channel
    return model.eval().to(device)


def load_focus_contour_model(path: Path, device: torch.device) -> torch.nn.Module:
    """Load a contour-only GF Focus checkpoint without changing legacy inference."""

    # Keep MONAI and the substantially larger SwinUNETR dependency lazy: every
    # established candidate-refiner controller follows the legacy path below.
    from focus_candidate_refiner import build_focus_model, focus_state_dict

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    input_mode = str(checkpoint.get("config", {}).get("input_mode", "soft"))
    if input_mode not in {"soft", "soft_seed", "soft_lung_seed"}:
        raise ValueError(f"unsupported Focus contour input mode: {input_mode}")
    input_channels = {
        "soft": 1,
        "soft_seed": 2,
        "soft_lung_seed": 3,
    }[input_mode]
    model = build_focus_model(use_checkpoint=False, in_channels=input_channels)
    model.load_state_dict(focus_state_dict(checkpoint), strict=True)
    model.candidate_refiner_input_mode = input_mode
    return model.eval().to(device)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_named_checkpoint_routes(values: list[str]) -> dict[str, list[Path]]:
    """Parse repeatable NAME=CHECKPOINT routes, allowing route ensembles."""

    routes: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError(f"candidate contour route must use NAME=CHECKPOINT: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"invalid candidate contour route: {value}")
        routes[name].append(Path(raw_path))
    return dict(routes)


def parse_source_contour_routes(values: list[str]) -> dict[str, str]:
    """Parse a complete one-to-one SOURCE=ROUTE deployment declaration."""

    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"source contour route must use SOURCE=ROUTE: {value}")
        source, route = (part.strip() for part in value.split("=", 1))
        if not source or not route or source in result:
            raise ValueError(f"invalid or duplicate source contour route: {value}")
        result[source] = route
    return result


def ensemble_refiner_batch(
    classifier_models: list[torch.nn.Module],
    images: torch.Tensor,
    contour_models: list[torch.nn.Module] | None = None,
    source_names: list[str] | tuple[str, ...] | None = None,
    candidate_contour_models: list[torch.nn.Module] | None = None,
    candidate_contour_models_by_route: dict[
        str, list[torch.nn.Module]
    ] | None = None,
    focus_contour_models_by_route: dict[str, list[torch.nn.Module]] | None = None,
    source_contour_routes: dict[str, str] | None = None,
    classifier_models_by_route: dict[str, list[torch.nn.Module]] | None = None,
    source_classifier_routes: dict[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensemble candidate confidence and, optionally, pretrained Focus contours."""

    if not classifier_models:
        raise ValueError("at least one classifier/refiner model is required")
    routed_candidate = bool(candidate_contour_models_by_route)
    routed_focus = bool(focus_contour_models_by_route)
    routed = routed_candidate or routed_focus or bool(source_contour_routes)
    candidate_route_names = set(candidate_contour_models_by_route or {})
    focus_route_names = set(focus_contour_models_by_route or {})
    overlapping_route_names = sorted(candidate_route_names & focus_route_names)
    if overlapping_route_names:
        raise ValueError(
            "candidate-refiner and Focus contour route names overlap: "
            f"{overlapping_route_names}"
        )
    if sum(
        bool(value)
        for value in (
            contour_models,
            candidate_contour_models,
            routed,
        )
    ) > 1:
        raise ValueError(
            "Focus, global candidate-refiner, and routed contour overrides are mutually exclusive"
        )
    if (routed_candidate or routed_focus) != bool(source_contour_routes):
        raise ValueError("routed contour models and source routes must be supplied together")
    routed_classifier = bool(classifier_models_by_route)
    if routed_classifier != bool(source_classifier_routes):
        raise ValueError(
            "routed classifier models and source routes must be supplied together"
        )

    def conditioned_input(
        model: torch.nn.Module,
        batch_images: torch.Tensor = images,
        batch_sources: list[str] | tuple[str, ...] | None = source_names,
    ) -> torch.Tensor:
        source_groups = tuple(
            getattr(model, "candidate_refiner_source_condition_groups", ())
        )
        if not source_groups:
            return batch_images
        if batch_sources is None:
            raise ValueError(
                "source names are required by a source-conditioned checkpoint"
            )
        return append_source_condition_batch(
            batch_images, batch_sources, source_groups
        )

    class_probabilities = []
    legacy_contours = []
    for model in classifier_models:
        class_logits, segmentation_logits = model(conditioned_input(model))
        class_probabilities.append(torch.sigmoid(class_logits).float())
        legacy_contours.append(torch.sigmoid(segmentation_logits).float())
    if classifier_models_by_route:
        assert source_classifier_routes is not None
        if source_names is None or len(source_names) != len(images):
            raise ValueError("source names must align with routed classifier batches")
        unknown_sources = sorted(set(source_names) - set(source_classifier_routes))
        unknown_routes = sorted(
            set(source_classifier_routes.values()) - set(classifier_models_by_route)
        )
        unused_routes = sorted(
            set(classifier_models_by_route) - set(source_classifier_routes.values())
        )
        if unknown_sources or unknown_routes or unused_routes:
            raise ValueError(
                "invalid routed classifier coverage: "
                f"sources={unknown_sources} routes={unknown_routes} unused={unused_routes}"
            )
        classification_mean = torch.empty_like(class_probabilities[0])
        assigned_classification = torch.zeros(
            len(images), dtype=torch.bool, device=images.device
        )
        for route, route_models in classifier_models_by_route.items():
            if not route_models:
                raise ValueError(f"empty routed classifier ensemble: {route}")
            row_indices = [
                index
                for index, source in enumerate(source_names)
                if source_classifier_routes[source] == route
            ]
            if not row_indices:
                continue
            indices = torch.as_tensor(
                row_indices, dtype=torch.long, device=images.device
            )
            route_images = images.index_select(0, indices)
            route_sources = [source_names[index] for index in row_indices]
            probabilities = []
            for model in route_models:
                logits, _ = model(conditioned_input(model, route_images, route_sources))
                probabilities.append(torch.sigmoid(logits).float())
            classification_mean.index_copy_(
                0, indices, torch.stack(probabilities).mean(0)
            )
            assigned_classification[indices] = True
        if not bool(torch.all(assigned_classification)):
            raise ValueError("routed classifier inference left unassigned batch rows")
    else:
        classification_mean = torch.stack(class_probabilities).mean(0)
    def focus_probability(
        model: torch.nn.Module, batch_images: torch.Tensor
    ) -> torch.Tensor:
        input_mode = getattr(model, "candidate_refiner_input_mode", "soft")
        if input_mode == "soft":
            contour_input = batch_images[:, 0:1]
        elif input_mode == "soft_seed":
            contour_input = batch_images[:, (0, 2)]
        elif input_mode == "soft_lung_seed":
            contour_input = batch_images[:, (0, 1, 2)]
        else:
            raise ValueError(f"unsupported Focus contour input mode: {input_mode}")
        logits = model(contour_input)
        if logits.ndim != 5 or logits.shape[1] != 2:
            raise ValueError("Focus logits must have shape (B,2,D,H,W)")
        return torch.sigmoid(logits[:, 1:2] - logits[:, 0:1]).float()

    routed_models_by_route = {
        **(candidate_contour_models_by_route or {}),
        **(focus_contour_models_by_route or {}),
    }
    if routed_models_by_route:
        assert source_contour_routes is not None
        if source_names is None or len(source_names) != len(images):
            raise ValueError("source names must align with routed contour batches")
        unknown_sources = sorted(set(source_names) - set(source_contour_routes))
        unknown_routes = sorted(
            set(source_contour_routes.values())
            - set(routed_models_by_route)
        )
        unused_routes = sorted(
            set(routed_models_by_route)
            - set(source_contour_routes.values())
        )
        if unknown_sources or unknown_routes or unused_routes:
            raise ValueError(
                "invalid routed contour coverage: "
                f"sources={unknown_sources} routes={unknown_routes} "
                f"unused={unused_routes}"
            )
        contour_mean = torch.empty_like(legacy_contours[0])
        assigned = torch.zeros(len(images), dtype=torch.bool, device=images.device)
        for route, route_models in routed_models_by_route.items():
            if not route_models:
                raise ValueError(f"empty routed contour ensemble: {route}")
            row_indices = [
                index
                for index, source in enumerate(source_names)
                if source_contour_routes[source] == route
            ]
            if not row_indices:
                continue
            indices = torch.as_tensor(row_indices, dtype=torch.long, device=images.device)
            route_images = images.index_select(0, indices)
            route_sources = [source_names[index] for index in row_indices]
            route_probabilities = []
            for model in route_models:
                if route in focus_route_names:
                    route_probabilities.append(
                        focus_probability(model, route_images)
                    )
                else:
                    _, segmentation_logits = model(
                        conditioned_input(model, route_images, route_sources)
                    )
                    route_probabilities.append(
                        torch.sigmoid(segmentation_logits).float()
                    )
            contour_mean.index_copy_(
                0, indices, torch.stack(route_probabilities).mean(0)
            )
            assigned[indices] = True
        if not bool(torch.all(assigned)):
            raise ValueError("routed contour inference left unassigned batch rows")
        contour_probabilities = None
    elif candidate_contour_models:
        contour_probabilities = []
        for model in candidate_contour_models:
            _, segmentation_logits = model(conditioned_input(model))
            contour_probabilities.append(torch.sigmoid(segmentation_logits).float())
    elif contour_models:
        contour_probabilities = [focus_probability(model, images) for model in contour_models]
    else:
        contour_probabilities = legacy_contours
    if routed_models_by_route:
        segmentation_mean = contour_mean
    else:
        assert contour_probabilities is not None
        segmentation_mean = torch.stack(contour_probabilities).mean(0)
    return classification_mean, segmentation_mean


@torch.no_grad()
def infer(args: argparse.Namespace) -> None:
    manifest = json.loads((args.work / "manifest.json").read_text())
    records = manifest["records"]
    if not records:
        raise RuntimeError("empty inference manifest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_model_cache: dict[Path, torch.nn.Module] = {}
    focus_model_cache: dict[Path, torch.nn.Module] = {}

    def cached_candidate_model(path: Path) -> torch.nn.Module:
        identity = path.resolve()
        if identity not in candidate_model_cache:
            candidate_model_cache[identity] = load_model(path, device)
        return candidate_model_cache[identity]

    def cached_focus_model(path: Path) -> torch.nn.Module:
        identity = path.resolve()
        if identity not in focus_model_cache:
            focus_model_cache[identity] = load_focus_contour_model(path, device)
        return focus_model_cache[identity]

    models = [cached_candidate_model(path) for path in args.checkpoint]
    contour_paths = list(getattr(args, "contour_checkpoint", None) or [])
    contour_models = [cached_focus_model(path) for path in contour_paths]
    candidate_contour_paths = list(
        getattr(args, "candidate_contour_checkpoint", None) or []
    )
    candidate_contour_models = [
        cached_candidate_model(path) for path in candidate_contour_paths
    ]
    routed_path_values = list(
        getattr(args, "candidate_contour_route", None) or []
    )
    routed_paths = parse_named_checkpoint_routes(routed_path_values)
    focus_routed_path_values = list(
        getattr(args, "focus_contour_route", None) or []
    )
    focus_routed_paths = parse_named_checkpoint_routes(focus_routed_path_values)
    source_contour_routes = parse_source_contour_routes(
        list(getattr(args, "source_contour_route", None) or [])
    )
    classifier_routed_path_values = list(
        getattr(args, "classifier_route", None) or []
    )
    classifier_routed_paths = parse_named_checkpoint_routes(
        classifier_routed_path_values
    )
    source_classifier_routes = parse_source_contour_routes(
        list(getattr(args, "source_classifier_route", None) or [])
    )
    routed_models = {
        name: [cached_candidate_model(path) for path in paths]
        for name, paths in routed_paths.items()
    }
    focus_routed_models = {
        name: [cached_focus_model(path) for path in paths]
        for name, paths in focus_routed_paths.items()
    }
    overlapping_route_names = sorted(set(routed_models) & set(focus_routed_models))
    if overlapping_route_names:
        raise ValueError(
            "candidate-refiner and Focus contour route names overlap: "
            f"{overlapping_route_names}"
        )
    classifier_routed_models = {
        name: [cached_candidate_model(path) for path in paths]
        for name, paths in classifier_routed_paths.items()
    }
    fusion_path = getattr(args, "classifier_fusion", None)
    fusion_deployment = load_qualified_fusion(fusion_path) if fusion_path else None
    if fusion_deployment is not None:
        if (
            not contour_models
            or candidate_contour_models
            or routed_models
            or focus_routed_models
            or classifier_routed_models
        ):
            raise ValueError(
                "Focus classifier fusion requires one global Focus contour family"
            )
        observed_modes = {
            str(getattr(model, "candidate_refiner_input_mode", "soft"))
            for model in contour_models
        }
        expected_mode = str(fusion_deployment["focus_input_mode"])
        if observed_modes != {expected_mode}:
            raise ValueError(
                "Focus classifier fusion input mode does not match contour checkpoints"
            )
    compress_predictions = not bool(
        getattr(args, "uncompressed_predictions", False)
    )
    durable_predictions = compress_predictions

    def checkpoint_identity(paths: list[Path]) -> list[dict[str, str]]:
        return [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in paths
        ]

    implementation_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("candidate_refiner_model.py").resolve(),
        Path(__file__).with_name(
            "candidate_refiner_source_conditioning.py"
        ).resolve(),
        Path(__file__).with_name("focus_candidate_refiner.py").resolve(),
        Path(__file__).with_name("focus_classifier_fusion.py").resolve(),
    ]
    inference_identity = {
        "schema_version": 1,
        "implementation": checkpoint_identity(implementation_paths),
        "runtime": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "checkpoints": checkpoint_identity(list(args.checkpoint)),
        "contour_checkpoints": checkpoint_identity(contour_paths),
        "candidate_contour_checkpoints": checkpoint_identity(
            candidate_contour_paths
        ),
        "routed_candidate_contours": {
            name: checkpoint_identity(paths)
            for name, paths in sorted(routed_paths.items())
        },
        "routed_focus_contours": {
            name: checkpoint_identity(paths)
            for name, paths in sorted(focus_routed_paths.items())
        },
        "routed_classifiers": {
            name: checkpoint_identity(paths)
            for name, paths in sorted(classifier_routed_paths.items())
        },
        "source_contour_routes": dict(sorted(source_contour_routes.items())),
        "source_classifier_routes": dict(sorted(source_classifier_routes.items())),
        "classifier_fusion": (
            {
                "path": str(Path(fusion_path).resolve()),
                "sha256": file_sha256(Path(fusion_path)),
            }
            if fusion_path
            else None
        ),
        "prediction_cache_compressed": compress_predictions,
        "prediction_cache_fsync": durable_predictions,
    }
    inference_fingerprint = hashlib.sha256(
        json.dumps(
            inference_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    plan_path = args.work / "INFERENCE_PLAN.json"
    prior_fingerprint = ""
    if plan_path.is_file():
        try:
            prior_fingerprint = str(
                json.loads(plan_path.read_text(encoding="utf-8")).get(
                    "inference_fingerprint", ""
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            prior_fingerprint = ""
    same_plan = prior_fingerprint == inference_fingerprint
    output = args.work / "predictions"
    if output.is_dir() and not same_plan:
        suffix = prior_fingerprint[:12] or "unbound"
        stale = args.work / f"predictions.stale.{suffix}"
        index = 1
        while stale.exists():
            stale = args.work / f"predictions.stale.{suffix}.{index}"
            index += 1
        output.rename(stale)
    output.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "schema_version": 1,
        "inference_fingerprint": inference_fingerprint,
        "identity": inference_identity,
    }
    plan_temporary = plan_path.with_name(f".{plan_path.name}.{os.getpid()}.tmp")
    plan_temporary.write_text(
        json.dumps(plan_payload, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(plan_temporary, plan_path)

    def reusable_prediction(record: dict) -> bool:
        path = output / f"{record['patch_id']}.npz"
        if not path.is_file():
            return False
        expected_compression = (
            zipfile.ZIP_DEFLATED if compress_predictions else zipfile.ZIP_STORED
        )
        try:
            with zipfile.ZipFile(path) as stored:
                members = [item for item in stored.infolist() if not item.is_dir()]
                if not members or any(
                    item.compress_type != expected_compression for item in members
                ):
                    return False
            with np.load(path, allow_pickle=False) as prediction, np.load(
                record["patch"], allow_pickle=False
            ) as patch:
                if set(prediction.files) != {
                    "classification_probability",
                    "segmentation_probability",
                }:
                    return False
                classification = prediction["classification_probability"]
                segmentation = prediction["segmentation_probability"]
                image = patch["image"]
                if (
                    classification.shape != ()
                    or image.ndim != 4
                    or segmentation.shape != (1, *image.shape[1:])
                    or not np.isfinite(classification).all()
                    or not np.isfinite(segmentation).all()
                ):
                    return False
                minimum = min(float(classification), float(segmentation.min()))
                maximum = max(float(classification), float(segmentation.max()))
                return minimum >= 0.0 and maximum <= 1.0
        except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile):
            return False

    records_by_patch_id = {str(record["patch_id"]): record for record in records}
    if len(records_by_patch_id) != len(records):
        raise ValueError("duplicate inference patch IDs")
    if routed_models or focus_routed_models:
        selected_routes = {**routed_models, **focus_routed_models}
        manifest_sources = {
            str(record.get("source_name", "")).strip() for record in records
        }
        if "" in manifest_sources or set(source_contour_routes) != manifest_sources:
            missing = sorted(manifest_sources - set(source_contour_routes))
            extra = sorted(set(source_contour_routes) - manifest_sources)
            raise ValueError(
                f"source contour routes do not cover manifest: missing={missing} extra={extra}"
            )
        if set(source_contour_routes.values()) != set(selected_routes):
            raise ValueError("source contour routes and loaded route names differ")
    if classifier_routed_models:
        manifest_sources = {
            str(record.get("source_name", "")).strip() for record in records
        }
        if "" in manifest_sources or set(source_classifier_routes) != manifest_sources:
            missing = sorted(manifest_sources - set(source_classifier_routes))
            extra = sorted(set(source_classifier_routes) - manifest_sources)
            raise ValueError(
                "source classifier routes do not cover manifest: "
                f"missing={missing} extra={extra}"
            )
        if set(source_classifier_routes.values()) != set(classifier_routed_models):
            raise ValueError("source classifier routes and loaded route names differ")
    pending_records = list(records)
    reused_predictions = 0
    if bool(getattr(args, "resume", False)) and same_plan:
        pending_records = []
        for record in records:
            if reusable_prediction(record):
                reused_predictions += 1
            else:
                pending_records.append(record)
    loader = (
        DataLoader(
            InferenceDataset(pending_records),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        if pending_records
        else ()
    )
    completed = reused_predictions
    written_predictions = 0
    for images, patch_ids, source_names in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            class_mean, segmentation_mean = ensemble_refiner_batch(
                models,
                images,
                contour_models=contour_models,
                source_names=list(source_names),
                candidate_contour_models=candidate_contour_models,
                candidate_contour_models_by_route=routed_models,
                focus_contour_models_by_route=focus_routed_models,
                source_contour_routes=source_contour_routes,
                classifier_models_by_route=classifier_routed_models,
                source_classifier_routes=source_classifier_routes,
            )
            if fusion_deployment is not None:
                batch_records = [records_by_patch_id[str(patch_id)] for patch_id in patch_ids]
                class_mean = apply_focus_classifier_fusion(
                    class_mean,
                    segmentation_mean,
                    images,
                    batch_records,
                    fusion_deployment,
                )
        class_mean = class_mean.cpu().numpy()
        segmentation_mean = segmentation_mean.cpu().numpy()
        for patch_id, class_probability, segmentation_probability in zip(
            patch_ids, class_mean, segmentation_mean
        ):
            atomic_npz(
                output / f"{patch_id}.npz",
                compressed=compress_predictions,
                durable=durable_predictions,
                classification_probability=np.asarray(class_probability, dtype=np.float32),
                segmentation_probability=segmentation_probability.astype(np.float16),
            )
            completed += 1
            written_predictions += 1
        print(f"inferred {completed}/{len(records)}", flush=True)
    (args.work / "INFERENCE_COMPLETE.json").write_text(
        json.dumps(
            {
                "candidates": len(records),
                "inference_plan": str(plan_path.resolve()),
                "inference_plan_sha256": file_sha256(plan_path),
                "inference_fingerprint": inference_fingerprint,
                "reused_predictions": reused_predictions,
                "written_predictions": written_predictions,
                "checkpoints": [str(path.resolve()) for path in args.checkpoint],
                "checkpoint_sha256": [file_sha256(path) for path in args.checkpoint],
                "contour_checkpoints": [
                    str(path.resolve()) for path in contour_paths
                ],
                "contour_checkpoint_sha256": [
                    file_sha256(path) for path in contour_paths
                ],
                "contour_input_modes": [
                    str(getattr(model, "candidate_refiner_input_mode", "soft"))
                    for model in contour_models
                ],
                "prediction_cache_compressed": compress_predictions,
                "prediction_cache_fsync": durable_predictions,
                "contour_source": (
                    "mixed_routed_candidate_refiner_focus"
                    if routed_models and focus_routed_models
                    else (
                        "gf_focus_swinunetr"
                        if contour_paths or focus_routed_models
                        else "candidate_refiner"
                    )
                ),
                "source_condition_groups": [
                    list(
                        getattr(
                            model, "candidate_refiner_source_condition_groups", ()
                        )
                    )
                    for model in models
                ],
                "candidate_contour_checkpoints": [
                    str(path.resolve()) for path in candidate_contour_paths
                ],
                "candidate_contour_checkpoint_sha256": [
                    file_sha256(path) for path in candidate_contour_paths
                ],
                "candidate_contour_source_condition_groups": [
                    list(
                        getattr(
                            model, "candidate_refiner_source_condition_groups", ()
                        )
                    )
                    for model in candidate_contour_models
                ],
                "routed_candidate_contours": {
                    name: {
                        "checkpoints": [str(path.resolve()) for path in routed_paths[name]],
                        "checkpoint_sha256": [
                            file_sha256(path) for path in routed_paths[name]
                        ],
                        "source_condition_groups": [
                            list(
                                getattr(
                                    model,
                                    "candidate_refiner_source_condition_groups",
                                    (),
                                )
                            )
                            for model in routed_models[name]
                        ],
                    }
                    for name in sorted(routed_paths)
                },
                "routed_focus_contours": {
                    name: {
                        "checkpoints": [
                            str(path.resolve()) for path in focus_routed_paths[name]
                        ],
                        "checkpoint_sha256": [
                            file_sha256(path) for path in focus_routed_paths[name]
                        ],
                        "input_modes": [
                            str(
                                getattr(
                                    model,
                                    "candidate_refiner_input_mode",
                                    "soft",
                                )
                            )
                            for model in focus_routed_models[name]
                        ],
                    }
                    for name in sorted(focus_routed_paths)
                },
                "source_contour_routes": dict(sorted(source_contour_routes.items())),
                "routed_classifiers": {
                    name: {
                        "checkpoints": [
                            str(path.resolve()) for path in classifier_routed_paths[name]
                        ],
                        "checkpoint_sha256": [
                            file_sha256(path) for path in classifier_routed_paths[name]
                        ],
                        "source_condition_groups": [
                            list(
                                getattr(
                                    model,
                                    "candidate_refiner_source_condition_groups",
                                    (),
                                )
                            )
                            for model in classifier_routed_models[name]
                        ],
                    }
                    for name in sorted(classifier_routed_paths)
                },
                "source_classifier_routes": dict(
                    sorted(source_classifier_routes.items())
                ),
                "classifier_fusion": (
                    {
                        "decision": str(fusion_path.resolve()),
                        "decision_sha256": file_sha256(fusion_path),
                        "focus_input_mode": str(
                            fusion_deployment["focus_input_mode"]
                        ),
                        "feature_names": list(DEPLOYMENT_FEATURE_NAMES),
                    }
                    if fusion_deployment is not None
                    else {}
                ),
            },
            indent=2,
        )
        + "\n"
    )


def local_target_grid(
    patch_shape: tuple[int, int, int],
    patch_affine: np.ndarray,
    source_shape: tuple[int, int, int],
    source_affine: np.ndarray,
) -> tuple[tuple[slice, slice, slice], np.ndarray, tuple[int, int, int]]:
    corners = np.asarray(
        list(itertools.product(*[(0, size - 1) for size in patch_shape])),
        dtype=np.float64,
    )
    world = nib.affines.apply_affine(patch_affine, corners)
    source_voxels = nib.affines.apply_affine(np.linalg.inv(source_affine), world)
    low = np.maximum(np.floor(source_voxels.min(axis=0)).astype(int) - 1, 0)
    high = np.minimum(
        np.ceil(source_voxels.max(axis=0)).astype(int) + 2,
        np.asarray(source_shape),
    )
    if np.any(high <= low):
        raise ValueError("patch does not overlap source grid")
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = low
    local_affine = source_affine @ translation
    local_shape = tuple(int(value) for value in high - low)
    return slices, local_affine, local_shape


def apply_local_replacement(
    output: np.ndarray,
    slices: tuple[slice, slice, slice],
    base_local: np.ndarray,
    refined: np.ndarray,
) -> None:
    """Replace only the observed crop footprint of a connected base lesion."""
    local_output = output[slices].copy()
    local_output[base_local] = False
    local_output |= refined
    output[slices] = local_output


def initialize_output(base: np.ndarray, empty_base: bool) -> np.ndarray:
    """Return an isolated deployment canvas with or without the base mask."""
    return np.zeros_like(base, dtype=bool) if empty_base else base.astype(bool, copy=True)


def parse_source_thresholds(values: list[str] | None) -> dict[str, float]:
    """Parse repeatable NAME=PROBABILITY source-specific classification gates."""
    thresholds: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"source threshold must be NAME=PROBABILITY: {value}")
        name, raw_threshold = value.split("=", 1)
        if not name or name in thresholds:
            raise ValueError(f"invalid or duplicate source threshold: {name!r}")
        threshold = float(raw_threshold)
        if not 0 < threshold < 1:
            raise ValueError(f"source threshold must be in (0,1): {value}")
        thresholds[name] = threshold
    return thresholds


MATERIALIZATION_MODES = {
    "seed",
    "seed_crop",
    "prune_base",
    "refined",
    "refined_or_seed",
    "seed_union",
    "seed_intersection",
    "replace",
    "approved_replace",
}


def parse_source_modes(values: list[str] | None) -> dict[str, str]:
    """Parse repeatable NAME=MODE source-specific deployment choices."""

    modes: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"source mode must be NAME=MODE: {value}")
        name, mode = value.split("=", 1)
        if not name or name in modes or mode not in MATERIALIZATION_MODES:
            raise ValueError(f"invalid or duplicate source mode: {value!r}")
        modes[name] = mode
    return modes


def record_classification_threshold(
    record: dict, global_threshold: float, source_thresholds: dict[str, float]
) -> float:
    source_name = record.get("source_name")
    return float(source_thresholds.get(str(source_name), global_threshold))


def record_segmentation_threshold(
    record: dict, global_threshold: float, source_thresholds: dict[str, float]
) -> float:
    source_name = record.get("source_name")
    return float(source_thresholds.get(str(source_name), global_threshold))


def materialize(args: argparse.Namespace) -> None:
    manifest = json.loads((args.work / "manifest.json").read_text())
    source_thresholds = parse_source_thresholds(
        getattr(args, "source_classification_threshold", None)
    )
    source_segmentation_thresholds = parse_source_thresholds(
        getattr(args, "source_segmentation_threshold", None)
    )
    source_modes = parse_source_modes(getattr(args, "source_mode", None))
    component_policy = str(manifest.get("component_policy", "union"))
    if source_thresholds or source_segmentation_thresholds or source_modes:
        if component_policy != "source_separate":
            raise ValueError(
                "source-specific thresholds/modes require a source-separated manifest"
            )
        known_sources = set(manifest.get("source_names", []))
        unknown_sources = (
            set(source_thresholds) | set(source_segmentation_thresholds) | set(source_modes)
        ) - known_sources
        if unknown_sources:
            raise ValueError(
                f"source settings reference unknown sources: {sorted(unknown_sources)}"
            )
    if args.empty_base and "prune_base" in source_modes.values():
        raise ValueError("prune_base requires a nonempty base deployment")
    records_by_case: dict[str, list[dict]] = defaultdict(list)
    for record in manifest["records"]:
        records_by_case[record["case"]].append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = []
    reused_cases: list[str] = []
    case_records = manifest.get("case_records")
    if not isinstance(case_records, list) or not case_records:
        raise ValueError("manifest contains no case records")
    case_limit = getattr(args, "case_limit", None)
    if case_limit:
        case_records = sorted(case_records, key=lambda value: value["case"])[
            :case_limit
        ]
    for case_index, case_record in enumerate(
        sorted(case_records, key=lambda value: value["case"]), 1
    ):
        case = case_record["case"]
        records = records_by_case.get(case, [])
        reference = nib.load(case_record["image"])
        output_path = args.output / f"{case}.nii.gz"
        if bool(getattr(args, "resume", False)) and output_path.is_file():
            existing = nib.load(str(output_path))
            if not same_grid(reference, existing):
                raise ValueError(f"resumed output geometry mismatch: {case}")
            existing_values = np.asanyarray(existing.dataobj)
            if not np.isfinite(existing_values).all():
                raise ValueError(f"resumed output contains non-finite values: {case}")
            if not np.isin(existing_values, (0, 1)).all():
                raise ValueError(f"resumed output is not a binary mask: {case}")
            reused_cases.append(case)
            print(
                f"[{case_index:03d}/{len(case_records):03d}] {case} [reused]",
                flush=True,
            )
            continue
        proposal_dirs = [Path(path).parent for path in case_record["source_proposals"]]
        if component_policy == "union":
            union, _, _ = load_sources(proposal_dirs, case, reference)
            component_map = cc3d.connected_components(
                union.astype(np.uint8), connectivity=26
            )

            def seed_for_component(component_id: int) -> np.ndarray:
                return component_map == component_id

        elif component_policy == "source_separate":
            source_masks, _ = load_source_masks(proposal_dirs, case, reference)
            separated_maps, separated_instances = source_separated_instances(
                source_masks
            )
            instance_by_id = {
                int(instance["component_id"]): instance
                for instance in separated_instances
            }

            def seed_for_component(component_id: int) -> np.ndarray:
                if component_id not in instance_by_id:
                    raise ValueError(
                        f"component inventory mismatch: {case} component {component_id}"
                    )
                return instance_mask(separated_maps, instance_by_id[component_id])

        else:
            raise ValueError(f"invalid component policy in manifest: {component_policy}")
        base_path = args.base / f"{case}.nii.gz"
        base_nii = nib.load(str(base_path))
        if not same_grid(reference, base_nii):
            raise ValueError(f"base geometry mismatch: {case}")
        base = np.asanyarray(base_nii.dataobj).astype(bool)
        output = initialize_output(base, args.empty_base)
        base_component_map = cc3d.connected_components(
            base.astype(np.uint8), connectivity=26
        )
        base_component_statistics = cc3d.statistics(base_component_map)
        base_component_voxels = np.asarray(
            base_component_statistics["voxel_counts"]
        )
        base_component_bboxes = base_component_statistics["bounding_boxes"]

        def clear_base_components(component_ids: np.ndarray) -> None:
            for base_id in map(int, component_ids):
                bbox = tuple(base_component_bboxes[base_id])
                output_local = output[bbox]
                output_local[base_component_map[bbox] == base_id] = False

        if float(manifest.get("tile_stride_mm", 0.0)) > 0:
            records_by_component: dict[int, list[dict]] = defaultdict(list)
            for record in records:
                records_by_component[int(record["component_id"])].append(record)
            for component_id, component_records in sorted(records_by_component.items()):
                seed = seed_for_component(component_id)
                component_source_names = {
                    str(record.get("source_name"))
                    for record in component_records
                    if record.get("source_name") is not None
                }
                if len(component_source_names) > 1:
                    raise ValueError(
                        f"component spans multiple source identities: {case} {component_id}"
                    )
                component_threshold = record_classification_threshold(
                    component_records[0],
                    args.classification_threshold,
                    source_thresholds,
                )
                component_segmentation_threshold = record_segmentation_threshold(
                    component_records[0],
                    args.segmentation_threshold,
                    source_segmentation_thresholds,
                )
                component_mode = source_modes.get(
                    str(component_records[0].get("source_name", "")), args.mode
                )
                prune_only = component_mode == "prune_base"
                refined_full = None if prune_only else np.zeros_like(seed, dtype=bool)
                classifications = []
                accepted_tiles = 0
                accepted_slices = []
                for record in sorted(
                    component_records, key=lambda value: int(value.get("tile_index", 0))
                ):
                    prediction_path = (
                        args.work / "predictions" / f"{record['patch_id']}.npz"
                    )
                    if not prediction_path.is_file():
                        raise FileNotFoundError(prediction_path)
                    with np.load(prediction_path, allow_pickle=False) as prediction:
                        classification = float(
                            prediction["classification_probability"]
                        )
                        if prune_only:
                            classifications.append(classification)
                            if classification >= component_threshold:
                                accepted_tiles += 1
                            continue
                        segmentation_probability = prediction[
                            "segmentation_probability"
                        ].astype(np.float32)[0]
                    with np.load(record["patch"], allow_pickle=False) as patch:
                        patch_affine = patch["patch_affine"]
                    classifications.append(classification)
                    if classification < component_threshold:
                        continue
                    slices, local_affine, local_shape = local_target_grid(
                        segmentation_probability.shape,
                        patch_affine,
                        reference.shape[:3],
                        reference.affine,
                    )
                    local_probability = resample_array(
                        segmentation_probability,
                        patch_affine,
                        local_affine,
                        local_shape,
                        1,
                        0.0,
                    )
                    assert refined_full is not None
                    refined_full[slices] |= (
                        local_probability >= component_segmentation_threshold
                    )
                    accepted_tiles += 1
                    accepted_slices.append(slices)

                seed_voxels = int(seed.sum())
                base_ids = np.unique(base_component_map[seed & (base_component_map > 0)])
                if prune_only:
                    refined_voxels = 0
                    seed_coverage = refined_overlap = volume_ratio = 0.0
                    base_refined_dice = 0.0
                    base_voxels = int(base_component_voxels[base_ids].sum()) if base_ids.size else 0
                    base_seed_intersection = int(
                        np.count_nonzero(np.isin(base_component_map[seed], base_ids))
                    ) if base_ids.size else 0
                    base_seed_coverage = base_seed_intersection / max(base_voxels, 1)
                    accepted = accepted_tiles > 0
                    if not accepted and base_ids.size:
                        clear_base_components(base_ids)
                else:
                    assert refined_full is not None
                    refined_voxels = int(refined_full.sum())
                    intersection = int(np.count_nonzero(refined_full & seed))
                    seed_coverage = intersection / max(seed_voxels, 1)
                    refined_overlap = intersection / max(refined_voxels, 1)
                    volume_ratio = refined_voxels / max(seed_voxels, 1)
                    base_region = (
                        np.isin(base_component_map, base_ids)
                        if base_ids.size
                        else np.zeros_like(seed)
                    )
                    base_voxels = int(base_region.sum())
                    base_seed_coverage = float(np.count_nonzero(seed & base_region)) / max(
                        base_voxels, 1
                    )
                    base_intersection = int(np.count_nonzero(refined_full & base_region))
                    base_refined_dice = (
                        2.0 * base_intersection
                        / max(refined_voxels + base_voxels, 1)
                    )
                    classified = accepted_tiles > 0
                    refinement_accepted = (
                        classified
                        and refined_voxels > 0
                        and seed_coverage >= args.min_seed_coverage
                        and refined_overlap >= args.min_refined_overlap
                        and args.min_volume_ratio <= volume_ratio <= args.max_volume_ratio
                        and (
                            component_mode not in ("replace", "approved_replace")
                            or (
                                base_ids.size > 0
                                and base_refined_dice >= args.min_base_dice
                                and base_seed_coverage >= args.min_base_seed_coverage
                            )
                        )
                    )
                    accepted = classified and (
                        component_mode in ("seed", "seed_crop", "refined_or_seed")
                        or refinement_accepted
                    )
                    if accepted:
                        if component_mode == "seed":
                            output |= seed
                        elif component_mode == "seed_crop":
                            for slices in accepted_slices:
                                output[slices] |= seed[slices]
                        elif component_mode == "refined":
                            output |= refined_full
                        elif component_mode == "refined_or_seed":
                            output |= refined_full if refinement_accepted else seed
                        elif component_mode == "seed_union":
                            output |= seed
                            output |= refined_full
                        elif component_mode == "seed_intersection":
                            output |= seed & refined_full
                        elif component_mode in ("replace", "approved_replace"):
                            if component_mode == "approved_replace":
                                output |= base_region
                            output[base_region] = False
                            output |= refined_full
                audit.append(
                    {
                        "case": case,
                        "patch_id": f"{case}__c{component_id:04d}",
                        "component_id": component_id,
                        "tiles": len(component_records),
                        "accepted_tiles": accepted_tiles,
                        "classification_probability": (
                            max(classifications) if classifications else 0.0
                        ),
                        "classification_threshold": component_threshold,
                        "segmentation_threshold": component_segmentation_threshold,
                        "source_name": component_records[0].get("source_name", ""),
                        "deployment_mode": component_mode,
                        "accepted": int(accepted),
                        "refinement_accepted": int(
                            refinement_accepted if not prune_only else False
                        ),
                        "pruned": int(component_mode == "prune_base" and not accepted),
                        "seed_mm3": component_records[0]["seed_mm3"],
                        "refined_voxels": refined_voxels,
                        "seed_coverage": seed_coverage,
                        "refined_overlap": refined_overlap,
                        "volume_ratio": volume_ratio,
                        "base_refined_dice": base_refined_dice,
                        "base_seed_coverage": base_seed_coverage,
                        "base_crop_coverage": 1.0 if base_ids.size else 0.0,
                    }
                )
            header = base_nii.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(
                nib.Nifti1Image(output.astype(np.uint8), base_nii.affine, header),
                str(output_path),
            )
            print(f"[{case_index:03d}/{len(case_records):03d}] {case}", flush=True)
            continue
        for record in records:
            component_id = int(record["component_id"])
            seed = seed_for_component(component_id)
            classification_threshold = record_classification_threshold(
                record, args.classification_threshold, source_thresholds
            )
            segmentation_threshold = record_segmentation_threshold(
                record, args.segmentation_threshold, source_segmentation_thresholds
            )
            component_mode = source_modes.get(str(record.get("source_name", "")), args.mode)
            prediction_path = args.work / "predictions" / f"{record['patch_id']}.npz"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            with np.load(prediction_path, allow_pickle=False) as prediction, np.load(
                record["patch"], allow_pickle=False
            ) as patch:
                classification = float(prediction["classification_probability"])
                segmentation_probability = prediction["segmentation_probability"].astype(np.float32)[0]
                patch_affine = patch["patch_affine"]
            classified = classification >= classification_threshold
            accepted = classified
            refinement_accepted = False
            seed_coverage = refined_overlap = volume_ratio = 0.0
            base_refined_dice = 0.0
            base_seed_coverage = 0.0
            base_crop_coverage = 0.0
            refined_voxels = 0
            if component_mode == "prune_base":
                base_ids = np.unique(base_component_map[seed & (base_component_map > 0)])
                base_region = (
                    np.isin(base_component_map, base_ids)
                    if base_ids.size
                    else np.zeros_like(seed)
                )
                base_seed_coverage = float(np.count_nonzero(seed & base_region)) / max(
                    int(base_region.sum()), 1
                )
                if not accepted:
                    output[base_region] = False
            elif classified and component_mode == "seed":
                output |= seed
            elif classified and component_mode == "seed_crop":
                slices, _, _ = local_target_grid(
                    segmentation_probability.shape,
                    patch_affine,
                    reference.shape[:3],
                    reference.affine,
                )
                output[slices] |= seed[slices]
            elif classified:
                slices, local_affine, local_shape = local_target_grid(
                    segmentation_probability.shape,
                    patch_affine,
                    reference.shape[:3],
                    reference.affine,
                )
                local_probability = resample_array(
                    segmentation_probability,
                    patch_affine,
                    local_affine,
                    local_shape,
                    1,
                    0.0,
                )
                refined = local_probability >= segmentation_threshold
                seed_local = seed[slices]
                intersection = int(np.count_nonzero(refined & seed_local))
                refined_voxels = int(refined.sum())
                seed_voxels = int(seed.sum())
                seed_coverage = intersection / max(seed_voxels, 1)
                refined_overlap = intersection / max(refined_voxels, 1)
                volume_ratio = refined_voxels / max(seed_voxels, 1)
                base_ids = np.unique(base_component_map[seed & (base_component_map > 0)])
                base_region = np.isin(base_component_map, base_ids) if base_ids.size else np.zeros_like(seed)
                base_local = base_region[slices]
                base_crop_coverage = float(base_local.sum()) / max(
                    int(base_region.sum()), 1
                )
                base_seed_coverage = float(np.count_nonzero(seed & base_region)) / max(
                    int(base_region.sum()), 1
                )
                base_intersection = int(np.count_nonzero(refined & base_local))
                base_refined_dice = (
                    2.0 * base_intersection
                    / max(int(refined.sum()) + int(base_local.sum()), 1)
                )
                refinement_accepted = (
                    refined_voxels > 0
                    and seed_coverage >= args.min_seed_coverage
                    and refined_overlap >= args.min_refined_overlap
                    and args.min_volume_ratio <= volume_ratio <= args.max_volume_ratio
                    and (
                        component_mode not in ("replace", "approved_replace")
                        or (
                            base_ids.size > 0
                            and base_refined_dice >= args.min_base_dice
                            and base_seed_coverage >= args.min_base_seed_coverage
                        )
                    )
                )
                accepted = refinement_accepted or component_mode == "refined_or_seed"
                if accepted:
                    if component_mode == "refined":
                        output[slices] |= refined
                    elif component_mode == "refined_or_seed":
                        if refinement_accepted:
                            output[slices] |= refined
                        else:
                            output |= seed
                    elif component_mode == "seed_intersection":
                        output[slices] |= refined & seed_local
                    elif component_mode == "replace":
                        apply_local_replacement(output, slices, base_local, refined)
                    elif component_mode == "approved_replace":
                        output |= base_region
                        apply_local_replacement(output, slices, base_local, refined)
                    elif component_mode == "seed_union":
                        output |= seed
                        output[slices] |= refined
                    else:
                        raise ValueError(f"unsupported materialization mode: {component_mode}")
            audit.append(
                {
                    "case": case,
                    "patch_id": record["patch_id"],
                    "component_id": component_id,
                    "classification_probability": classification,
                    "classification_threshold": classification_threshold,
                    "segmentation_threshold": segmentation_threshold,
                    "source_name": record.get("source_name", ""),
                    "deployment_mode": component_mode,
                    "accepted": int(accepted),
                    "refinement_accepted": int(refinement_accepted),
                    "pruned": int(component_mode == "prune_base" and not accepted),
                    "seed_mm3": record["seed_mm3"],
                    "refined_voxels": refined_voxels,
                    "seed_coverage": seed_coverage,
                    "refined_overlap": refined_overlap,
                    "volume_ratio": volume_ratio,
                    "base_refined_dice": base_refined_dice,
                    "base_seed_coverage": base_seed_coverage,
                    "base_crop_coverage": base_crop_coverage,
                }
            )
        header = base_nii.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(
            nib.Nifti1Image(output.astype(np.uint8), base_nii.affine, header),
            str(output_path),
        )
        print(f"[{case_index:03d}/{len(case_records):03d}] {case}", flush=True)
    audit_path = args.output.with_name(args.output.name + ".components.csv")
    with audit_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]) if audit else ["case"])
        writer.writeheader()
        writer.writerows(audit)
    (args.output / "CANDIDATE_REFINER_COMPLETE.json").write_text(
        json.dumps(
            {
                "cases": len(case_records),
                "materialized_cases_this_invocation": len(case_records)
                - len(reused_cases),
                "reused_case_masks_verified": len(reused_cases),
                "reused_cases": reused_cases,
                "component_audit_scope": (
                    "new_cases_only" if reused_cases else "all_cases"
                ),
                "candidates_audited_this_invocation": len(audit),
                "accepted_audited_this_invocation": sum(
                    row["accepted"] for row in audit
                ),
                "classification_threshold": args.classification_threshold,
                "source_classification_thresholds": source_thresholds,
                "segmentation_threshold": args.segmentation_threshold,
                "source_segmentation_thresholds": source_segmentation_thresholds,
                "source_deployment_modes": source_modes,
                "mode": args.mode,
                "empty_base": bool(args.empty_base),
                "component_policy": str(
                    manifest.get("component_policy", "union")
                ),
                "gates": {
                    "min_seed_coverage": args.min_seed_coverage,
                    "min_refined_overlap": args.min_refined_overlap,
                    "min_volume_ratio": args.min_volume_ratio,
                    "max_volume_ratio": args.max_volume_ratio,
                    "min_base_dice": args.min_base_dice,
                    "min_base_seed_coverage": args.min_base_seed_coverage,
                },
                "base": str(args.base.resolve()),
                "audit": str(audit_path.resolve()),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--images", type=Path, required=True)
    prepare_parser.add_argument("--proposal", action="append", default=[])
    prepare_parser.add_argument("--base", type=Path)
    prepare_parser.add_argument("--only-disconnected", action="store_true")
    prepare_parser.add_argument("--only-connected", action="store_true")
    prepare_parser.add_argument("--work", type=Path, required=True)
    prepare_parser.add_argument("--shape", default="64,64,64")
    prepare_parser.add_argument("--spacing", default="1.5,1.5,1.5")
    prepare_parser.add_argument("--min-mm3", type=float, default=5.0)
    prepare_parser.add_argument("--max-mm3", type=float, default=20000.0)
    prepare_parser.add_argument("--bbox-margin-mm", type=float, default=0.0)
    prepare_parser.add_argument("--tile-stride-mm", type=float, default=0.0)
    prepare_parser.add_argument(
        "--component-policy",
        choices=("union", "source_separate"),
        default="union",
    )
    prepare_parser.add_argument("--case-limit", type=int, default=0)
    prepare_parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a finite cached patch only when its shape, affine, and seed match.",
    )

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--work", type=Path, required=True)
    infer_parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    infer_parser.add_argument(
        "--contour-checkpoint",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional fine-tuned GF Focus checkpoint(s) used only for contour "
            "probabilities; candidate confidence still comes from --checkpoint."
        ),
    )
    infer_parser.add_argument(
        "--candidate-contour-checkpoint",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional candidate-refiner checkpoint(s) used only for contour "
            "probabilities; confidence still comes from --checkpoint. This is "
            "mutually exclusive with --contour-checkpoint."
        ),
    )
    infer_parser.add_argument(
        "--candidate-contour-route",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help=(
            "Candidate-refiner contour checkpoint for a named route. Repeat a "
            "name to ensemble checkpoints within that route. Requires a complete "
            "--source-contour-route declaration."
        ),
    )
    infer_parser.add_argument(
        "--focus-contour-route",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help=(
            "GF Focus contour checkpoint for a named route. Repeat a name to "
            "ensemble checkpoints within that route. Requires a complete "
            "--source-contour-route declaration."
        ),
    )
    infer_parser.add_argument(
        "--source-contour-route",
        action="append",
        default=[],
        metavar="SOURCE=NAME",
        help="Assign every source-separated proposal stream to one contour route.",
    )
    infer_parser.add_argument(
        "--classifier-route",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help=(
            "Candidate-classifier checkpoint for a named source-family route. "
            "Repeat a name to ensemble checkpoints within that route."
        ),
    )
    infer_parser.add_argument(
        "--source-classifier-route",
        action="append",
        default=[],
        metavar="SOURCE=NAME",
        help="Assign every proposal source to one classifier route.",
    )
    infer_parser.add_argument(
        "--classifier-fusion",
        type=Path,
        help=(
            "Qualified exact707 OOF Focus/B32 classifier-fusion decision. "
            "Requires a global --contour-checkpoint family."
        ),
    )
    infer_parser.add_argument("--batch-size", type=int, default=8)
    infer_parser.add_argument("--workers", type=int, default=2)
    infer_parser.add_argument(
        "--uncompressed-predictions",
        action="store_true",
        help=(
            "write dense probability caches as atomic ZIP_STORED NPZ files; "
            "use only for large temporary sweeps with sufficient disk space"
        ),
    )
    infer_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse only finite, shape-valid probability files whose cache mode "
            "matches the hash-bound inference plan"
        ),
    )

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--work", type=Path, required=True)
    materialize_parser.add_argument("--base", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse an existing output mask only after verifying reference-grid "
            "geometry, finite values, and binary labels. The component audit "
            "then explicitly covers newly materialized cases only."
        ),
    )
    materialize_parser.add_argument(
        "--empty-base",
        action="store_true",
        help="start from an empty mask so only accepted candidates are emitted",
    )
    materialize_parser.add_argument("--classification-threshold", type=float, required=True)
    materialize_parser.add_argument(
        "--source-classification-threshold",
        action="append",
        default=[],
        metavar="NAME=PROBABILITY",
        help="Override the classification gate for one source-separated proposal stream.",
    )
    materialize_parser.add_argument("--segmentation-threshold", type=float, default=0.5)
    materialize_parser.add_argument(
        "--source-segmentation-threshold",
        action="append",
        default=[],
        metavar="NAME=PROBABILITY",
        help="Override the contour threshold for one source-separated proposal stream.",
    )
    materialize_parser.add_argument(
        "--mode",
        choices=(
            "seed",
            "seed_crop",
            "refined",
            "refined_or_seed",
            "seed_union",
            "seed_intersection",
            "replace",
            "approved_replace",
        ),
        default="refined",
    )
    materialize_parser.add_argument(
        "--source-mode",
        action="append",
        default=[],
        metavar="NAME=MODE",
        help="Override the deployment mode for one source-separated proposal stream.",
    )
    materialize_parser.add_argument("--min-seed-coverage", type=float, default=0.05)
    materialize_parser.add_argument("--min-refined-overlap", type=float, default=0.05)
    materialize_parser.add_argument("--min-volume-ratio", type=float, default=0.25)
    materialize_parser.add_argument("--max-volume-ratio", type=float, default=4.0)
    materialize_parser.add_argument(
        "--case-limit",
        type=int,
        default=0,
        help="Optional deterministic prefix for parity tests; zero processes all cases.",
    )
    materialize_parser.add_argument("--min-base-dice", type=float, default=0.25)
    materialize_parser.add_argument(
        "--min-base-seed-coverage", type=float, default=0.25
    )
    args = parser.parse_args()
    if args.command == "prepare":
        if args.only_disconnected and args.only_connected:
            parser.error("--only-disconnected and --only-connected are mutually exclusive")
        if (args.only_disconnected or args.only_connected) and args.base is None:
            parser.error("connected/disconnected selection requires --base")
        if args.bbox_margin_mm < 0 or args.tile_stride_mm < 0:
            parser.error("invalid bounding-box tiling geometry")
        prepare(args)
    elif args.command == "infer":
        routed_candidate = bool(args.candidate_contour_route)
        routed_focus = bool(args.focus_contour_route)
        routed = routed_candidate or routed_focus or bool(args.source_contour_route)
        candidate_route_names = set(
            parse_named_checkpoint_routes(args.candidate_contour_route)
        )
        focus_route_names = set(parse_named_checkpoint_routes(args.focus_contour_route))
        overlap = sorted(candidate_route_names & focus_route_names)
        if overlap:
            parser.error(
                "candidate-refiner and Focus contour route names overlap: "
                f"{overlap}"
            )
        if sum(
            bool(value)
            for value in (
                args.contour_checkpoint,
                args.candidate_contour_checkpoint,
                routed,
            )
        ) > 1:
            parser.error(
                "Focus, global candidate-refiner, and routed contour overrides are mutually exclusive"
            )
        if (routed_candidate or routed_focus) != bool(args.source_contour_route):
            parser.error(
                "a routed contour family and --source-contour-route are required together"
            )
        if bool(args.classifier_route) != bool(args.source_classifier_route):
            parser.error(
                "--classifier-route and --source-classifier-route are required together"
            )
        if args.classifier_fusion and (
            not args.contour_checkpoint
            or args.candidate_contour_checkpoint
            or routed
            or args.classifier_route
        ):
            parser.error(
                "--classifier-fusion requires global --contour-checkpoint only"
            )
        infer(args)
    else:
        if not 0 < args.classification_threshold < 1 or not 0 < args.segmentation_threshold < 1:
            parser.error("probability thresholds must be in (0,1)")
        materialize(args)


if __name__ == "__main__":
    main()
