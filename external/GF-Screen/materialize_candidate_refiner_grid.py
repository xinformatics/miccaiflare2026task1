#!/usr/bin/env python3
"""Materialize many full-volume refiner configurations in one case pass.

The reference CLI materializes one threshold/mode at a time and consequently
reloads full CT/proposal volumes and resamples every accepted tile repeatedly.
This implementation preserves its tile-stride deployment semantics while
sharing component extraction, prediction reads, and spatial resampling across
an explicitly frozen configuration grid.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np

from apply_candidate_refiner import (
    initialize_output,
    load_source_masks,
    load_sources,
    local_target_grid,
    resample_array,
    same_grid,
)
from proposal_component_instances import (
    instance_local_mask,
    source_separated_instances,
)


ALLOWED_MODES = {
    "refined",
    "refined_or_seed",
    "seed",
    "seed_crop",
    "seed_union",
    "prune_base",
}
GEOMETRY_GATE_KEYS = {
    "min_seed_coverage",
    "min_refined_overlap",
    "min_volume_ratio",
    "max_volume_ratio",
}
AUDIT_FIELDNAMES = (
    "case",
    "patch_id",
    "component_id",
    "tiles",
    "accepted_tiles",
    "classification_probability",
    "classification_threshold",
    "segmentation_threshold",
    "source_name",
    "deployment_mode",
    "accepted",
    "refinement_accepted",
    "pruned",
    "seed_mm3",
    "refined_voxels",
    "seed_coverage",
    "refined_overlap",
    "volume_ratio",
    "base_refined_dice",
    "base_seed_coverage",
    "base_crop_coverage",
)


def load_configurations(path: Path, known_sources: set[str]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs = payload.get("configurations", payload) if isinstance(payload, dict) else payload
    if not isinstance(configs, list) or not configs:
        raise ValueError("configuration grid is empty")
    result = []
    seen = set()
    for raw in configs:
        if not isinstance(raw, dict):
            raise ValueError("each configuration must be an object")
        tag = str(raw.get("tag", ""))
        threshold = float(raw.get("classification_threshold"))
        segmentation_threshold = (
            float(raw["segmentation_threshold"])
            if raw.get("segmentation_threshold") is not None
            else None
        )
        mode = str(raw.get("mode", ""))
        source_thresholds = {
            str(name): float(value)
            for name, value in dict(raw.get("source_classification_thresholds", {})).items()
        }
        source_segmentation_thresholds = {
            str(name): float(value)
            for name, value in dict(
                raw.get("source_segmentation_thresholds", {})
            ).items()
        }
        source_modes = {
            str(name): str(value)
            for name, value in dict(raw.get("source_deployment_modes", {})).items()
        }
        geometry_gates = {
            str(name): float(value)
            for name, value in dict(raw.get("geometry_gates", {})).items()
        }
        if not tag or "/" in tag or "\\" in tag or tag in seen:
            raise ValueError(f"invalid or duplicate configuration tag: {tag!r}")
        if not 0 < threshold < 1 or mode not in ALLOWED_MODES:
            raise ValueError(f"invalid threshold/mode for {tag}")
        if segmentation_threshold is not None and not 0 < segmentation_threshold < 1:
            raise ValueError(f"invalid segmentation threshold for {tag}")
        if set(source_thresholds) - known_sources:
            raise ValueError(
                f"unknown source thresholds for {tag}: {sorted(set(source_thresholds)-known_sources)}"
            )
        if set(source_segmentation_thresholds) - known_sources:
            raise ValueError(
                f"unknown source segmentation thresholds for {tag}: "
                f"{sorted(set(source_segmentation_thresholds)-known_sources)}"
            )
        if set(source_modes) - known_sources:
            raise ValueError(
                f"unknown source deployment modes for {tag}: "
                f"{sorted(set(source_modes)-known_sources)}"
            )
        if any(value not in ALLOWED_MODES for value in source_modes.values()):
            raise ValueError(f"invalid source deployment mode for {tag}")
        if set(geometry_gates) - GEOMETRY_GATE_KEYS:
            raise ValueError(f"unknown geometry gate for {tag}")
        for name in ("min_seed_coverage", "min_refined_overlap"):
            if name in geometry_gates and not 0 <= geometry_gates[name] <= 1:
                raise ValueError(f"invalid geometry gate for {tag}: {name}")
        if any(
            geometry_gates.get(name, 0) < 0
            for name in ("min_volume_ratio", "max_volume_ratio")
        ):
            raise ValueError(f"invalid volume-ratio gate for {tag}")
        if (
            "min_volume_ratio" in geometry_gates
            and "max_volume_ratio" in geometry_gates
            and geometry_gates["min_volume_ratio"]
            > geometry_gates["max_volume_ratio"]
        ):
            raise ValueError(f"inverted volume-ratio gates for {tag}")
        if any(not 0 < value < 1 for value in source_thresholds.values()):
            raise ValueError(f"invalid source threshold for {tag}")
        if any(
            not 0 < value < 1 for value in source_segmentation_thresholds.values()
        ):
            raise ValueError(f"invalid source segmentation threshold for {tag}")
        seen.add(tag)
        result.append(
            {
                "tag": tag,
                "classification_threshold": threshold,
                "segmentation_threshold": segmentation_threshold,
                "mode": mode,
                "source_classification_thresholds": source_thresholds,
                "source_segmentation_thresholds": source_segmentation_thresholds,
                "source_deployment_modes": source_modes,
                "geometry_gates": geometry_gates,
            }
        )
    return result


def effective_threshold(config: dict, source_name: str) -> float:
    return float(
        config["source_classification_thresholds"].get(
            source_name, config["classification_threshold"]
        )
    )


def effective_segmentation_threshold(
    config: dict, source_name: str, global_threshold: float
) -> float:
    configured_global = config.get("segmentation_threshold")
    return float(
        config["source_segmentation_thresholds"].get(
            source_name,
            global_threshold if configured_global is None else configured_global,
        )
    )


def effective_mode(config: dict, source_name: str) -> str:
    return str(config["source_deployment_modes"].get(source_name, config["mode"]))


def merge_local_tiles(
    accepted_tiles: list[tuple[float, tuple[slice, slice, slice], np.ndarray]],
) -> tuple[tuple[slice, slice, slice] | None, np.ndarray | None]:
    """Union accepted native-grid tile masks without allocating a full volume."""

    if not accepted_tiles:
        return None, None
    low = np.min(
        [[int(axis.start) for axis in slices] for _, slices, _ in accepted_tiles],
        axis=0,
    )
    high = np.max(
        [[int(axis.stop) for axis in slices] for _, slices, _ in accepted_tiles],
        axis=0,
    )
    merged = np.zeros(tuple((high - low).tolist()), dtype=bool)
    for _, slices, local_refined in accepted_tiles:
        destination = tuple(
            slice(int(axis.start - low[index]), int(axis.stop - low[index]))
            for index, axis in enumerate(slices)
        )
        if merged[destination].shape != local_refined.shape:
            raise ValueError("local refined tile shape does not match native slices")
        merged[destination] |= local_refined
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high)), merged


def overlap_views(
    first: tuple[slice, slice, slice],
    second: tuple[slice, slice, slice],
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]] | None:
    low = np.maximum(
        [int(axis.start) for axis in first], [int(axis.start) for axis in second]
    )
    high = np.minimum(
        [int(axis.stop) for axis in first], [int(axis.stop) for axis in second]
    )
    if np.any(high <= low):
        return None
    first_low = np.asarray([int(axis.start) for axis in first])
    second_low = np.asarray([int(axis.start) for axis in second])
    first_view = tuple(
        slice(int(a), int(b)) for a, b in zip(low - first_low, high - first_low)
    )
    second_view = tuple(
        slice(int(a), int(b)) for a, b in zip(low - second_low, high - second_low)
    )
    return first_view, second_view


def materialize_grid(
    work: Path,
    base_root: Path,
    output_root: Path,
    configuration_path: Path,
    fold: int,
    segmentation_threshold: float,
    empty_base: bool,
    min_seed_coverage: float,
    min_refined_overlap: float,
    min_volume_ratio: float,
    max_volume_ratio: float,
    case_limit: int = 0,
    writer_workers: int = 2,
) -> None:
    if not 0 < segmentation_threshold < 1 or fold < 0:
        raise ValueError("invalid segmentation threshold or fold")
    if writer_workers not in (1, 2):
        raise ValueError("writer_workers must be 1 or 2")
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    if float(manifest.get("tile_stride_mm", 0.0)) <= 0:
        raise ValueError("grid materializer requires a positive tile stride")
    component_policy = str(manifest.get("component_policy", "union"))
    if component_policy not in {"union", "source_separate"}:
        raise ValueError(f"unsupported component policy: {component_policy}")
    known_sources = set(map(str, manifest.get("source_names", [])))
    configs = load_configurations(configuration_path, known_sources)
    if empty_base and any(
        "prune_base" in config["source_deployment_modes"].values()
        or config["mode"] == "prune_base"
        for config in configs
    ):
        raise ValueError("prune_base requires a nonempty base deployment")
    if component_policy != "source_separate" and any(
        config["source_classification_thresholds"]
        or config["source_segmentation_thresholds"]
        or config["source_deployment_modes"]
        for config in configs
    ):
        raise ValueError("source thresholds require source-separated components")

    records_by_case: dict[str, list[dict]] = defaultdict(list)
    for record in manifest.get("records", []):
        records_by_case[str(record["case"])].append(record)
    case_records = manifest.get("case_records")
    if not isinstance(case_records, list) or not case_records:
        raise ValueError("manifest contains no case records")
    case_records = sorted(case_records, key=lambda value: str(value["case"]))
    if case_limit:
        case_records = case_records[:case_limit]

    outputs = {
        config["tag"]: output_root / config["tag"] / f"fold{fold}"
        for config in configs
    }
    for directory in outputs.values():
        directory.mkdir(parents=True, exist_ok=True)
        # A previous terminal must never remain visible while its payload is
        # being rewritten. The new marker is committed only after every mask
        # write and streamed audit has completed.
        (directory / "CANDIDATE_REFINER_COMPLETE.json").unlink(missing_ok=True)
    # Stream component audits instead of retaining one Python dictionary per
    # candidate and configuration.  The main twelve-point grid can otherwise
    # accumulate hundreds of thousands of large dictionaries per fold.
    audit_paths = {
        tag: directory.with_name(directory.name + ".components.csv")
        for tag, directory in outputs.items()
    }
    audit_streams = {
        tag: path.open("w", newline="", encoding="utf-8")
        for tag, path in audit_paths.items()
    }
    audit_writers = {
        tag: csv.DictWriter(stream, fieldnames=AUDIT_FIELDNAMES)
        for tag, stream in audit_streams.items()
    }
    for writer in audit_writers.values():
        writer.writeheader()
    audit_counts = {
        config["tag"]: {"candidates": 0, "accepted": 0} for config in configs
    }

    for case_index, case_record in enumerate(case_records, 1):
        case = str(case_record["case"])
        records = records_by_case.get(case, [])
        reference = nib.load(str(case_record["image"]))
        proposal_dirs = [Path(path).parent for path in case_record["source_proposals"]]
        if component_policy == "union":
            union, _, _ = load_sources(proposal_dirs, case, reference)
            component_map = cc3d.connected_components(
                union.astype(np.uint8), connectivity=26
            )
            component_statistics = cc3d.statistics(component_map)

            def seed_for_component(
                component_id: int,
            ) -> tuple[tuple[slice, slice, slice], np.ndarray]:
                bbox = tuple(component_statistics["bounding_boxes"][component_id])
                return bbox, component_map[bbox] == component_id

        else:
            source_masks, _ = load_source_masks(proposal_dirs, case, reference)
            separated_maps, separated_instances = source_separated_instances(source_masks)
            instance_by_id = {
                int(instance["component_id"]): instance
                for instance in separated_instances
            }

            def seed_for_component(
                component_id: int,
            ) -> tuple[tuple[slice, slice, slice], np.ndarray]:
                if component_id not in instance_by_id:
                    raise ValueError(
                        f"component inventory mismatch: {case} component {component_id}"
                    )
                instance = instance_by_id[component_id]
                return tuple(instance["bbox"]), instance_local_mask(
                    separated_maps, instance
                )

        base_path = base_root / f"{case}.nii.gz"
        base_nii = nib.load(str(base_path))
        if not same_grid(reference, base_nii):
            raise ValueError(f"base geometry mismatch: {case}")
        base = np.asanyarray(base_nii.dataobj).astype(bool)
        # Retain sparse, local operations while components are evaluated and
        # construct only one full-volume configuration at a time below.  A
        # dense array per configuration multiplied peak memory by the grid
        # size (twelve full CT masks in the main OOF sweep) and could trigger
        # the host pressure guard even though the refined regions are sparse.
        case_operations: dict[str, list[tuple]] = {
            config["tag"]: [] for config in configs
        }
        base_component_map = cc3d.connected_components(
            base.astype(np.uint8), connectivity=26
        )
        base_component_statistics = cc3d.statistics(base_component_map)
        base_component_voxels = np.asarray(
            base_component_statistics["voxel_counts"]
        )
        base_component_bboxes = base_component_statistics["bounding_boxes"]

        def clear_base_components(target: np.ndarray, component_ids: np.ndarray) -> None:
            """Clear whole base components using only their local bounding boxes."""

            for base_id in map(int, component_ids):
                bbox = tuple(base_component_bboxes[base_id])
                target_local = target[bbox]
                target_local[base_component_map[bbox] == base_id] = False
        records_by_component: dict[int, list[dict]] = defaultdict(list)
        for record in records:
            records_by_component[int(record["component_id"])].append(record)

        for component_id, component_records in sorted(records_by_component.items()):
            component_bbox, seed_local = seed_for_component(component_id)
            source_names = {
                str(record.get("source_name"))
                for record in component_records
                if record.get("source_name") is not None
            }
            if len(source_names) > 1:
                raise ValueError(
                    f"component spans multiple source identities: {case} {component_id}"
                )
            source_name = next(iter(source_names), "")
            operating_points = sorted(
                {
                    (
                        effective_threshold(config, source_name),
                        effective_segmentation_threshold(
                            config, source_name, segmentation_threshold
                        ),
                    )
                    for config in configs
                }
            )
            minimum_threshold = min(point[0] for point in operating_points)
            tile_results: list[
                tuple[float, tuple[slice, slice, slice], np.ndarray]
            ] = []
            classifications = []
            for record in sorted(
                component_records, key=lambda value: int(value.get("tile_index", 0))
            ):
                prediction_path = work / "predictions" / f"{record['patch_id']}.npz"
                if not prediction_path.is_file():
                    raise FileNotFoundError(prediction_path)
                with np.load(prediction_path, allow_pickle=False) as prediction, np.load(
                    record["patch"], allow_pickle=False
                ) as patch:
                    classification = float(prediction["classification_probability"])
                    classifications.append(classification)
                    if classification < minimum_threshold:
                        continue
                    probability = prediction["segmentation_probability"].astype(np.float32)[0]
                    patch_affine = patch["patch_affine"]
                slices, local_affine, local_shape = local_target_grid(
                    probability.shape,
                    patch_affine,
                    reference.shape[:3],
                    reference.affine,
                )
                local_probability = resample_array(
                    probability,
                    patch_affine,
                    local_affine,
                    local_shape,
                    1,
                    0.0,
                )
                tile_results.append(
                    (classification, slices, local_probability)
                )

            seed_voxels = int(seed_local.sum())
            base_component_local = base_component_map[component_bbox]
            base_ids = np.unique(base_component_local[seed_local & (base_component_local > 0)])
            base_voxels = int(base_component_voxels[base_ids].sum()) if base_ids.size else 0
            base_seed_coverage = float(
                np.count_nonzero(seed_local & np.isin(base_component_local, base_ids))
            ) / max(base_voxels, 1)
            configs_by_operating_point: dict[tuple[float, float], list[dict]] = defaultdict(list)
            for config in configs:
                point = (
                    effective_threshold(config, source_name),
                    effective_segmentation_threshold(
                        config, source_name, segmentation_threshold
                    ),
                )
                configs_by_operating_point[point].append(config)
            for threshold, contour_threshold in operating_points:
                accepted_tiles = [
                    (classification, slices, probability >= contour_threshold)
                    for classification, slices, probability in tile_results
                    if classification >= threshold
                ]
                refined_bbox, refined_local = merge_local_tiles(accepted_tiles)
                refined_voxels = int(refined_local.sum()) if refined_local is not None else 0
                overlap = (
                    overlap_views(component_bbox, refined_bbox)
                    if refined_bbox is not None
                    else None
                )
                intersection = (
                    int(np.count_nonzero(seed_local[overlap[0]] & refined_local[overlap[1]]))
                    if overlap is not None and refined_local is not None
                    else 0
                )
                seed_coverage = intersection / max(seed_voxels, 1)
                refined_overlap = intersection / max(refined_voxels, 1)
                volume_ratio = refined_voxels / max(seed_voxels, 1)
                base_intersection = (
                    int(
                        np.count_nonzero(
                            refined_local
                            & np.isin(base_component_map[refined_bbox], base_ids)
                        )
                    )
                    if refined_bbox is not None and refined_local is not None and base_ids.size
                    else 0
                )
                base_refined_dice = 2.0 * base_intersection / max(
                    refined_voxels + base_voxels, 1
                )
                for config in configs_by_operating_point[(threshold, contour_threshold)]:
                    mode = effective_mode(config, source_name)
                    classified = len(accepted_tiles) > 0
                    gates = {
                        "min_seed_coverage": min_seed_coverage,
                        "min_refined_overlap": min_refined_overlap,
                        "min_volume_ratio": min_volume_ratio,
                        "max_volume_ratio": max_volume_ratio,
                        **config["geometry_gates"],
                    }
                    if gates["min_volume_ratio"] > gates["max_volume_ratio"]:
                        raise ValueError(
                            f"inverted effective volume-ratio gates for {config['tag']}"
                        )
                    refinement_accepted = classified and (
                        refined_voxels > 0
                        and seed_coverage >= gates["min_seed_coverage"]
                        and refined_overlap >= gates["min_refined_overlap"]
                        and gates["min_volume_ratio"]
                        <= volume_ratio
                        <= gates["max_volume_ratio"]
                    )
                    accepted = classified and (
                        mode in ("seed", "seed_crop", "prune_base")
                        or mode == "refined_or_seed"
                        or refinement_accepted
                    )
                    if accepted:
                        if mode == "seed":
                            case_operations[config["tag"]].append(
                                ("or", component_bbox, seed_local)
                            )
                        elif mode == "seed_crop":
                            cropped_seed = np.zeros_like(seed_local)
                            for _, tile_slices, _ in accepted_tiles:
                                overlap = overlap_views(component_bbox, tile_slices)
                                if overlap is not None:
                                    cropped_seed[overlap[0]] |= seed_local[overlap[0]]
                            case_operations[config["tag"]].append(
                                ("or", component_bbox, cropped_seed)
                            )
                        elif mode == "prune_base":
                            pass
                        elif mode == "refined":
                            assert refined_bbox is not None and refined_local is not None
                            case_operations[config["tag"]].append(
                                ("or", refined_bbox, refined_local)
                            )
                        elif mode == "refined_or_seed":
                            if refinement_accepted:
                                assert refined_bbox is not None and refined_local is not None
                                case_operations[config["tag"]].append(
                                    ("or", refined_bbox, refined_local)
                                )
                            else:
                                case_operations[config["tag"]].append(
                                    ("or", component_bbox, seed_local)
                                )
                        elif mode == "seed_union":
                            assert refined_bbox is not None and refined_local is not None
                            case_operations[config["tag"]].extend(
                                [
                                    ("or", component_bbox, seed_local),
                                    ("or", refined_bbox, refined_local),
                                ]
                            )
                    elif mode == "prune_base" and base_ids.size:
                        case_operations[config["tag"]].append(
                            ("prune", base_ids.copy(), None)
                        )
                    audit_row = {
                            "case": case,
                            "patch_id": f"{case}__c{component_id:04d}",
                            "component_id": component_id,
                            "tiles": len(component_records),
                            "accepted_tiles": len(accepted_tiles),
                            "classification_probability": max(classifications)
                            if classifications
                            else 0.0,
                            "classification_threshold": threshold,
                            "segmentation_threshold": contour_threshold,
                            "source_name": source_name,
                            "deployment_mode": mode,
                            "accepted": int(accepted),
                            "refinement_accepted": int(refinement_accepted),
                            "pruned": int(mode == "prune_base" and not accepted),
                            "seed_mm3": component_records[0]["seed_mm3"],
                            "refined_voxels": refined_voxels,
                            "seed_coverage": seed_coverage,
                            "refined_overlap": refined_overlap,
                            "volume_ratio": volume_ratio,
                            "base_refined_dice": base_refined_dice,
                            "base_seed_coverage": base_seed_coverage,
                            "base_crop_coverage": 1.0 if base_ids.size else 0.0,
                        }
                    audit_writers[config["tag"]].writerow(audit_row)
                    audit_counts[config["tag"]]["candidates"] += 1
                    audit_counts[config["tag"]]["accepted"] += int(accepted)

        # Keep masks in their final uint8 dtype so saving does not allocate a
        # second full-volume cast. At most two independent gzip writes overlap;
        # this lets the larger main grid use otherwise idle CPU cores without
        # restoring the former one-dense-volume-per-configuration footprint.
        pending_writes = []
        with ThreadPoolExecutor(
            max_workers=min(writer_workers, len(configs))
        ) as writer_pool:
            for config in configs:
                case_output = initialize_output(base, empty_base).astype(
                    np.uint8, copy=False
                )
                for operation, location, local_mask in case_operations[config["tag"]]:
                    if operation == "or":
                        case_output[location] |= local_mask
                    elif operation == "prune":
                        clear_base_components(case_output, location)
                    else:
                        raise ValueError(f"unknown deferred operation: {operation}")
                header = base_nii.header.copy()
                header.set_data_dtype(np.uint8)
                image = nib.Nifti1Image(
                    case_output,
                    base_nii.affine.copy(),
                    header,
                )
                pending_writes.append(
                    writer_pool.submit(
                        nib.save,
                        image,
                        str(outputs[config["tag"]] / f"{case}.nii.gz"),
                    )
                )
                if len(pending_writes) >= 2:
                    pending_writes.pop(0).result()
                del case_output, image
            for pending_write in pending_writes:
                pending_write.result()
        print(f"[{case_index:03d}/{len(case_records):03d}] {case}", flush=True)

    for stream in audit_streams.values():
        stream.close()
    for config in configs:
        tag = config["tag"]
        audit_path = audit_paths[tag]
        marker = outputs[tag] / "CANDIDATE_REFINER_COMPLETE.json"
        marker.write_text(
            json.dumps(
                {
                    "cases": len(case_records),
                    "candidates": audit_counts[tag]["candidates"],
                    "accepted": audit_counts[tag]["accepted"],
                    "classification_threshold": config["classification_threshold"],
                    "source_classification_thresholds": config[
                        "source_classification_thresholds"
                    ],
                    "segmentation_threshold": (
                        config["segmentation_threshold"]
                        if config["segmentation_threshold"] is not None
                        else segmentation_threshold
                    ),
                    "fallback_segmentation_threshold": segmentation_threshold,
                    "source_segmentation_thresholds": config[
                        "source_segmentation_thresholds"
                    ],
                    "source_deployment_modes": config["source_deployment_modes"],
                    "mode": config["mode"],
                    "empty_base": bool(empty_base),
                    "component_policy": component_policy,
                    "gates": {
                        "min_seed_coverage": min_seed_coverage,
                        "min_refined_overlap": min_refined_overlap,
                        "min_volume_ratio": min_volume_ratio,
                        "max_volume_ratio": max_volume_ratio,
                        **config["geometry_gates"],
                    },
                    "base": str(base_root.resolve()),
                    "audit": str(audit_path.resolve()),
                    "materializer": "shared_case_grid_v1",
                    "configuration_grid": str(configuration_path.resolve()),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--configurations", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--segmentation-threshold", type=float, required=True)
    parser.add_argument("--empty-base", action="store_true")
    parser.add_argument("--min-seed-coverage", type=float, default=0.05)
    parser.add_argument("--min-refined-overlap", type=float, default=0.05)
    parser.add_argument("--min-volume-ratio", type=float, default=0.25)
    parser.add_argument("--max-volume-ratio", type=float, default=4.0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--writer-workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    materialize_grid(
        args.work,
        args.base,
        args.output_root,
        args.configurations,
        args.fold,
        args.segmentation_threshold,
        args.empty_base,
        args.min_seed_coverage,
        args.min_refined_overlap,
        args.min_volume_ratio,
        args.max_volume_ratio,
        args.case_limit,
        args.writer_workers,
    )


if __name__ == "__main__":
    main()
