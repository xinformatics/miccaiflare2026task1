#!/usr/bin/env python3
"""Audit trusted-region context on deterministic Dataset501 training patches.

This reads only challenge training segmentations. It does not inspect public or
hidden data and does not mutate any training target. Four foreground-centered
patches per case reproduce the production sampler mixture deterministically:
three use stored foreground locations and one uses a uniformly selected lesion
component and voxel.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import blosc2
import numpy as np
from scipy import ndimage


PATCH_SHAPE = (128, 160, 160)
SPACING_MM = (2.4, 1.6, 1.6)
POLICIES = {
    "component_box_8mm": ("box", tuple(int(np.ceil(8 / x)) for x in SPACING_MM)),
    "shape_linf_band_12mm": (
        "band",
        tuple(int(np.ceil(12 / x)) for x in SPACING_MM),
    ),
    "component_box_16mm": (
        "box",
        tuple(int(np.ceil(16 / x)) for x in SPACING_MM),
    ),
}
QUANTILES = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)


def source_from_identifier(identifier: str) -> str:
    if "_" not in identifier:
        raise ValueError(f"identifier has no source separator: {identifier}")
    return identifier.rsplit("_", 1)[0]


def select_source_cases(
    folder: Path, per_source: int = 10, excluded: set[str] | None = None
) -> tuple[list[str], dict[str, int]]:
    excluded = excluded or set()
    identifiers = sorted(
        identifier
        for path in folder.glob("*_seg.b2nd")
        if (identifier := path.name[: -len("_seg.b2nd")]) not in excluded
    )
    grouped: dict[str, list[str]] = defaultdict(list)
    for identifier in identifiers:
        grouped[source_from_identifier(identifier)].append(identifier)
    selected: list[str] = []
    counts: dict[str, int] = {}
    for source in sorted(grouped):
        cases = grouped[source]
        count = min(int(per_source), len(cases))
        indices = np.linspace(0, len(cases) - 1, count, dtype=int)
        chosen = [cases[int(index)] for index in indices]
        if len(set(chosen)) != count:
            raise RuntimeError(f"source sampling duplicated an identifier: {source}")
        selected.extend(chosen)
        counts[source] = len(cases)
    return selected, counts


def inventory_sha256(identifiers: list[str]) -> str:
    return hashlib.sha256(("\n".join(identifiers) + "\n").encode()).hexdigest()


def crop_centered(segmentation: np.ndarray, center: np.ndarray) -> np.ndarray:
    if segmentation.ndim != 3 or len(center) != 3:
        raise ValueError("expected a 3-D segmentation and center")
    lower = np.asarray(center, dtype=np.int64) - np.asarray(PATCH_SHAPE) // 2
    need_to_pad = np.maximum(0, np.asarray(PATCH_SHAPE) - np.asarray(segmentation.shape))
    minimum_lower = -(need_to_pad // 2)
    lower = np.maximum(lower, minimum_lower)
    upper = lower + np.asarray(PATCH_SHAPE)
    output = np.full(PATCH_SHAPE, -1, dtype=segmentation.dtype)
    source_slices = tuple(
        slice(max(0, int(start)), min(size, int(stop)))
        for start, stop, size in zip(lower, upper, segmentation.shape, strict=True)
    )
    destination_slices = tuple(
        slice(max(0, -int(start)), max(0, -int(start)) + (source.stop - source.start))
        for start, source in zip(lower, source_slices, strict=True)
    )
    output[destination_slices] = segmentation[source_slices]
    return output


def component_uniform_center(foreground: np.ndarray, seed: int) -> np.ndarray:
    coordinates = np.where(foreground)
    if coordinates[0].size == 0:
        raise ValueError("foreground-empty case")
    lower = np.asarray([int(axis.min()) for axis in coordinates], dtype=np.int64)
    region = tuple(
        slice(int(start), int(axis.max()) + 1)
        for start, axis in zip(lower, coordinates, strict=True)
    )
    labeled, count = ndimage.label(
        foreground[region], structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    rng = np.random.RandomState(seed)
    selected_component = int(rng.randint(1, count + 1))
    component_coordinates = np.argwhere(labeled == selected_component)
    selected = component_coordinates[int(rng.randint(0, len(component_coordinates)))]
    return selected.astype(np.int64) + lower


def deterministic_centers(
    identifier: str, segmentation: np.ndarray, properties: dict
) -> list[tuple[str, np.ndarray]]:
    class_locations = properties.get("class_locations", {})
    locations = None
    for key, value in class_locations.items():
        if str(key) in {"1", "(1,)"}:
            locations = np.asarray(value)
            break
    if locations is None or len(locations) == 0:
        raise ValueError(f"missing stored foreground locations: {identifier}")
    indices = np.linspace(0, len(locations) - 1, 3, dtype=int)
    centers = [("foreground_voxel", locations[int(index), 1:].astype(np.int64)) for index in indices]
    seed = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:4], "little")
    centers.append(
        ("component_uniform", component_uniform_center(segmentation == 1, seed))
    )
    return centers


def trusted_mask(foreground: np.ndarray, policy: str, margins: tuple[int, int, int]) -> np.ndarray:
    trusted = np.zeros_like(foreground, dtype=bool)
    coordinates = np.where(foreground)
    if coordinates[0].size == 0:
        return trusted
    if policy == "box":
        lower = [int(axis.min()) for axis in coordinates]
        region = tuple(
            slice(start, int(axis.max()) + 1)
            for start, axis in zip(lower, coordinates, strict=True)
        )
        labeled, count = ndimage.label(
            foreground[region], structure=np.ones((3, 3, 3), dtype=np.uint8)
        )
        for component_slice in ndimage.find_objects(labeled, count):
            if component_slice is None:
                continue
            expanded = tuple(
                slice(
                    max(0, offset + int(axis.start) - margin),
                    min(size, offset + int(axis.stop) + margin),
                )
                for offset, axis, margin, size in zip(
                    lower, component_slice, margins, foreground.shape, strict=True
                )
            )
            trusted[expanded] = True
        return trusted
    if policy != "band":
        raise ValueError(f"unknown policy: {policy}")
    lower = [max(0, int(axis.min()) - margin) for axis, margin in zip(coordinates, margins, strict=True)]
    upper = [
        min(size, int(axis.max()) + margin + 1)
        for axis, margin, size in zip(coordinates, margins, foreground.shape, strict=True)
    ]
    region = tuple(slice(start, stop) for start, stop in zip(lower, upper, strict=True))
    trusted[region] = ndimage.maximum_filter(
        foreground[region],
        size=tuple(2 * margin + 1 for margin in margins),
        mode="constant",
        cval=0,
    )
    return trusted


def patch_policy_metrics(segmentation: np.ndarray, policy: str, margins: tuple[int, int, int]) -> dict[str, float | int]:
    foreground = segmentation == 1
    trusted = trusted_mask(foreground, policy, margins)
    foreground_voxels = int(foreground.sum())
    retained_foreground = int((trusted & foreground).sum())
    trusted_background = int((trusted & (segmentation == 0)).sum())
    trusted_unknown = int((trusted & ~((segmentation == 0) | foreground)).sum())
    return {
        "foreground_voxels": foreground_voxels,
        "retained_foreground_voxels": retained_foreground,
        "trusted_background_voxels": trusted_background,
        "trusted_unknown_voxels": trusted_unknown,
        "background_to_foreground_ratio": trusted_background / max(1, foreground_voxels),
    }


def quantile_dict(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.quantile(array, q)) for q in QUANTILES}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(folder: Path, output: Path, per_source: int, inventory_audit: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(inventory_audit.read_text(encoding="utf-8"))
    if (
        inventory.get("status") != "PASS"
        or inventory.get("dataset_case_count") != 17575
        or inventory.get("positive_case_count") != 17536
        or inventory.get("empty_foreground_case_count") != 39
        or inventory.get("public_read")
        or inventory.get("hidden_read")
    ):
        raise ValueError("Dataset501 positive/empty inventory audit is invalid")
    empty_cases = set(inventory["empty_foreground_cases"])
    selected, source_inventory = select_source_cases(folder, per_source, empty_cases)
    patch_rows: list[dict] = []
    for case_index, identifier in enumerate(selected, start=1):
        segmentation = np.asarray(
            blosc2.open(
                urlpath=str(folder / f"{identifier}_seg.b2nd"),
                mode="r",
                dparams={"nthreads": 1},
                mmap_mode="r",
            )
        )
        if segmentation.ndim != 4 or segmentation.shape[0] != 1:
            raise ValueError(f"unexpected segmentation shape: {identifier} {segmentation.shape}")
        segmentation = segmentation[0]
        with (folder / f"{identifier}.pkl").open("rb") as handle:
            properties = pickle.load(handle)
        for sample_index, (center_kind, center) in enumerate(
            deterministic_centers(identifier, segmentation, properties)
        ):
            patch = crop_centered(segmentation, center)
            row = {
                "identifier": identifier,
                "source": source_from_identifier(identifier),
                "sample_index": sample_index,
                "center_kind": center_kind,
                "center_zyx": [int(x) for x in center],
                "policies": {},
            }
            for name, (policy, margins) in POLICIES.items():
                row["policies"][name] = patch_policy_metrics(patch, policy, margins)
            patch_rows.append(row)
        print(f"[{case_index:03d}/{len(selected):03d}] {identifier}", flush=True)

    policy_summary = {}
    for name in POLICIES:
        rows = [row["policies"][name] for row in patch_rows]
        ratios = [float(row["background_to_foreground_ratio"]) for row in rows]
        policy_summary[name] = {
            "margin_voxels_zyx": list(POLICIES[name][1]),
            "background_to_foreground_ratio_quantiles": quantile_dict(ratios),
            "pooled_background_to_foreground_ratio": sum(int(row["trusted_background_voxels"]) for row in rows)
            / max(1, sum(int(row["foreground_voxels"]) for row in rows)),
            "foreground_voxels": sum(int(row["foreground_voxels"]) for row in rows),
            "retained_foreground_voxels": sum(int(row["retained_foreground_voxels"]) for row in rows),
            "trusted_background_voxels": sum(int(row["trusted_background_voxels"]) for row in rows),
            "trusted_unknown_voxels": sum(int(row["trusted_unknown_voxels"]) for row in rows),
        }
    band = policy_summary["shape_linf_band_12mm"]
    active = policy_summary["component_box_16mm"]
    median_ratio = band["background_to_foreground_ratio_quantiles"]["0.5"]
    foreground_retained = band["foreground_voxels"] == band["retained_foreground_voxels"]
    status = "PASS_CONTEXT_ADEQUATE" if median_ratio >= 5.0 and foreground_retained else "REVIEW_CONTEXT_INADEQUATE"
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": status,
        "decision": "RETAIN_12MM_SHAPE_BAND_CANDIDATE" if status.startswith("PASS") else "DO_NOT_ARM_12MM_SHAPE_BAND",
        "question": "Does the proposed 12-mm trusted shape band retain adequate local negative context on real Dataset501 foreground-centered patches?",
        "selection": {
            "policy": "up to 10 floor-linspace identifiers per filename-derived source, sorted before inspection",
            "per_source": per_source,
            "dataset_cases": int(inventory["dataset_case_count"]),
            "positive_dataset_cases": sum(source_inventory.values()),
            "excluded_empty_cases": len(empty_cases),
            "empty_case_fraction": len(empty_cases) / int(inventory["dataset_case_count"]),
            "uniform_equivalent_empty_draws_per_500_case_epoch": 500
            * len(empty_cases)
            / int(inventory["dataset_case_count"]),
            "positive_inventory_audit": {
                "path": str(inventory_audit.resolve()),
                "sha256": hashlib.sha256(inventory_audit.read_bytes()).hexdigest(),
            },
            "sources": len(source_inventory),
            "selected_cases": len(selected),
            "selected_inventory_sha256": inventory_sha256(selected),
            "patches_per_case": 4,
            "patches": len(patch_rows),
            "sampler_mix": "three stored foreground-voxel centers plus one uniformly selected component/voxel center",
            "patch_shape_zyx": list(PATCH_SHAPE),
            "spacing_mm_zyx": list(SPACING_MM),
            "source_case_counts": source_inventory,
        },
        "predeclared_gate": {
            "minimum_median_background_to_foreground_ratio": 5.0,
            "requires_all_annotated_foreground_retained": True,
        },
        "policies": policy_summary,
        "comparison": {
            "band12_trusted_background_fraction_of_active_box16": band["trusted_background_voxels"]
            / max(1, active["trusted_background_voxels"]),
            "band12_reduction_in_trusted_background_vs_active_box16": 1.0
            - band["trusted_background_voxels"] / max(1, active["trusted_background_voxels"]),
        },
        "limitations": [
            "Training masks identify reliable annotated positives but cannot reveal omitted lesions, so contamination is assessed separately on complete public masks.",
            "The deterministic patches reproduce foreground-center geometry before stochastic spatial augmentation; they do not replay the live random augmentation stream.",
            "This audit is a supervision-sufficiency check, not evidence of segmentation performance.",
            "The 39 foreground-empty cases are excluded because no foreground-centered patch exists; they are 0.222 percent of Dataset501 and become zero-gradient samples under both partial-region policies.",
        ],
        "public_labels_read": False,
        "hidden_read": False,
        "training_mutated": False,
        "submission_created": False,
        "uploaded": False,
    }
    rows_path = output.with_name("patch_metrics.jsonl")
    rows_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in patch_rows), encoding="utf-8")
    payload["patch_metrics"] = {
        "path": str(rows_path.resolve()),
        "sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-source", type=int, default=10)
    parser.add_argument("--positive-inventory", required=True, type=Path)
    args = parser.parse_args()
    payload = run(args.folder, args.output, args.per_source, args.positive_inventory)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
