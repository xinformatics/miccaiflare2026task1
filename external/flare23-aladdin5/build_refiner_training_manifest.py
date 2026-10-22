#!/usr/bin/env python3
"""Build a label-contract-safe FLARE23 component-refiner mining manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def same_grid(first: nib.Nifti1Image, second: nib.Nifti1Image) -> bool:
    return first.shape == second.shape and np.allclose(
        first.affine, second.affine, rtol=0, atol=1e-4
    )


def qualifying_components(
    proposal_path: Path, label_path: Path | None, minimum_mm3: float
) -> tuple[int, int, int]:
    proposal_image = nib.load(str(proposal_path))
    proposal = np.asanyarray(proposal_image.dataobj) > 0
    labels, count = cc3d.connected_components(
        proposal.astype(np.uint8), connectivity=26, return_N=True
    )
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    voxel_mm3 = float(abs(np.linalg.det(proposal_image.affine[:3, :3])))
    eligible = [
        component_id
        for component_id in range(1, count + 1)
        if sizes[component_id] * voxel_mm3 >= minimum_mm3
    ]
    if label_path is None:
        return len(eligible), 0, int(np.count_nonzero(proposal))
    label_image = nib.load(str(label_path))
    if not same_grid(proposal_image, label_image):
        raise RuntimeError(f"proposal/label geometry mismatch: {proposal_path}")
    target = np.asanyarray(label_image.dataobj) > 0
    positive = sum(bool(np.any(target[labels == component_id])) for component_id in eligible)
    target_components = int(
        cc3d.connected_components(target.astype(np.uint8), connectivity=26).max()
    )
    return len(eligible), positive, target_components


def stage_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = source.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve() != resolved:
            raise RuntimeError(f"stale staged proposal: {destination}")
    elif destination.exists():
        raise RuntimeError(f"non-symlink staged proposal: {destination}")
    else:
        destination.symlink_to(resolved)


def pseudo_case_name(image_path: Path) -> str:
    name = image_path.name
    if name.endswith("_0000.nii.gz"):
        return name.removesuffix("_0000.nii.gz")
    return name.removesuffix(".nii.gz")


def _qualifying_task(task: tuple[Path, Path | None, float]) -> tuple[int, int, int]:
    return qualifying_components(*task)


def evaluate_components(
    tasks: list[tuple[Path, Path | None, float]], workers: int
) -> list[tuple[int, int, int]]:
    """Evaluate masks concurrently while preserving deterministic input order."""
    if workers == 1:
        return [qualifying_components(*task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_qualifying_task, tasks, chunksize=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-manifest", required=True, type=Path)
    parser.add_argument("--exact-proposals", required=True, type=Path)
    parser.add_argument("--pseudo-manifest", required=True, type=Path)
    parser.add_argument("--pseudo-proposals", required=True, type=Path)
    parser.add_argument("--pseudo-validation", required=True, type=Path)
    parser.add_argument("--healthy-images", required=True, type=Path)
    parser.add_argument("--healthy-proposals", required=True, type=Path)
    parser.add_argument("--healthy-validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-mm3", default=1000.0, type=float)
    parser.add_argument("--workers", default=1, type=int)
    args = parser.parse_args()
    if args.minimum_mm3 <= 0:
        raise ValueError("minimum component volume must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    for validation_path, expected in (
        (args.pseudo_validation, 2856),
        (args.healthy_validation, 172),
    ):
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE" or payload.get("cases") != expected:
            raise RuntimeError(f"incomplete proposal validation: {validation_path}")
        if payload.get("public_read") or payload.get("hidden_read"):
            raise RuntimeError(f"invalid proposal provenance: {validation_path}")

    args.output.mkdir(parents=True, exist_ok=True)
    proposal_stage = args.output / "proposals"
    empty_labels = args.output / "healthy_empty_labels"
    proposal_stage.mkdir(exist_ok=True)
    empty_labels.mkdir(exist_ok=True)
    items = []
    counts = {
        "exact_positive_cases": 0,
        "pseudo_consensus_cases": 0,
        "pseudo_multilesion_cases": 0,
        "healthy_negative_cases": 0,
    }

    exact = json.loads(args.exact_manifest.read_text(encoding="utf-8"))
    if exact.get("cases") != 707 or len(exact.get("items", [])) != 707:
        raise RuntimeError("exact manifest is not the frozen 707-case inventory")
    exact_items = exact["items"]
    exact_tasks = [
        (
            args.exact_proposals / str(source_item["prediction_name"]),
            Path(str(source_item["label"])),
            args.minimum_mm3,
        )
        for source_item in exact_items
    ]
    exact_results = evaluate_components(exact_tasks, args.workers)
    for source_item, (_, positives, _) in zip(exact_items, exact_results, strict=True):
        case = str(source_item["name"])
        source = args.exact_proposals / str(source_item["prediction_name"])
        if positives == 0:
            continue
        name = f"L707__{case}"
        prediction_name = name + ".nii.gz"
        stage_link(source, proposal_stage / prediction_name)
        items.append(
            {
                "name": name,
                "image": str(Path(str(source_item["image"])).resolve()),
                "label": str(Path(str(source_item["label"])).resolve()),
                "cohort": str(source_item["cohort"]),
                "prediction_name": prediction_name,
                "complete_label": False,
                "positive_only_label": True,
                "training_only": False,
                "label_contract": "partial_primary_positive_only",
                "source_case": case,
            }
        )
        counts["exact_positive_cases"] += 1

    pseudo = json.loads(args.pseudo_manifest.read_text(encoding="utf-8"))
    pseudo_items = list(pseudo.get("training", [])) + list(pseudo.get("validation", []))
    if len(pseudo_items) != 2311:
        raise RuntimeError(f"expected 2311 GF pseudo-positive cases, found {len(pseudo_items)}")
    pseudo_tasks = []
    for source_item in pseudo_items:
        image = Path(str(source_item["image"]))
        case = pseudo_case_name(image)
        source = args.pseudo_proposals / f"{case}.nii.gz"
        label = Path(str(source_item["label"]))
        pseudo_tasks.append((source, label, args.minimum_mm3))
    pseudo_results = evaluate_components(pseudo_tasks, args.workers)
    for source_item, (_, positives, target_components) in zip(
        pseudo_items, pseudo_results, strict=True
    ):
        image = Path(str(source_item["image"]))
        case = pseudo_case_name(image)
        source = args.pseudo_proposals / f"{case}.nii.gz"
        label = Path(str(source_item["label"]))
        if positives == 0:
            continue
        name = f"PU__{case}"
        prediction_name = name + ".nii.gz"
        stage_link(source, proposal_stage / prediction_name)
        family = "AMOS" if case.startswith("amos_") else "MSD"
        items.append(
            {
                "name": name,
                "image": str(image.resolve()),
                "label": str(label.resolve()),
                "cohort": f"pseudo_consensus:{family}",
                "prediction_name": prediction_name,
                "complete_label": False,
                "positive_only_label": True,
                "training_only": True,
                "label_contract": "gf_flare23_consensus_positive_only",
                "source_case": case,
            }
        )
        counts["pseudo_consensus_cases"] += 1
        if target_components > 1:
            counts["pseudo_multilesion_cases"] += 1

    healthy_sources = sorted(args.healthy_images.glob("*_0000.nii.gz"))
    if len(healthy_sources) != 172:
        raise RuntimeError(f"expected 172 staged healthy images, found {len(healthy_sources)}")
    healthy_tasks = [
        (
            args.healthy_proposals
            / f"{image.name.removesuffix('_0000.nii.gz')}.nii.gz",
            None,
            args.minimum_mm3,
        )
        for image in healthy_sources
    ]
    healthy_results = evaluate_components(healthy_tasks, args.workers)
    for image, (eligible, _, _) in zip(
        healthy_sources, healthy_results, strict=True
    ):
        case = image.name.removesuffix("_0000.nii.gz")
        source = args.healthy_proposals / f"{case}.nii.gz"
        if eligible == 0:
            continue
        name = f"HN__{case}"
        prediction_name = name + ".nii.gz"
        stage_link(source, proposal_stage / prediction_name)
        image_nii = nib.load(str(image))
        empty_path = empty_labels / prediction_name
        if not empty_path.is_file():
            header = image_nii.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(
                nib.Nifti1Image(
                    np.zeros(image_nii.shape, dtype=np.uint8), image_nii.affine, header
                ),
                str(empty_path),
            )
        items.append(
            {
                "name": name,
                "image": str(image.resolve()),
                "label": str(empty_path.resolve()),
                "cohort": "healthy_complete_negative",
                "prediction_name": prediction_name,
                "complete_label": True,
                "positive_only_label": False,
                "training_only": True,
                "label_contract": "healthy_complete_negative",
                "source_case": case,
            }
        )
        counts["healthy_negative_cases"] += 1

    if not counts["exact_positive_cases"] or not counts["pseudo_consensus_cases"]:
        raise RuntimeError("no positive training cases survived proposal qualification")
    if not counts["healthy_negative_cases"]:
        raise RuntimeError("no healthy FLARE23 hard negatives survived qualification")
    if len({item["name"] for item in items}) != len(items):
        raise RuntimeError("combined manifest case identities are not unique")
    if len(list(proposal_stage.glob("*.nii.gz"))) != len(items):
        raise RuntimeError("staged proposal inventory mismatch")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "description": (
            "FLARE23 component proposals with partial-label positives only, "
            "GF-FLARE23 Train-Unlabel consensus positives, and complete healthy negatives"
        ),
        "minimum_mm3": args.minimum_mm3,
        "cases": len(items),
        "counts": counts,
        "items": items,
        "assets": {
            "exact_manifest_sha256": sha256(args.exact_manifest),
            "pseudo_manifest_sha256": sha256(args.pseudo_manifest),
            "pseudo_validation_sha256": sha256(args.pseudo_validation),
            "healthy_validation_sha256": sha256(args.healthy_validation),
        },
        "constraints": {
            "unmatched_partial_proposals_are_negative": False,
            "pseudo_and_healthy_validation_selection": False,
            "public_read": False,
            "hidden_read": False,
        },
    }
    manifest_path = args.output / "manifest.json"
    atomic_json(manifest_path, manifest)
    terminal = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "cases": len(items),
        "counts": counts,
        "proposal_members": len(list(proposal_stage.glob("*.nii.gz"))),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output / "MANIFEST_COMPLETE.json", terminal)
    print(json.dumps(terminal, indent=2))


if __name__ == "__main__":
    main()
