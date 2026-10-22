#!/usr/bin/env python3
"""Score only the frozen ResEnc S6 deployments on fixed public data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GF_ROOT = PROJECT_ROOT / "external" / "GF-Screen"
PANCANCER_ROOT = PROJECT_ROOT / "external" / "STU-Net" / "pancancer"
sys.path.insert(0, str(GF_ROOT))
sys.path.insert(0, str(PANCANCER_ROOT))
from fast_official_lesion_metrics import (  # noqa: E402
    prepare_surface_reference,
    score_case_with_reference,
)
from analyze_stunet_gf_ensemble import load_compact_probability  # noqa: E402
from nnunet.materialize_resenc_s6_deployment import (  # noqa: E402
    DEPLOYMENTS,
    deployment_mask,
    load_binary,
)
from nnunet.select_resenc_s6_public import load_policy, sha256_file  # noqa: E402


def validate_public_labels(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    if "validation-public-labels" not in parts:
        raise ValueError("fixed-public evaluator requires Validation-Public-Labels")
    if "validation-hidden-labels" in parts:
        raise ValueError("fixed-public evaluator forbids hidden labels")


def source_signature(paths: dict[str, Path], policy_sha256: str) -> dict:
    result = {"policy_sha256": policy_sha256}
    for role, path in paths.items():
        stat = path.stat()
        result[role] = {
            "path": str(path.resolve()),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return result


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def score_case(item: tuple[str, str, str, str, str, str]) -> dict:
    name, labels_dir, anchor_dir, resenc_dir, freetumor_dir, policy_sha = item
    paths = {
        "label": Path(labels_dir) / name,
        "anchor": Path(anchor_dir) / name,
        "resenc": Path(resenc_dir) / name,
        "freetumor": Path(freetumor_dir) / name,
        "probability": (
            Path(resenc_dir) / f"{name.removesuffix('.nii.gz')}.prob16.npz"
        ),
    }
    label_image, truth = load_binary(paths["label"])
    anchor_image, anchor = load_binary(paths["anchor"], label_image)
    _, resenc = load_binary(paths["resenc"], label_image)
    _, freetumor = load_binary(paths["freetumor"], label_image)
    probability = load_compact_probability(paths["probability"], label_image.shape)
    spacing = tuple(float(value) for value in label_image.header.get_zooms()[:3])
    voxel_mm3 = float(abs(np.linalg.det(anchor_image.affine[:3, :3])))
    reference = prepare_surface_reference(truth, spacing) if truth.any() else None
    metrics = {}
    for deployment in DEPLOYMENTS:
        mask = deployment_mask(
            deployment,
            anchor,
            resenc,
            freetumor,
            probability,
            voxel_mm3,
            spacing,
        )
        dsc, nsd = score_case_with_reference(truth, mask, spacing, reference)
        metrics[deployment] = [float(dsc), float(nsd)]
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "case": name,
        "source_signature": source_signature(paths, policy_sha),
        "metrics": metrics,
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def cache_valid(path: Path, signature: dict) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "COMPLETE"
            or payload.get("source_signature") != signature
            or set(payload.get("metrics", {})) != set(DEPLOYMENTS)
            or payload.get("public_labels_read") is not True
            or payload.get("hidden_read")
            or payload.get("submission_created")
            or payload.get("uploaded")
        ):
            return None
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def evaluate(
    policy_path: Path,
    labels: Path,
    anchor: Path,
    resenc: Path,
    freetumor: Path,
    cache: Path,
    workers: int,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    validate_public_labels(labels)
    policy = load_policy(policy_path)
    policy_sha = sha256_file(policy_path)
    names = sorted(path.name for path in labels.glob("*.nii.gz"))
    if len(names) != 50:
        raise ValueError(f"fixed-public labels must contain 50 cases, found {len(names)}")
    expected = set(names)
    for role, directory in (
        ("anchor", anchor),
        ("resenc", resenc),
        ("freetumor", freetumor),
    ):
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected:
            raise ValueError(f"{role} fixed-public inventory mismatch")
    probability_names = {
        path.name for path in resenc.glob("*.prob16.npz")
    }
    if probability_names != {
        f"{name.removesuffix('.nii.gz')}.prob16.npz" for name in names
    }:
        raise ValueError("ResEnc fixed-public probability inventory mismatch")
    cache.mkdir(parents=True, exist_ok=True)
    completed, pending = {}, []
    for name in names:
        paths = {
            "label": labels / name,
            "anchor": anchor / name,
            "resenc": resenc / name,
            "freetumor": freetumor / name,
            "probability": resenc / f"{name.removesuffix('.nii.gz')}.prob16.npz",
        }
        signature = source_signature(paths, policy_sha)
        cache_path = cache / f"{name.removesuffix('.nii.gz')}.metrics.json"
        payload = cache_valid(cache_path, signature)
        if payload is None:
            pending.append(
                (
                    name,
                    str(labels),
                    str(anchor),
                    str(resenc),
                    str(freetumor),
                    policy_sha,
                )
            )
        else:
            completed[name] = payload
    def save(payload: dict) -> None:
        name = payload["case"]
        atomic_json(
            cache / f"{name.removesuffix('.nii.gz')}.metrics.json", payload
        )
        completed[name] = payload
        print(f"S6 fixed-public metric progress {len(completed)}/50: {name}", flush=True)
    if workers == 1:
        for payload in map(score_case, pending):
            save(payload)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(score_case, item) for item in pending]
            for future in as_completed(futures):
                save(future.result())
    if set(completed) != expected:
        raise RuntimeError("incomplete S6 fixed-public case metrics")
    metrics = {}
    for deployment in DEPLOYMENTS:
        values = np.asarray(
            [completed[name]["metrics"][deployment] for name in names], dtype=float
        )
        metrics[deployment] = {
            "Lesion_DSC": float(values[:, 0].mean()),
            "Lesion_NSD": float(values[:, 1].mean()),
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "case_count": 50,
        "policy": {
            "path": str(policy_path.resolve()),
            "sha256": policy_sha,
        },
        "evaluated_deployments": list(DEPLOYMENTS),
        "metrics": metrics,
        "case_cache": str(cache.resolve()),
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--resenc", required=True, type=Path)
    parser.add_argument("--freetumor", required=True, type=Path)
    parser.add_argument("--case-cache", required=True, type=Path)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = evaluate(
        args.policy,
        args.labels,
        args.anchor,
        args.resenc,
        args.freetumor,
        args.case_cache,
        args.workers,
    )
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
