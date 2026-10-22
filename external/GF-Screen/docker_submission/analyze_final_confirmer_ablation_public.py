#!/usr/bin/env python3
"""Measure the unique public value of the expensive final confirmers.

This is a frozen source ablation, not a threshold sweep.  It reconstructs the
exact mask immediately before the final FreeTumor/STU-Net OR gate and compares
four predeclared deployments: no final confirmation, STU only, FreeTumor only,
and the historical OR.  Only public images/labels and immutable prediction
caches are read; hidden data, submission creation, and upload are impossible.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


DOCKER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DOCKER_ROOT.parents[2]
GF_ROOT = DOCKER_ROOT.parent
for source in (DOCKER_ROOT, GF_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from fast_official_lesion_metrics import score_case  # noqa: E402
from winner_mask_recipe import (  # noqa: E402
    add_guided_components,
    confirm_whole_components_26,
    filter_components_physical_6,
    union_touching_support_components_26,
)


VARIANTS = ("pre_final", "stunet_only", "freetumor_only", "historical_or")


def load_binary(path: Path, reference: nib.Nifti1Image | None = None) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    if reference is not None and (
        image.shape != reference.shape
        or not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-3)
    ):
        raise ValueError(f"geometry mismatch: {path}")
    array = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(array), (0, 1)).all():
        raise ValueError(f"nonbinary mask: {path}")
    return array.astype(bool, copy=False), image


def reconstruct_pre_final(name: str, paths: dict[str, str]) -> tuple[np.ndarray, nib.Nifti1Image]:
    directories = {key: Path(value) for key, value in paths.items()}
    bbox_raw, reference = load_binary(directories["bbox_raw"] / name)
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    bbox_filtered = filter_components_physical_6(bbox_raw, spacing)
    prompt_whole, _ = load_binary(directories["prompt_whole"] / name, reference)
    prompt_tumor, _ = load_binary(directories["prompt_tumor"] / name, reference)
    prompt_support = prompt_whole | prompt_tumor
    prompt_confirmed = confirm_whole_components_26(bbox_filtered, prompt_support)
    guided_mask, _ = load_binary(directories["guided"] / name, reference)
    guided = add_guided_components(prompt_confirmed, guided_mask, prompt_support, spacing)
    cbg, _ = load_binary(directories["cbg"] / name, reference)
    cbg_confirmed = confirm_whole_components_26(guided, cbg)
    return union_touching_support_components_26(cbg_confirmed, cbg), reference


def case_ablation(item: tuple[str, dict[str, str]]) -> dict[str, object]:
    name, paths = item
    directories = {key: Path(value) for key, value in paths.items()}
    anchor, reference = reconstruct_pre_final(name, paths)
    freetumor, _ = load_binary(directories["freetumor"] / name, reference)
    probability_path = directories["stunet"] / f"{name.removesuffix('.nii.gz')}.prob16.npz"
    with np.load(probability_path) as archive:
        probability = archive["lesion_probability_uint16"]
    if probability.dtype != np.uint16 or probability.shape != anchor.shape:
        raise ValueError(f"invalid STU probability: {probability_path}")
    stunet = probability.astype(np.float32) / 65535.0 > 0.70
    target, _ = load_binary(directories["labels"] / name, reference)
    supports = {
        "pre_final": np.ones(anchor.shape, dtype=bool),
        "stunet_only": stunet,
        "freetumor_only": freetumor,
        "historical_or": stunet | freetumor,
    }
    variants: dict[str, object] = {}
    for variant, support in supports.items():
        prediction = anchor if variant == "pre_final" else confirm_whole_components_26(anchor, support)
        dsc, nsd = score_case(
            target.astype(np.uint8, copy=False),
            prediction.astype(np.uint8, copy=False),
            reference.header.get_zooms()[:3],
        )
        variants[variant] = {
            "Lesion_DSC": float(dsc),
            "Lesion_NSD": float(nsd),
            "foreground_voxels": int(prediction.sum()),
        }
    return {
        "case": name,
        "anchor_voxels": int(anchor.sum()),
        "stunet_support_voxels": int(np.count_nonzero(anchor & stunet)),
        "freetumor_support_voxels": int(np.count_nonzero(anchor & freetumor)),
        "variants": variants,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCKER_ROOT / "analysis/final_confirmer_ablation_public_20260827.json",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1, 4]")

    paths = {
        "bbox_raw": PROJECT_ROOT / "external/GF-Screen/predictions/gf_a055_bbox_pu_best_ep280/public_raw",
        "prompt_whole": PROJECT_ROOT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti",
        "prompt_tumor": PROJECT_ROOT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti",
        "guided": PROJECT_ROOT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500",
        "cbg": PROJECT_ROOT / "external/GF-Screen/public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/deployed_mm1000_new2k_s25",
        "freetumor": PROJECT_ROOT / "external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions",
        "stunet": PROJECT_ROOT / "external/STU-Net/pancancer/public_sweeps/stunetb_matched_e160_20260826/component_predictions",
        "labels": PROJECT_ROOT / "validation/Validation-Public-Labels",
    }
    names = sorted(path.name for path in paths["labels"].glob("*.nii.gz"))
    if len(names) != 50:
        raise RuntimeError(f"fixed-public inventory must contain 50 cases, found {len(names)}")
    expected = set(names)
    for role, directory in paths.items():
        nifti_names = {path.name for path in directory.glob("*.nii.gz")}
        if role == "stunet":
            probability_names = {
                f"{path.name.removesuffix('.prob16.npz')}.nii.gz"
                for path in directory.glob("*.prob16.npz")
            }
            if nifti_names != expected or probability_names != expected:
                raise RuntimeError(f"{role} inventory mismatch: {directory}")
        elif nifti_names != expected:
            raise RuntimeError(f"{role} inventory mismatch: {directory}")

    rendered_paths = {key: str(value.resolve()) for key, value in paths.items()}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cases = list(executor.map(case_ablation, ((name, rendered_paths) for name in names)))
    aggregates = {}
    for variant in VARIANTS:
        dsc = np.asarray([case["variants"][variant]["Lesion_DSC"] for case in cases])
        nsd = np.asarray([case["variants"][variant]["Lesion_NSD"] for case in cases])
        aggregates[variant] = {
            "Lesion_DSC": float(dsc.mean()),
            "Lesion_NSD": float(nsd.mean()),
            "zero_DSC_cases": int(np.count_nonzero(dsc == 0)),
        }
    historical = aggregates["historical_or"]
    for variant in VARIANTS:
        aggregates[variant]["delta_DSC_vs_historical_or"] = float(
            aggregates[variant]["Lesion_DSC"] - historical["Lesion_DSC"]
        )
        aggregates[variant]["delta_NSD_vs_historical_or"] = float(
            aggregates[variant]["Lesion_NSD"] - historical["Lesion_NSD"]
        )
    if not np.isclose(historical["Lesion_DSC"], 0.6216639257651584, rtol=0.0, atol=5e-8):
        raise RuntimeError("historical OR did not reproduce the immutable public DSC")

    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision_scope": "FROZEN_FINAL_CONFIRMER_SOURCE_ABLATION_ONLY",
        "variants": list(VARIANTS),
        "aggregates": aggregates,
        "cases": cases,
        "checkpoint_sha256": {
            "freetumor": sha256(PROJECT_ROOT / "external/FreeTumor/pretrained_models/baseline/model_baseline_segmentor.pt"),
            "stunet": sha256(PROJECT_ROOT / "nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_inference_epoch_0160.pth"),
        },
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
