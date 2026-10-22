#!/usr/bin/env python3
"""Score the frozen winner after every historical mask-composition stage."""

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


ROOT = Path(__file__).resolve().parents[3]
GF_ROOT = ROOT / "external/GF-Screen"
DOCKER_ROOT = GF_ROOT / "docker_submission"
for source in (GF_ROOT, DOCKER_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from fast_official_lesion_metrics import score_case  # noqa: E402
from winner_mask_recipe import (  # noqa: E402
    confirm_with_freetumor_or_stunet_p70,
    union_touching_support_components_26,
)


STAGES = (
    "bbox_filtered",
    "prompt_confirmed",
    "guided",
    "cbg_confirmed",
    "touching_cbg",
    "final",
)


def binary(path: Path, reference: nib.Nifti1Image | None = None) -> tuple[np.ndarray, nib.Nifti1Image]:
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


def score_one(item: tuple[str, dict[str, str]]) -> dict[str, object]:
    name, rendered = item
    paths = {key: Path(value) for key, value in rendered.items()}
    target, reference = binary(paths["labels"] / name)
    masks = {}
    for stage in STAGES[:4]:
        masks[stage], _ = binary(paths[stage] / name, reference)
    cbg, _ = binary(paths["cbg"] / name, reference)
    masks["touching_cbg"] = union_touching_support_components_26(
        masks["cbg_confirmed"], cbg
    )
    freetumor, _ = binary(paths["freetumor"] / name, reference)
    probability_path = paths["stunet"] / f"{name.removesuffix('.nii.gz')}.prob16.npz"
    with np.load(probability_path) as archive:
        probability = archive["lesion_probability_uint16"]
    masks["final"] = confirm_with_freetumor_or_stunet_p70(
        masks["touching_cbg"], freetumor, probability
    )
    result = {"case": name, "stages": {}}
    spacing = reference.header.get_zooms()[:3]
    target_u8 = target.astype(np.uint8, copy=False)
    for stage in STAGES:
        prediction = masks[stage]
        dsc, nsd = score_case(
            target_u8, prediction.astype(np.uint8, copy=False), spacing
        )
        result["stages"][stage] = {
            "Lesion_DSC": float(dsc),
            "Lesion_NSD": float(nsd),
            "foreground_voxels": int(prediction.sum()),
        }
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
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
        default=DOCKER_ROOT / "analysis/winner_stage_value_public_20260827.json",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1, 4]")
    paths = {
        "bbox_filtered": GF_ROOT / "predictions/gf_a055_bbox_pu_best_ep280/public_mm1000",
        "prompt_confirmed": ROOT / "external/BiomedParse/public_sweeps/combined_prompt_confirmation_20260820/screen/predictions/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf",
        "guided": GF_ROOT / "public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25",
        "cbg_confirmed": GF_ROOT / "public_sweeps/champion_cbg41_component_confirmation_20260822/predictions/cbg_confirm_distance_mm_0_rescue_min_mm3_inf",
        "cbg": GF_ROOT / "public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/deployed_mm1000_new2k_s25",
        "freetumor": ROOT / "external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions",
        "stunet": ROOT / "external/STU-Net/pancancer/public_sweeps/stunetb_matched_e160_20260826/component_predictions",
        "labels": ROOT / "validation/Validation-Public-Labels",
    }
    names = sorted(path.name for path in paths["labels"].glob("*.nii.gz"))
    expected = set(names)
    if len(expected) != 50:
        raise RuntimeError("fixed-public inventory is not 50")
    for role, directory in paths.items():
        names_here = {path.name for path in directory.glob("*.nii.gz")}
        if names_here != expected:
            raise RuntimeError(f"{role} NIfTI inventory mismatch")
        if role == "stunet":
            probability_names = {
                f"{path.name.removesuffix('.prob16.npz')}.nii.gz"
                for path in directory.glob("*.prob16.npz")
            }
            if probability_names != expected:
                raise RuntimeError("STU probability inventory mismatch")
    rendered = {key: str(value.resolve()) for key, value in paths.items()}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cases = list(executor.map(score_one, ((name, rendered) for name in names)))
    aggregates = {}
    previous = None
    for stage in STAGES:
        dsc = np.asarray([case["stages"][stage]["Lesion_DSC"] for case in cases])
        nsd = np.asarray([case["stages"][stage]["Lesion_NSD"] for case in cases])
        aggregates[stage] = {
            "Lesion_DSC": float(dsc.mean()),
            "Lesion_NSD": float(nsd.mean()),
            "zero_DSC_cases": int(np.count_nonzero(dsc == 0)),
            "delta_DSC_from_previous": None if previous is None else float(dsc.mean() - aggregates[previous]["Lesion_DSC"]),
            "delta_NSD_from_previous": None if previous is None else float(nsd.mean() - aggregates[previous]["Lesion_NSD"]),
        }
        previous = stage
    final = aggregates["final"]
    if not np.isclose(final["Lesion_DSC"], 0.6216639257651584, rtol=0.0, atol=5e-8):
        raise RuntimeError("final stage did not reproduce the immutable winner DSC")
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision_scope": "FROZEN_WINNER_STAGE_VALUE_DIAGNOSTIC",
        "stages": list(STAGES),
        "aggregates": aggregates,
        "cases": cases,
        "source_sha256": {
            "freetumor": sha256(ROOT / "external/FreeTumor/pretrained_models/baseline/model_baseline_segmentor.pt"),
            "stunet": sha256(ROOT / "nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_inference_epoch_0160.pth"),
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
