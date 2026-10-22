#!/usr/bin/env python3
"""Voxel-exact public replay audit for :mod:`winner_mask_recipe`.

The verifier consumes only frozen prediction caches and public images/labels.
It does not run a network, read hidden data, or create a submission.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np

from winner_mask_recipe import (
    add_guided_components,
    confirm_whole_components_26,
    confirm_with_freetumor_or_stunet_p70,
    filter_components_physical_6,
    union_touching_support_components_26,
)


EXPECTED_DSC = 0.6216639257651584


def load_mask(path: Path, reference: nib.Nifti1Image | None = None):
    image = nib.load(str(path))
    if reference is not None and (
        image.shape != reference.shape
        or not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-3)
    ):
        raise ValueError(f"geometry mismatch: {path}")
    values = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(values), (0, 1)).all():
        raise ValueError(f"nonbinary mask: {path}")
    return values.astype(bool, copy=False), image


def require_equal(case: str, stage: str, actual: np.ndarray, expected: np.ndarray):
    changed = int(np.count_nonzero(actual ^ expected))
    if changed:
        raise AssertionError(f"{case}: {stage} differs by {changed} voxels")


def independent_final_reference(
    anchor: np.ndarray, freetumor: np.ndarray, probability: np.ndarray
) -> np.ndarray:
    labels = cc3d.connected_components(anchor.astype(np.uint8), connectivity=26)
    supported = freetumor | (probability.astype(np.float32) / 65535.0 > 0.70)
    keep = np.zeros(int(labels.max()) + 1, dtype=bool)
    keep[np.unique(labels[anchor & supported])] = True
    keep[0] = False
    return keep[labels]


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * int(np.count_nonzero(prediction & target)) / denominator


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    project = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis/winner_recipe_public_equivalence.json",
    )
    args = parser.parse_args()

    paths = {
        "bbox_raw": project / "external/GF-Screen/predictions/gf_a055_bbox_pu_best_ep280/public_raw",
        "bbox_filtered": project / "external/GF-Screen/predictions/gf_a055_bbox_pu_best_ep280/public_mm1000",
        "prompt_whole": project / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti",
        "prompt_tumor": project / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti",
        "prompt_union": project / "external/BiomedParse/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050",
        "prompt_confirmed": project / "external/BiomedParse/public_sweeps/combined_prompt_confirmation_20260820/screen/predictions/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf",
        "guided": project / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500",
        "guided_expected": project / "external/GF-Screen/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25",
        "cbg": project / "external/GF-Screen/public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/deployed_mm1000_new2k_s25",
        "cbg_confirmed": project / "external/GF-Screen/public_sweeps/champion_cbg41_component_confirmation_20260822/predictions/cbg_confirm_distance_mm_0_rescue_min_mm3_inf",
        "freetumor": project / "external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions",
        "stunet": project / "external/STU-Net/pancancer/public_sweeps/stunetb_matched_e160_20260826/component_predictions",
        "labels": project / "validation/Validation-Public-Labels",
    }
    names = sorted(path.name for path in paths["bbox_raw"].glob("*.nii.gz"))
    if len(names) != 50:
        raise RuntimeError(f"expected 50 public cases, found {len(names)}")
    for role, directory in paths.items():
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != set(names):
            raise RuntimeError(f"{role} inventory mismatch: {directory}")

    final_digest = hashlib.sha256()
    case_dice: list[float] = []
    totals = {
        "bbox_raw": 0,
        "bbox_filtered": 0,
        "prompt_confirmed": 0,
        "guided": 0,
        "cbg_confirmed": 0,
        "touching_cbg": 0,
        "final": 0,
    }
    for index, name in enumerate(names, 1):
        bbox_raw, reference = load_mask(paths["bbox_raw"] / name)
        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
        bbox_filtered = filter_components_physical_6(bbox_raw, spacing)
        expected, _ = load_mask(paths["bbox_filtered"] / name, reference)
        require_equal(name, "bbox_filter", bbox_filtered, expected)

        prompt_whole, _ = load_mask(paths["prompt_whole"] / name, reference)
        prompt_tumor, _ = load_mask(paths["prompt_tumor"] / name, reference)
        prompt_union = prompt_whole | prompt_tumor
        expected, _ = load_mask(paths["prompt_union"] / name, reference)
        require_equal(name, "prompt_union", prompt_union, expected)
        prompt_confirmed = confirm_whole_components_26(bbox_filtered, prompt_union)
        expected, _ = load_mask(paths["prompt_confirmed"] / name, reference)
        require_equal(name, "prompt_confirmation", prompt_confirmed, expected)

        guided_mask, _ = load_mask(paths["guided"] / name, reference)
        guided = add_guided_components(prompt_confirmed, guided_mask, prompt_union, spacing)
        expected, _ = load_mask(paths["guided_expected"] / name, reference)
        require_equal(name, "guided_new2k_s25", guided, expected)

        cbg, _ = load_mask(paths["cbg"] / name, reference)
        cbg_confirmed = confirm_whole_components_26(guided, cbg)
        expected, _ = load_mask(paths["cbg_confirmed"] / name, reference)
        require_equal(name, "cbg_confirmation", cbg_confirmed, expected)
        touching_cbg = union_touching_support_components_26(cbg_confirmed, cbg)

        freetumor, _ = load_mask(paths["freetumor"] / name, reference)
        probability_path = paths["stunet"] / f"{name.removesuffix('.nii.gz')}.prob16.npz"
        with np.load(probability_path) as archive:
            probability = archive["lesion_probability_uint16"]
        final = confirm_with_freetumor_or_stunet_p70(touching_cbg, freetumor, probability)
        independent = independent_final_reference(touching_cbg, freetumor, probability)
        require_equal(name, "final_confirmation_reference", final, independent)
        target, _ = load_mask(paths["labels"] / name, reference)
        case_dice.append(dice(final, target))
        final_digest.update(name.encode("utf-8") + b"\0")
        final_digest.update(np.asarray(final, dtype=np.uint8).tobytes(order="C"))

        for key, mask in (
            ("bbox_raw", bbox_raw),
            ("bbox_filtered", bbox_filtered),
            ("prompt_confirmed", prompt_confirmed),
            ("guided", guided),
            ("cbg_confirmed", cbg_confirmed),
            ("touching_cbg", touching_cbg),
            ("final", final),
        ):
            totals[key] += int(mask.sum())
        print(f"[{index:02d}/50] {name}", flush=True)

    mean_dsc = float(np.mean(case_dice))
    if not np.isclose(mean_dsc, EXPECTED_DSC, rtol=0.0, atol=5e-8):
        raise AssertionError(f"final DSC mismatch: {mean_dsc} != {EXPECTED_DSC}")
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": "WINNER_MASK_RECIPE_PUBLIC_EQUIVALENT",
        "cases": len(names),
        "voxel_exact_stages": [
            "bbox_filter",
            "prompt_union",
            "prompt_confirmation",
            "guided_new2k_s25",
            "cbg_confirmation",
            "final_confirmation_reference",
        ],
        "final_mean_DSC": mean_dsc,
        "expected_final_mean_DSC": EXPECTED_DSC,
        "aggregate_final_mask_sha256": final_digest.hexdigest(),
        "foreground_voxel_totals": totals,
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
