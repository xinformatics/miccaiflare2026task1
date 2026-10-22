#!/usr/bin/env python3
"""Public-only deployment ablations requested in ``claude_AUG30.txt``.

The script consumes frozen model-product caches. It does not run a network,
read hidden data, package predictions, or upload anything. Its purposes are:

* isolate the two BiomedParse roles in the exact winner recipe;
* isolate STU-Net and FreeTumor in the final whole-component confirmation; and
* test support-scaled admission of guided components larger than 2,000 mm3.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np


HERE = Path(__file__).resolve().parent
GF_ROOT = HERE.parent
PROJECT = GF_ROOT.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GF_ROOT))

from fast_official_lesion_metrics import (  # noqa: E402
    prepare_surface_reference,
    score_case_with_reference,
)
from winner_mask_recipe import (  # noqa: E402
    add_guided_components,
    confirm_whole_components_26,
    filter_components_physical_6,
    union_touching_support_components_26,
)


TAU_LARGE = (0.40, 0.50, 0.60, 0.70)
EXPECTED_FULL_DSC = 0.6216639257651584


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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


def confirm_final(
    anchor: np.ndarray,
    freetumor: np.ndarray,
    stunet_probability_uint16: np.ndarray,
    mode: str,
) -> np.ndarray:
    stunet = stunet_probability_uint16.astype(np.float32) / 65535.0 > 0.70
    if mode == "either":
        support = freetumor | stunet
    elif mode == "stunet":
        support = stunet
    elif mode == "freetumor":
        support = freetumor
    elif mode == "none":
        return anchor.copy()
    else:
        raise ValueError(f"invalid final confirmation mode: {mode}")
    return confirm_whole_components_26(anchor, support)


def add_support_scaled_components(
    base: np.ndarray,
    guided: np.ndarray,
    support: np.ndarray,
    spacing: tuple[float, float, float],
    tau_large: float,
) -> tuple[np.ndarray, int, float]:
    """Keep the frozen small rule and add a support-scaled large path."""

    labels = cc3d.connected_components(guided.astype(np.uint8), connectivity=26)
    counts = np.bincount(labels.ravel())
    base_overlap = np.bincount(labels[base], minlength=len(counts))
    support_counts = np.bincount(labels[support], minlength=len(counts))
    voxel_mm3 = float(np.prod(spacing))
    volumes = counts * voxel_mm3
    fractions = np.divide(
        support_counts,
        counts,
        out=np.zeros_like(volumes, dtype=np.float64),
        where=counts > 0,
    )
    supported_mm3 = support_counts * voxel_mm3
    disconnected = base_overlap == 0
    small = (
        disconnected
        & (volumes <= 2_000.0)
        & (fractions >= 0.25)
        & (supported_mm3 >= 100.0)
    )
    large = disconnected & (volumes > 2_000.0) & (fractions >= tau_large)
    keep = small | large
    keep[0] = False
    large[0] = False
    return base | keep[labels], int(np.count_nonzero(large)), float(volumes[large].sum())


def paths() -> dict[str, Path]:
    return {
        "bbox_raw": PROJECT / "external/GF-Screen/predictions/gf_a055_bbox_pu_best_ep280/public_raw",
        "prompt_whole": PROJECT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti",
        "prompt_tumor": PROJECT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti",
        "guided": PROJECT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500",
        "cbg": PROJECT / "external/GF-Screen/public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/deployed_mm1000_new2k_s25",
        "freetumor": PROJECT / "external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions",
        "stunet": PROJECT / "external/STU-Net/pancancer/public_sweeps/stunetb_matched_e160_20260826/component_predictions",
        "labels": PROJECT / "validation/Validation-Public-Labels",
    }


def compose_downstream(
    base: np.ndarray,
    cbg: np.ndarray,
    freetumor: np.ndarray,
    probability: np.ndarray,
    final_mode: str = "either",
) -> np.ndarray:
    cbg_confirmed = confirm_whole_components_26(base, cbg)
    touching_cbg = union_touching_support_components_26(cbg_confirmed, cbg)
    return confirm_final(touching_cbg, freetumor, probability, final_mode)


def evaluate_case(name: str, serialized_inputs: dict[str, str]):
    """Evaluate every frozen configuration for one public case."""

    inputs = {role: Path(path) for role, path in serialized_inputs.items()}
    bbox_raw, reference = load_mask(inputs["bbox_raw"] / name)
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    prompt_whole, _ = load_mask(inputs["prompt_whole"] / name, reference)
    prompt_tumor, _ = load_mask(inputs["prompt_tumor"] / name, reference)
    guided_gf, _ = load_mask(inputs["guided"] / name, reference)
    cbg, _ = load_mask(inputs["cbg"] / name, reference)
    freetumor, _ = load_mask(inputs["freetumor"] / name, reference)
    target, _ = load_mask(inputs["labels"] / name, reference)
    probability_path = inputs["stunet"] / f"{name.removesuffix('.nii.gz')}.prob16.npz"
    with np.load(probability_path) as archive:
        probability = archive["lesion_probability_uint16"]
    if probability.shape != bbox_raw.shape or probability.dtype != np.uint16:
        raise ValueError(f"invalid STU probability: {probability_path}")

    bbox_filtered = filter_components_physical_6(bbox_raw, spacing, 1000.0)
    prompt_support = prompt_whole | prompt_tumor
    prompt_confirmed = confirm_whole_components_26(bbox_filtered, prompt_support)
    guided_full = add_guided_components(prompt_confirmed, guided_gf, prompt_support, spacing)
    guided_without_confirmation = add_guided_components(
        bbox_filtered, guided_gf, prompt_support, spacing
    )
    cbg_confirmed = confirm_whole_components_26(guided_full, cbg)
    touching_cbg = union_touching_support_components_26(cbg_confirmed, cbg)
    configurations = {
        "bbox_filtered_no_biomed_no_downstream": bbox_filtered,
        "biomed_confirmation_only_no_downstream": prompt_confirmed,
        "full_without_biomed_confirmation": compose_downstream(
            guided_without_confirmation, cbg, freetumor, probability
        ),
        "full_without_biomed_addition": compose_downstream(
            prompt_confirmed, cbg, freetumor, probability
        ),
        "full_without_biomed": compose_downstream(
            bbox_filtered, cbg, freetumor, probability
        ),
        "pre_final_no_stu_or_freetumor": touching_cbg,
        "final_stunet_only": confirm_final(
            touching_cbg, freetumor, probability, "stunet"
        ),
        "final_freetumor_only": confirm_final(
            touching_cbg, freetumor, probability, "freetumor"
        ),
        "final_full": confirm_final(
            touching_cbg, freetumor, probability, "either"
        ),
    }
    case_large_counts: dict[float, int] = {}
    case_large_mm3: dict[float, float] = {}
    for tau in TAU_LARGE:
        extended, count, volume = add_support_scaled_components(
            prompt_confirmed, guided_gf, prompt_support, spacing, tau
        )
        case_large_counts[tau] = count
        case_large_mm3[tau] = volume
        configurations[f"final_tau_large_{tau:.2f}"] = compose_downstream(
            extended, cbg, freetumor, probability
        )

    surface_reference = prepare_surface_reference(target, spacing)
    row: dict[str, str | float] = {"Name": name}
    case_scores: dict[str, tuple[float, float]] = {}
    for configuration, prediction in configurations.items():
        dsc, nsd = score_case_with_reference(
            target.astype(np.uint8),
            prediction.astype(np.uint8),
            spacing,
            surface_reference,
        )
        case_scores[configuration] = float(dsc), float(nsd)
        row[f"{configuration}_DSC"] = float(dsc)
        row[f"{configuration}_NSD"] = float(nsd)
    return row, case_scores, case_large_counts, case_large_mm3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-case", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    inputs = paths()
    names = sorted(path.name for path in inputs["bbox_raw"].glob("*.nii.gz"))
    if len(names) != 50:
        raise RuntimeError(f"expected 50 public cases, found {len(names)}")
    expected = set(names)
    for role, directory in inputs.items():
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected:
            raise RuntimeError(f"{role} inventory mismatch: {directory}")

    configuration_names = [
        "bbox_filtered_no_biomed_no_downstream",
        "biomed_confirmation_only_no_downstream",
        "full_without_biomed_confirmation",
        "full_without_biomed_addition",
        "full_without_biomed",
        "pre_final_no_stu_or_freetumor",
        "final_stunet_only",
        "final_freetumor_only",
        "final_full",
    ] + [f"final_tau_large_{tau:.2f}" for tau in TAU_LARGE]
    values = {name: {"Lesion_DSC": [], "Lesion_NSD": []} for name in configuration_names}
    per_case: list[dict] = []
    large_counts = {tau: 0 for tau in TAU_LARGE}
    large_mm3 = {tau: 0.0 for tau in TAU_LARGE}

    serialized_inputs = {role: str(directory) for role, directory in inputs.items()}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(
            evaluate_case,
            names,
            [serialized_inputs] * len(names),
            chunksize=1,
        )
        for index, (row, case_scores, case_counts, case_mm3) in enumerate(results, 1):
            for configuration, (dsc, nsd) in case_scores.items():
                values[configuration]["Lesion_DSC"].append(dsc)
                values[configuration]["Lesion_NSD"].append(nsd)
            for tau in TAU_LARGE:
                large_counts[tau] += case_counts[tau]
                large_mm3[tau] += case_mm3[tau]
            per_case.append(row)
            print(f"[{index:02d}/50] {row['Name']}", flush=True)

    metrics = {
        configuration: {
            metric: float(np.mean(metric_values))
            for metric, metric_values in scores.items()
        }
        for configuration, scores in values.items()
    }
    if not np.isclose(
        metrics["final_full"]["Lesion_DSC"], EXPECTED_FULL_DSC, rtol=0.0, atol=5e-8
    ):
        raise AssertionError(
            f"winner replay drift: {metrics['final_full']['Lesion_DSC']} != {EXPECTED_FULL_DSC}"
        )
    full = metrics["final_full"]
    ranking = sorted(
        (
            {
                "configuration": configuration,
                **metric,
                "DSC_delta_vs_full": metric["Lesion_DSC"] - full["Lesion_DSC"],
                "NSD_delta_vs_full": metric["Lesion_NSD"] - full["Lesion_NSD"],
            }
            for configuration, metric in metrics.items()
        ),
        key=lambda row: (row["Lesion_DSC"], row["Lesion_NSD"]),
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "role": "public_only_cached_deployment_ablation",
        "cases": len(names),
        "metrics": metrics,
        "ranking": ranking,
        "large_component_admission": {
            f"{tau:.2f}": {
                "accepted_component_occurrences": large_counts[tau],
                "accepted_volume_mm3": large_mm3[tau],
            }
            for tau in TAU_LARGE
        },
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    args.per_case.parent.mkdir(parents=True, exist_ok=True)
    with args.per_case.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case[0]))
        writer.writeheader()
        writer.writerows(per_case)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
