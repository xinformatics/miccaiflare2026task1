#!/usr/bin/env python3
"""Score the frozen label-blind FreeTumor proxy on fixed public labels."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[3]
GF_ROOT = ROOT / "external/GF-Screen"
DOCKER_ROOT = GF_ROOT / "docker_submission"
for source in (GF_ROOT, DOCKER_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from analyze_final_confirmer_ablation_public import (  # noqa: E402
    load_binary,
    reconstruct_pre_final,
)
from fast_official_lesion_metrics import score_case  # noqa: E402
from winner_mask_recipe import confirm_whole_components_26  # noqa: E402


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)
VARIANTS = ("stunet_only", "tumor_fraction_040_only", "high_precision_proxy_v1", "historical_or")


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


def tumor_rule(row: dict) -> bool:
    return float(row["prompt_tumor"]["fraction"]) >= 0.40


def proxy_rule(row: dict) -> bool:
    tumor = float(row["prompt_tumor"]["fraction"])
    whole = float(row["prompt_whole"]["fraction"])
    bbox = float(row["bbox_raw"]["fraction"])
    volume = float(row["volume_mm3"])
    return bool(
        tumor >= 0.40
        or (whole >= 0.40 and bbox < 0.50)
        or (whole >= 0.75 and volume >= 800.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=DOCKER_ROOT / "FREE_TEACHER_PROXY_POLICY_20260827.json",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DOCKER_ROOT / "analysis/free_teacher_proxy_features_public_20260827.json",
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=DOCKER_ROOT / "analysis/final_confirmer_ablation_public_20260827.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCKER_ROOT / "analysis/free_teacher_proxy_public_metrics_20260827.json",
    )
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    features = json.loads(args.features.read_text(encoding="utf-8"))
    ablation = json.loads(args.ablation.read_text(encoding="utf-8"))
    expected_feature_sha = policy.get("feature_analysis", {}).get("sha256")
    if (
        policy.get("status") != "FROZEN_BEFORE_PUBLIC_LABEL_SCORE"
        or expected_feature_sha != sha256(args.features)
        or policy.get("public_labels_used_to_define_rule") is not False
        or features.get("status") != "COMPLETE"
        or features.get("public_labels_read") is not False
        or features.get("case_count") != 50
        or ablation.get("status") != "COMPLETE"
        or ablation.get("public_labels_read") is not True
        or ablation.get("hidden_read")
    ):
        raise ValueError("invalid frozen FreeTumor proxy evidence")
    feature_cases = {case["case"]: case for case in features["cases"]}
    baseline_cases = {case["case"]: case for case in ablation["cases"]}
    names = sorted(feature_cases)
    if len(names) != 50 or set(names) != set(baseline_cases):
        raise ValueError("proxy case inventory mismatch")
    selected = {"tumor_fraction_040_only": {}, "high_precision_proxy_v1": {}}
    teacher_counts = {}
    for variant, rule in (
        ("tumor_fraction_040_only", tumor_rule),
        ("high_precision_proxy_v1", proxy_rule),
    ):
        unique = rejected = 0
        for name in names:
            component_ids = []
            for row in feature_cases[name]["components"]:
                if row["stunet"]["touches"] or not rule(row):
                    continue
                component_ids.append(int(row["component"]))
                if row["freetumor"]["touches"]:
                    unique += 1
                else:
                    rejected += 1
            if component_ids:
                selected[variant][name] = component_ids
        teacher_counts[variant] = {
            "teacher_unique_components_recovered": unique,
            "teacher_rejected_components_recovered": rejected,
        }
    expected = {
        "tumor_fraction_040_only": (1, 0),
        "high_precision_proxy_v1": (3, 0),
    }
    for variant, values in expected.items():
        observed = teacher_counts[variant]
        if (observed["teacher_unique_components_recovered"], observed["teacher_rejected_components_recovered"]) != values:
            raise RuntimeError(f"frozen proxy selection drift: {variant}={observed}")

    paths = {
        "bbox_raw": GF_ROOT / "predictions/gf_a055_bbox_pu_best_ep280/public_raw",
        "prompt_whole": ROOT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti",
        "prompt_tumor": ROOT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti",
        "guided": ROOT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500",
        "cbg": GF_ROOT / "public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/deployed_mm1000_new2k_s25",
        "stunet": ROOT / "external/STU-Net/pancancer/public_sweeps/stunetb_matched_e160_20260826/component_predictions",
        "labels": ROOT / "validation/Validation-Public-Labels",
    }
    label_parts = {part.casefold() for part in paths["labels"].resolve().parts}
    if "validation-public-labels" not in label_parts or "validation-hidden-labels" in label_parts:
        raise ValueError("proxy evaluator requires fixed public labels")
    rendered = {key: str(value.resolve()) for key, value in paths.items()}
    recomputed: dict[str, dict[str, dict]] = {variant: {} for variant in selected}
    changed_names = sorted(set().union(*(set(value) for value in selected.values())))
    for name in changed_names:
        anchor, reference = reconstruct_pre_final(name, rendered)
        labels, count = ndimage.label(anchor, structure=CONNECTIVITY_26)
        probability_path = paths["stunet"] / f"{name.removesuffix('.nii.gz')}.prob16.npz"
        with np.load(probability_path) as archive:
            probability = archive["lesion_probability_uint16"]
        stunet = probability.astype(np.float32) / 65535.0 > 0.70
        truth, _ = load_binary(paths["labels"] / name, reference)
        for variant in selected:
            support = stunet.copy()
            for component_id in selected[variant].get(name, []):
                if not 1 <= component_id <= int(count):
                    raise RuntimeError("proxy component identifier drift")
                support[labels == component_id] = True
            prediction = confirm_whole_components_26(anchor, support)
            dsc, nsd = score_case(
                truth.astype(np.uint8, copy=False),
                prediction.astype(np.uint8, copy=False),
                reference.header.get_zooms()[:3],
            )
            recomputed[variant][name] = {
                "Lesion_DSC": float(dsc),
                "Lesion_NSD": float(nsd),
                "foreground_voxels": int(prediction.sum()),
            }
    cases = []
    for name in names:
        baseline = baseline_cases[name]["variants"]
        variants = {
            "stunet_only": baseline["stunet_only"],
            "tumor_fraction_040_only": recomputed["tumor_fraction_040_only"].get(name, baseline["stunet_only"]),
            "high_precision_proxy_v1": recomputed["high_precision_proxy_v1"].get(name, baseline["stunet_only"]),
            "historical_or": baseline["historical_or"],
        }
        cases.append({"case": name, "variants": variants})
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
    if not np.isclose(historical["Lesion_DSC"], 0.6216639257651584, rtol=0.0, atol=5e-8):
        raise RuntimeError("historical public reference drift")
    floor = policy["selection"]
    for variant, values in aggregates.items():
        values["delta_DSC_vs_historical_or"] = float(values["Lesion_DSC"] - historical["Lesion_DSC"])
        values["delta_NSD_vs_historical_or"] = float(values["Lesion_NSD"] - historical["Lesion_NSD"])
        values["quality_gate_pass"] = bool(
            values["Lesion_DSC"] >= float(floor["minimum_fixed_public_DSC"])
            and values["Lesion_NSD"] >= float(floor["minimum_fixed_public_NSD"])
        )
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": (
            "PROMOTE_PROXY_FOR_RUNTIME_GATE"
            if aggregates["high_precision_proxy_v1"]["quality_gate_pass"]
            else "KEEP_AS_RUNTIME_ABLATION_ONLY"
        ),
        "case_count": 50,
        "teacher_counts": teacher_counts,
        "changed_cases": changed_names,
        "aggregates": aggregates,
        "cases": cases,
        "policy": {"path": str(args.policy.resolve()), "sha256": sha256(args.policy)},
        "features": {"path": str(args.features.resolve()), "sha256": sha256(args.features)},
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "decision", "teacher_counts", "changed_cases", "aggregates")}, indent=2))


if __name__ == "__main__":
    main()
