#!/usr/bin/env python3
"""Find cheap cached supports that can replace FreeTumor at the final gate.

This analysis is label-blind. It treats the expensive FreeTumor output only as
a teacher for anchor-component confirmation and measures whether supports that
already exist in the winner graph (BiomedParse prompts, CBG, guided GF, and the
bbox GF proposal) can reproduce FreeTumor-unique decisions when STU p70 does
not confirm a component. No validation labels are opened.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from analyze_final_confirmer_ablation_public import load_binary  # noqa: E402
from winner_mask_recipe import (  # noqa: E402
    add_guided_components,
    confirm_whole_components_26,
    filter_components_physical_6,
    union_touching_support_components_26,
)


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


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


def overlap_features(count: int, overlap: int, voxel_mm3: float) -> dict:
    return {
        "voxels": overlap,
        "fraction": float(overlap / count),
        "mm3": float(overlap * voxel_mm3),
        "touches": bool(overlap),
    }


def case_features(item: tuple[str, dict[str, str]]) -> dict:
    name, rendered = item
    paths = {key: Path(value) for key, value in rendered.items()}
    bbox, reference = load_binary(paths["bbox_raw"] / name)
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    prompt_whole, _ = load_binary(paths["prompt_whole"] / name, reference)
    prompt_tumor, _ = load_binary(paths["prompt_tumor"] / name, reference)
    guided, _ = load_binary(paths["guided"] / name, reference)
    cbg, _ = load_binary(paths["cbg"] / name, reference)
    freetumor, _ = load_binary(paths["freetumor"] / name, reference)
    probability_path = paths["stunet"] / f"{name.removesuffix('.nii.gz')}.prob16.npz"
    with np.load(probability_path) as archive:
        probability = archive["lesion_probability_uint16"]
    if probability.dtype != np.uint16 or probability.shape != bbox.shape:
        raise ValueError(f"invalid STU probability: {probability_path}")
    stunet = probability.astype(np.float32) / 65535.0 > 0.70
    prompt_union = prompt_whole | prompt_tumor
    bbox_filtered = filter_components_physical_6(bbox, spacing)
    prompt_confirmed = confirm_whole_components_26(bbox_filtered, prompt_union)
    guided_stage = add_guided_components(
        prompt_confirmed, guided, prompt_union, spacing
    )
    cbg_confirmed = confirm_whole_components_26(guided_stage, cbg)
    anchor = union_touching_support_components_26(cbg_confirmed, cbg)
    labels, count = ndimage.label(anchor, structure=CONNECTIVITY_26)
    voxel_mm3 = float(abs(np.linalg.det(reference.affine[:3, :3])))
    component_counts = np.bincount(labels.ravel(), minlength=int(count) + 1)
    supports = {
        "stunet": stunet,
        "freetumor": freetumor,
        "prompt_union": prompt_union,
        "prompt_whole": prompt_whole,
        "prompt_tumor": prompt_tumor,
        "guided": guided,
        "cbg": cbg,
        "bbox_raw": bbox,
    }
    overlap_counts = {
        support_name: np.bincount(
            labels[support], minlength=int(count) + 1
        )
        for support_name, support in supports.items()
    }
    rows = []
    for component_id in range(1, int(count) + 1):
        voxels = int(component_counts[component_id])
        features = {
            support_name: overlap_features(
                voxels, int(counts[component_id]), voxel_mm3
            )
            for support_name, counts in overlap_counts.items()
        }
        stu = features["stunet"]
        free = features["freetumor"]
        rows.append(
            {
                "component": component_id,
                "voxels": voxels,
                "volume_mm3": float(voxels * voxel_mm3),
                "stunet": stu,
                "freetumor": free,
                "freetumor_unique_teacher_positive": bool(
                    free["touches"] and not stu["touches"]
                ),
                "prompt_union": features["prompt_union"],
                "prompt_whole": features["prompt_whole"],
                "prompt_tumor": features["prompt_tumor"],
                "guided": features["guided"],
                "cbg": features["cbg"],
                "bbox_raw": features["bbox_raw"],
            }
        )
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "case": name,
        "component_count": int(count),
        "components": rows,
        "public_labels_read": False,
        "hidden_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCKER_ROOT / "analysis/free_teacher_proxy_features_public_20260827.json",
    )
    parser.add_argument(
        "--case-cache",
        type=Path,
        default=DOCKER_ROOT / "analysis/free_teacher_proxy_case_cache_20260827",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1,4]")
    paths = {
        "bbox_raw": GF_ROOT / "predictions/gf_a055_bbox_pu_best_ep280/public_raw",
        "prompt_whole": ROOT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti",
        "prompt_tumor": ROOT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti",
        "guided": ROOT / "external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500",
        "cbg": GF_ROOT / "public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/deployed_mm1000_new2k_s25",
        "freetumor": ROOT / "external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions",
        "stunet": ROOT / "external/STU-Net/pancancer/public_sweeps/stunetb_matched_e160_20260826/component_predictions",
    }
    names = sorted(path.name for path in paths["bbox_raw"].glob("*.nii.gz"))
    expected = set(names)
    if len(names) != 50:
        raise RuntimeError("fixed-public image inventory must contain 50 cases")
    for role, directory in paths.items():
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected:
            raise RuntimeError(f"{role} inventory mismatch")
    rendered = {key: str(value.resolve()) for key, value in paths.items()}
    args.case_cache.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    pending = []
    for name in names:
        cache_path = args.case_cache / f"{name.removesuffix('.nii.gz')}.json"
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("status") == "COMPLETE"
            and cached.get("case") == name
            and cached.get("public_labels_read") is False
            and cached.get("hidden_read") is False
            and isinstance(cached.get("components"), list)
        ):
            completed[name] = cached
        else:
            pending.append(name)

    def save_case(payload: dict) -> None:
        name = payload["case"]
        atomic_json(
            args.case_cache / f"{name.removesuffix('.nii.gz')}.json", payload
        )
        completed[name] = payload
        print(f"free-teacher proxy progress {len(completed)}/50: {name}", flush=True)

    if args.workers == 1:
        for name in pending:
            save_case(case_features((name, rendered)))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(case_features, (name, rendered)): name
                for name in pending
            }
            for future in as_completed(futures):
                save_case(future.result())
    if set(completed) != expected:
        raise RuntimeError("incomplete label-blind proxy feature cache")
    cases = [completed[name] for name in names]
    components = [component for case in cases for component in case["components"]]
    unique = [row for row in components if row["freetumor_unique_teacher_positive"]]
    unresolved = [
        row for row in components
        if not row["stunet"]["touches"] and not row["freetumor"]["touches"]
    ]
    support_names = ("prompt_union", "prompt_whole", "prompt_tumor", "guided", "cbg", "bbox_raw")
    summary = {}
    for support in support_names:
        summary[support] = {
            "free_unique_touched": int(sum(row[support]["touches"] for row in unique)),
            "free_unique_total": len(unique),
            "unresolved_touched": int(sum(row[support]["touches"] for row in unresolved)),
            "unresolved_total": len(unresolved),
            "free_unique_min_fraction": (
                float(min(row[support]["fraction"] for row in unique)) if unique else None
            ),
            "free_unique_min_mm3": (
                float(min(row[support]["mm3"] for row in unique)) if unique else None
            ),
        }
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision_scope": "LABEL_BLIND_FREETUMOR_TEACHER_PROXY_FEATURES",
        "case_count": len(cases),
        "anchor_components": len(components),
        "stunet_confirmed_components": int(sum(row["stunet"]["touches"] for row in components)),
        "freetumor_unique_teacher_components": len(unique),
        "neither_teacher_confirmed_components": len(unresolved),
        "support_summary": summary,
        "case_cache": str(args.case_cache.resolve()),
        "cases": cases,
        "source_sha256": {
            "freetumor": sha256(ROOT / "external/FreeTumor/pretrained_models/baseline/model_baseline_segmentor.pt"),
            "stunet": sha256(ROOT / "nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_inference_epoch_0160.pth"),
        },
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in (
        "status", "anchor_components", "stunet_confirmed_components",
        "freetumor_unique_teacher_components", "neither_teacher_confirmed_components",
        "support_summary",
    )}, indent=2))


if __name__ == "__main__":
    main()
