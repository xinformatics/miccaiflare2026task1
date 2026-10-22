#!/usr/bin/env python3
"""Score replacement of the expensive final STU confirmer by cached supports.

This is a fixed-public runtime ablation.  It reconstructs the immutable winner
anchor, keeps complete 26-connected anchor components selected by the frozen
existing-support proxy, and compares that result with the historical STU OR
FreeTumor deployment.  It cannot read hidden data or create a submission.
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

from analyze_final_confirmer_ablation_public import load_binary, reconstruct_pre_final  # noqa: E402
from fast_official_lesion_metrics import score_case  # noqa: E402
from runtime_winner_mask_recipe import existing_support_proxy_mask  # noqa: E402
from winner_mask_recipe import confirm_whole_components_26  # noqa: E402


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


def evaluate_case(item: tuple[str, dict[str, str]]) -> dict[str, object]:
    name, rendered = item
    paths = {key: Path(value) for key, value in rendered.items()}
    anchor, reference = reconstruct_pre_final(name, rendered)
    whole, _ = load_binary(paths["prompt_whole"] / name, reference)
    tumor, _ = load_binary(paths["prompt_tumor"] / name, reference)
    bbox_raw, _ = load_binary(paths["bbox_raw"] / name, reference)
    truth, _ = load_binary(paths["labels"] / name, reference)
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    support = existing_support_proxy_mask(
        anchor=anchor,
        prompt_whole_body_lesion=whole,
        prompt_tumor=tumor,
        bbox_raw=bbox_raw,
        spacing=spacing,
    )
    prediction = confirm_whole_components_26(anchor, support)
    dsc, nsd = score_case(
        truth.astype(np.uint8, copy=False),
        prediction.astype(np.uint8, copy=False),
        spacing,
    )
    return {
        "case": name,
        "Lesion_DSC": float(dsc),
        "Lesion_NSD": float(nsd),
        "anchor_voxels": int(anchor.sum()),
        "support_voxels": int(support.sum()),
        "foreground_voxels": int(prediction.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--historical",
        type=Path,
        default=DOCKER_ROOT / "analysis/final_confirmer_ablation_public_20260827.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCKER_ROOT / "analysis/existing_support_proxy_only_public_20260827.json",
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
        "labels": PROJECT_ROOT / "validation/Validation-Public-Labels",
    }
    label_parts = {part.casefold() for part in paths["labels"].resolve().parts}
    if "validation-public-labels" not in label_parts or "validation-hidden-labels" in label_parts:
        raise ValueError("fixed-public labels are required")
    names = sorted(path.name for path in paths["labels"].glob("*.nii.gz"))
    if len(names) != 50:
        raise RuntimeError(f"fixed-public inventory must contain 50 cases, found {len(names)}")
    expected = set(names)
    for role, directory in paths.items():
        if {path.name for path in directory.glob("*.nii.gz")} != expected:
            raise RuntimeError(f"{role} inventory mismatch: {directory}")

    historical = json.loads(args.historical.read_text(encoding="utf-8"))
    if (
        historical.get("status") != "COMPLETE"
        or historical.get("public_labels_read") is not True
        or historical.get("hidden_read")
        or not np.isclose(
            historical["aggregates"]["historical_or"]["Lesion_DSC"],
            0.6216639257651584,
            rtol=0.0,
            atol=5e-8,
        )
    ):
        raise ValueError("invalid historical winner evidence")

    rendered = {key: str(value.resolve()) for key, value in paths.items()}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cases = list(executor.map(evaluate_case, ((name, rendered) for name in names)))
    dsc = np.asarray([case["Lesion_DSC"] for case in cases])
    nsd = np.asarray([case["Lesion_NSD"] for case in cases])
    winner = historical["aggregates"]["historical_or"]
    aggregate = {
        "Lesion_DSC": float(dsc.mean()),
        "Lesion_NSD": float(nsd.mean()),
        "zero_DSC_cases": int(np.count_nonzero(dsc == 0)),
        "delta_DSC_vs_historical_or": float(dsc.mean() - winner["Lesion_DSC"]),
        "delta_NSD_vs_historical_or": float(nsd.mean() - winner["Lesion_NSD"]),
    }
    aggregate["preserves_runtime_anchor"] = bool(
        aggregate["Lesion_DSC"] >= winner["Lesion_DSC"] - 0.005
        and aggregate["Lesion_NSD"] >= winner["Lesion_NSD"] - 0.005
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": "DROP_STUNET" if aggregate["preserves_runtime_anchor"] else "CROPPED_STUNET_REQUIRED",
        "case_count": len(cases),
        "aggregate": aggregate,
        "historical_or": winner,
        "cases": cases,
        "historical_evidence": {
            "path": str(args.historical.resolve()),
            "sha256": sha256(args.historical),
        },
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "decision", "aggregate")}, indent=2))


if __name__ == "__main__":
    main()
