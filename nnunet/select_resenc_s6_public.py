#!/usr/bin/env python3
"""Select one corrected ResEnc S6 endpoint under its frozen public policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ANALYSIS_TO_POLICY = {
    "resenc_standalone_p50": "resenc_standalone_p50",
    "gf_anchor_confirm_either_p70": "gf_anchor_confirm_either_p70",
    "gf_resenc_boundary3mm_p70p50_ftu": (
        "gf_resenc_boundary3mm_p70p50_ftu"
    ),
    "gf_union_resenc_p40_ftu_s10_mm3000": (
        "gf_union_resenc_p40_ftu_s10_mm3000"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_epoch_asset(value: str) -> tuple[int, Path]:
    try:
        epoch_text, path_text = value.split("=", 1)
        epoch = int(epoch_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("expected EPOCH=PATH") from error
    return epoch, Path(path_text)


def load_policy(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = [item["name"] for item in payload.get("deployments", [])]
    if (
        payload.get("status") != "PREDECLARED_BEFORE_TRAINING"
        or payload.get("checkpoint_epochs") != [100, 200, 300, 400, 500]
        or names != list(ANALYSIS_TO_POLICY.values())
        or payload.get("threshold_search_after_public_read") is not False
        or payload.get("public_used_for_training")
        or payload.get("hidden_read")
    ):
        raise ValueError("invalid ResEnc S6 public-selection policy")
    return payload


def select(
    policy_path: Path,
    analyses: dict[int, Path],
    checkpoints: dict[int, Path],
    *,
    policy_loader=None,
) -> dict:
    # Keep the base and positive-unlabeled selectors isolated.  The positive
    # selector used to replace this module's ``load_policy`` global at import
    # time.  That made selection depend on test/import order and could make a
    # completed base run reject its own frozen policy.
    policy = (policy_loader or load_policy)(policy_path)
    epochs = policy["checkpoint_epochs"]
    if sorted(analyses) != epochs or sorted(checkpoints) != epochs:
        raise ValueError("analysis/checkpoint epochs do not match frozen policy")
    incumbent = policy["incumbent_fixed_public"]
    qualification = policy["selection"]["qualification"]
    rows = []
    for epoch in epochs:
        analysis_path = analyses[epoch]
        checkpoint_path = checkpoints[epoch]
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        if (
            analysis.get("status") != "COMPLETE"
            or analysis.get("case_count") != 50
            or analysis.get("public_labels_read") is not True
            or analysis.get("hidden_read")
            or analysis.get("submission_created")
            or analysis.get("uploaded")
        ):
            raise ValueError(f"invalid fixed-public analysis at epoch {epoch}")
        metrics = analysis.get("metrics", {})
        for analysis_name, policy_name in ANALYSIS_TO_POLICY.items():
            if analysis_name not in metrics:
                raise ValueError(
                    f"epoch {epoch} is missing deployment {analysis_name}"
                )
            values = metrics[analysis_name]
            dsc = float(values["Lesion_DSC"])
            nsd = float(values["Lesion_NSD"])
            dsc_gain = dsc - float(incumbent["Lesion_DSC"])
            nsd_gain = nsd - float(incumbent["Lesion_NSD"])
            qualified = (
                dsc >= float(qualification["minimum_fixed_public_DSC"])
                and nsd >= float(qualification["minimum_fixed_public_NSD"])
                and dsc_gain >= float(
                    qualification["minimum_DSC_gain_over_incumbent"]
                )
                and nsd_gain >= float(
                    qualification["minimum_NSD_gain_over_incumbent"]
                )
            )
            rows.append(
                {
                    "epoch": epoch,
                    "deployment": policy_name,
                    "analysis_configuration": analysis_name,
                    "Lesion_DSC": dsc,
                    "Lesion_NSD": nsd,
                    "DSC_gain": dsc_gain,
                    "NSD_gain": nsd_gain,
                    "qualified": qualified,
                    "checkpoint": {
                        "path": str(checkpoint_path.resolve()),
                        "sha256": sha256_file(checkpoint_path),
                    },
                    "analysis": {
                        "path": str(analysis_path.resolve()),
                        "sha256": sha256_file(analysis_path),
                    },
                }
            )
    qualified_rows = [row for row in rows if row["qualified"]]
    selected = (
        max(
            qualified_rows,
            key=lambda row: (row["Lesion_DSC"], row["Lesion_NSD"]),
        )
        if qualified_rows
        else None
    )
    diagnostic_best = max(
        rows, key=lambda row: (row["Lesion_DSC"], row["Lesion_NSD"])
    )
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "decision": (
            "REVIEW_TRAINED_CANDIDATE" if selected else "NO_HIDDEN_ARCHIVE"
        ),
        "policy": {
            "path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path),
        },
        "predeclared_epochs": epochs,
        "predeclared_deployments": list(ANALYSIS_TO_POLICY.values()),
        "results": rows,
        "selected": selected,
        "diagnostic_best": diagnostic_best,
        "selection_rule": (
            "maximum DSC then NSD among rows satisfying absolute DSC/NSD, "
            "DSC-gain, and NSD-gain gates"
        ),
        "public_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(select_fn=select) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument(
        "--analysis", action="append", required=True, type=parse_epoch_asset
    )
    parser.add_argument(
        "--checkpoint", action="append", required=True, type=parse_epoch_asset
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyses = dict(args.analysis)
    checkpoints = dict(args.checkpoint)
    if len(analyses) != len(args.analysis) or len(checkpoints) != len(args.checkpoint):
        raise ValueError("duplicate epoch asset")
    payload = select_fn(args.policy, analyses, checkpoints)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
