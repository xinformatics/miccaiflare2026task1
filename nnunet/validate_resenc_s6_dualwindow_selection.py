#!/usr/bin/env python3
"""Validate a dual-window public selection before hidden inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from nnunet.select_resenc_s6_dualwindow_public import (
    EXPECTED_EPOCHS,
    load_policy,
)


DEPLOYMENTS = (
    "resenc_standalone_p50",
    "gf_anchor_confirm_either_p70",
    "gf_resenc_boundary3mm_p70p50_ftu",
    "gf_union_resenc_p40_ftu_s10_mm3000",
)
TRAINER_NAME = "nnUNetTrainerResEncS6PanCancerDualWindow"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SelectedCandidate:
    epoch: int
    deployment: str
    checkpoint: str
    checkpoint_sha256: str
    fixed_public_DSC: float
    fixed_public_NSD: float


def validate_public_selection(
    terminal_path: Path,
    policy_path: Path,
    model_dir: Path,
) -> SelectedCandidate | None:
    policy = load_policy(policy_path)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    policy_asset = terminal.get("policy", {})
    if (
        terminal.get("status") != "COMPLETE"
        or terminal.get("predeclared_epochs") != EXPECTED_EPOCHS
        or terminal.get("predeclared_deployments") != list(DEPLOYMENTS)
        or Path(policy_asset.get("path", "")).resolve() != policy_path.resolve()
        or policy_asset.get("sha256") != sha256_file(policy_path)
        or terminal.get("public_read") is not True
        or terminal.get("hidden_read")
        or terminal.get("submission_created")
        or terminal.get("uploaded")
    ):
        raise ValueError("invalid dual-window public selection terminal")
    decision = terminal.get("decision")
    selected = terminal.get("selected")
    if decision == "NO_HIDDEN_ARCHIVE":
        if selected is not None:
            raise ValueError("no-archive decision contains a selected candidate")
        return None
    if decision != "REVIEW_TRAINED_CANDIDATE" or not isinstance(selected, dict):
        raise ValueError("unexpected dual-window public decision")
    epoch = int(selected.get("epoch", -1))
    deployment = selected.get("deployment")
    dsc = float(selected.get("Lesion_DSC", float("nan")))
    nsd = float(selected.get("Lesion_NSD", float("nan")))
    qualification = policy["selection"]["qualification"]
    incumbent = policy["incumbent_fixed_public"]
    if (
        epoch not in EXPECTED_EPOCHS
        or deployment not in DEPLOYMENTS
        or selected.get("analysis_configuration") != deployment
        or selected.get("qualified") is not True
        or dsc < float(qualification["minimum_fixed_public_DSC"])
        or nsd < float(qualification["minimum_fixed_public_NSD"])
        or dsc - float(incumbent["Lesion_DSC"])
        < float(qualification["minimum_DSC_gain_over_incumbent"])
        or nsd - float(incumbent["Lesion_NSD"])
        < float(qualification["minimum_NSD_gain_over_incumbent"])
    ):
        raise ValueError("selected dual-window candidate does not pass frozen gates")
    checkpoint_name = (
        "checkpoint_final.pth"
        if epoch == 100
        else f"checkpoint_epoch_{epoch:04d}.pth"
    )
    expected_checkpoint = (model_dir / "fold_all" / checkpoint_name).resolve()
    checkpoint_asset = selected.get("checkpoint", {})
    checkpoint = Path(checkpoint_asset.get("path", ""))
    if (
        checkpoint.resolve() != expected_checkpoint
        or not checkpoint.is_file()
        or checkpoint_asset.get("sha256") != sha256_file(checkpoint)
    ):
        raise ValueError("selected dual-window checkpoint identity mismatch")
    return SelectedCandidate(
        epoch=epoch,
        deployment=deployment,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_asset["sha256"],
        fixed_public_DSC=dsc,
        fixed_public_NSD=nsd,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    selected = validate_public_selection(args.terminal, args.policy, args.model_dir)
    print(json.dumps(None if selected is None else asdict(selected), sort_keys=True))


if __name__ == "__main__":
    main()

