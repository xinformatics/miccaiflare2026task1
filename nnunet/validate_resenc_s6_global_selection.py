#!/usr/bin/env python3
"""Select the best already-qualified ResEnc S6 candidate across families.

Each family first passes its own predeclared public policy and endpoint
selector. This module does not rescore or retune anything: it validates those
terminals, binds the selected checkpoint by path and SHA256, then ranks the
qualified family winners by public DSC and NSD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from nnunet import select_resenc_s6_dualwindow_public
from nnunet import select_resenc_s6_organ_public
from nnunet import select_resenc_s6_positive_public
from nnunet import select_resenc_s6_public


DEPLOYMENTS = tuple(select_resenc_s6_public.ANALYSIS_TO_POLICY.values())
INCUMBENT = {
    "Lesion_DSC": 0.6216639257651584,
    "Lesion_NSD": 0.5347520263176905,
}
QUALIFICATION = {
    "minimum_fixed_public_DSC": 0.6366639257651584,
    "minimum_fixed_public_NSD": 0.5347520263176905,
    "minimum_DSC_gain_over_incumbent": 0.015,
    "minimum_NSD_gain_over_incumbent": 0.0,
    "material_trained_model_change_required": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FamilySpec:
    name: str
    trainer: str
    epochs: tuple[int, ...]
    final_epoch: int
    terminal: Path
    policy: Path
    model_dir: Path
    predictor: Path
    policy_loader: Callable[[Path], dict]


@dataclass(frozen=True)
class GlobalCandidate:
    family: str
    trainer: str
    predictor: str
    model_dir: str
    epoch: int
    deployment: str
    checkpoint: str
    checkpoint_sha256: str
    fixed_public_DSC: float
    fixed_public_NSD: float
    public_terminal: str
    public_terminal_sha256: str
    policy: str
    policy_sha256: str


def family_specs(project_root: Path) -> tuple[FamilySpec, ...]:
    results = project_root / "nnUNet_results"
    sweeps = project_root / "nnunet" / "public_sweeps"
    code = project_root / "nnunet"
    return (
        FamilySpec(
            "base",
            "nnUNetTrainerResEncS6PanCancerPartialBox",
            (100, 200, 300, 400, 500),
            500,
            sweeps / "resenc_s6_partialbox_endpoints_20260827" / "TERMINAL.json",
            code / "RESENC_S6_PUBLIC_SELECTION_POLICY_20260827.json",
            results
            / "Dataset501_PanCancer"
            / "nnUNetTrainerResEncS6PanCancerPartialBox__nnUNetPlans__3d_fullres_ResEncS6",
            code / "predict_resenc_s6_safe.py",
            select_resenc_s6_public.load_policy,
        ),
        FamilySpec(
            "positive",
            "nnUNetTrainerResEncS6PanCancerPositiveUnlabeled",
            (25, 50, 100),
            100,
            sweeps
            / "resenc_s6_positive_unlabeled_endpoints_20260827"
            / "TERMINAL.json",
            code / "RESENC_S6_POSITIVE_UNLABELED_PUBLIC_SELECTION_POLICY_20260827.json",
            results
            / "Dataset502_PanCancerPositiveUnlabeled"
            / "nnUNetTrainerResEncS6PanCancerPositiveUnlabeled__nnUNetPlans__3d_fullres_ResEncS6",
            code / "predict_resenc_s6_positive_safe.py",
            select_resenc_s6_positive_public.load_policy,
        ),
        FamilySpec(
            "organ",
            "nnUNetTrainerResEncS6PanCancerOrganTarget",
            (10, 25),
            25,
            sweeps / "resenc_s6_organ_target_endpoints_20260827" / "TERMINAL.json",
            code / "RESENC_S6_ORGAN_TARGET_PUBLIC_SELECTION_POLICY_20260827.json",
            results
            / "Dataset503_PanCancerOrganTarget"
            / "nnUNetTrainerResEncS6PanCancerOrganTarget__nnUNetPlans__3d_fullres_ResEncS6",
            code / "predict_resenc_s6_organ_safe.py",
            select_resenc_s6_organ_public.load_policy,
        ),
        FamilySpec(
            "dualwindow",
            "nnUNetTrainerResEncS6PanCancerDualWindow",
            (25, 50, 100),
            100,
            sweeps / "resenc_s6_dualwindow_endpoints_20260827" / "TERMINAL.json",
            code / "RESENC_S6_DUALWINDOW_PUBLIC_SELECTION_POLICY_20260827.json",
            results
            / "Dataset502_PanCancerPositiveUnlabeled"
            / "nnUNetTrainerResEncS6PanCancerDualWindow__nnUNetPlans__3d_fullres_ResEncS6",
            code / "predict_resenc_s6_dualwindow_safe.py",
            select_resenc_s6_dualwindow_public.load_policy,
        ),
    )


def validated_terminal_winner(terminal: dict, spec: FamilySpec, policy: dict) -> dict | None:
    results = terminal.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{spec.name} terminal has no result table")
    incumbent = policy["incumbent_fixed_public"]
    qualification = policy["selection"]["qualification"]
    expected_keys = {
        (epoch, deployment) for epoch in spec.epochs for deployment in DEPLOYMENTS
    }
    observed_keys: set[tuple[int, str]] = set()
    qualified_rows = []
    for row in results:
        try:
            epoch = int(row["epoch"])
            deployment = str(row["deployment"])
            dsc = float(row["Lesion_DSC"])
            nsd = float(row["Lesion_NSD"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid {spec.name} public result row") from error
        key = (epoch, deployment)
        if (
            key not in expected_keys
            or key in observed_keys
            or row.get("analysis_configuration") != deployment
            or not math.isfinite(dsc)
            or not math.isfinite(nsd)
        ):
            raise ValueError(f"invalid {spec.name} public result row")
        observed_keys.add(key)
        computed_qualified = (
            dsc >= float(qualification["minimum_fixed_public_DSC"])
            and nsd >= float(qualification["minimum_fixed_public_NSD"])
            and dsc - float(incumbent["Lesion_DSC"])
            >= float(qualification["minimum_DSC_gain_over_incumbent"])
            and nsd - float(incumbent["Lesion_NSD"])
            >= float(qualification["minimum_NSD_gain_over_incumbent"])
        )
        if row.get("qualified") is not computed_qualified:
            raise ValueError(f"{spec.name} public qualification flag mismatch")
        if computed_qualified:
            qualified_rows.append(row)
    if observed_keys != expected_keys:
        raise ValueError(f"{spec.name} public result coverage mismatch")
    winner = max(
        qualified_rows,
        key=lambda row: (float(row["Lesion_DSC"]), float(row["Lesion_NSD"])),
        default=None,
    )
    selected = terminal.get("selected")
    expected_decision = "REVIEW_TRAINED_CANDIDATE" if winner else "NO_HIDDEN_ARCHIVE"
    if terminal.get("decision") != expected_decision or selected != winner:
        raise ValueError(f"{spec.name} terminal winner mismatch")
    return winner


def validate_family_selection(spec: FamilySpec) -> GlobalCandidate | None:
    policy = spec.policy_loader(spec.policy)
    if (
        policy.get("incumbent_fixed_public") != INCUMBENT
        or policy.get("selection", {}).get("qualification") != QUALIFICATION
    ):
        raise ValueError(f"{spec.name} public policy has a noncomparable gate")
    terminal = json.loads(spec.terminal.read_text(encoding="utf-8"))
    policy_asset = terminal.get("policy", {})
    if (
        terminal.get("status") != "COMPLETE"
        or terminal.get("predeclared_epochs") != list(spec.epochs)
        or terminal.get("predeclared_deployments") != list(DEPLOYMENTS)
        or Path(policy_asset.get("path", "")).resolve() != spec.policy.resolve()
        or policy_asset.get("sha256") != sha256_file(spec.policy)
        or terminal.get("public_read") is not True
        or terminal.get("hidden_read")
        or terminal.get("submission_created")
        or terminal.get("uploaded")
    ):
        raise ValueError(f"invalid {spec.name} public selection terminal")
    selected = validated_terminal_winner(terminal, spec, policy)
    if selected is None:
        return None

    epoch = int(selected.get("epoch", -1))
    deployment = selected.get("deployment")
    dsc = float(selected.get("Lesion_DSC", float("nan")))
    nsd = float(selected.get("Lesion_NSD", float("nan")))
    qualification = policy["selection"]["qualification"]
    incumbent = policy["incumbent_fixed_public"]
    if (
        epoch not in spec.epochs
        or deployment not in DEPLOYMENTS
        or selected.get("analysis_configuration") != deployment
        or selected.get("qualified") is not True
        or not math.isfinite(dsc)
        or not math.isfinite(nsd)
        or dsc < float(qualification["minimum_fixed_public_DSC"])
        or nsd < float(qualification["minimum_fixed_public_NSD"])
        or dsc - float(incumbent["Lesion_DSC"])
        < float(qualification["minimum_DSC_gain_over_incumbent"])
        or nsd - float(incumbent["Lesion_NSD"])
        < float(qualification["minimum_NSD_gain_over_incumbent"])
    ):
        raise ValueError(f"selected {spec.name} candidate does not pass frozen gates")

    checkpoint_name = (
        "checkpoint_final.pth"
        if epoch == spec.final_epoch
        else f"checkpoint_epoch_{epoch:04d}.pth"
    )
    expected_checkpoint = (spec.model_dir / "fold_all" / checkpoint_name).resolve()
    checkpoint_asset = selected.get("checkpoint", {})
    checkpoint = Path(checkpoint_asset.get("path", ""))
    if (
        checkpoint.resolve() != expected_checkpoint
        or not checkpoint.is_file()
        or checkpoint_asset.get("sha256") != sha256_file(checkpoint)
        or not spec.predictor.is_file()
    ):
        raise ValueError(f"selected {spec.name} checkpoint identity mismatch")
    return GlobalCandidate(
        family=spec.name,
        trainer=spec.trainer,
        predictor=str(spec.predictor.resolve()),
        model_dir=str(spec.model_dir.resolve()),
        epoch=epoch,
        deployment=deployment,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_asset["sha256"],
        fixed_public_DSC=dsc,
        fixed_public_NSD=nsd,
        public_terminal=str(spec.terminal.resolve()),
        public_terminal_sha256=sha256_file(spec.terminal),
        policy=str(spec.policy.resolve()),
        policy_sha256=sha256_file(spec.policy),
    )


def select_global_candidate(specs: Iterable[FamilySpec]) -> GlobalCandidate | None:
    candidates = [candidate for spec in specs if (candidate := validate_family_selection(spec))]
    return max(
        candidates,
        key=lambda candidate: (
            candidate.fixed_public_DSC,
            candidate.fixed_public_NSD,
        ),
        default=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    selected = select_global_candidate(family_specs(args.project_root.resolve()))
    print(json.dumps(None if selected is None else asdict(selected), sort_keys=True))


if __name__ == "__main__":
    main()
