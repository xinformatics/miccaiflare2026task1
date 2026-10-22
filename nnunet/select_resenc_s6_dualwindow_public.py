#!/usr/bin/env python3
"""Select a dual-window ResEnc endpoint under its frozen public policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import nnunet.select_resenc_s6_public as base


EXPECTED_EPOCHS = [25, 50, 100]
MODEL_NAME = "nnUNetTrainerResEncS6PanCancerDualWindow"


def load_policy(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = [item["name"] for item in payload.get("deployments", [])]
    if (
        payload.get("status") != "PREDECLARED_BEFORE_TRAINING"
        or payload.get("model") != MODEL_NAME
        or payload.get("checkpoint_epochs") != EXPECTED_EPOCHS
        or names != list(base.ANALYSIS_TO_POLICY.values())
        or payload.get("threshold_search_after_public_read") is not False
        or payload.get("public_used_for_training")
        or payload.get("hidden_read")
    ):
        raise ValueError("invalid dual-window ResEnc public policy")
    return payload


def select(
    policy_path: Path,
    analyses: Mapping[int, Path],
    checkpoints: Mapping[int, Path],
) -> dict:
    return base.select(
        policy_path,
        dict(analyses),
        dict(checkpoints),
        policy_loader=load_policy,
    )


def main() -> None:
    base.main(select_fn=select)


if __name__ == "__main__":
    main()

