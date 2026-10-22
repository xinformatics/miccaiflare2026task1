#!/usr/bin/env python3
"""Select a positive-unlabeled ResEnc endpoint under its frozen policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import nnunet.select_resenc_s6_public as base


EXPECTED_EPOCHS = [25, 50, 100]


def load_policy(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = [item["name"] for item in payload.get("deployments", [])]
    if (
        payload.get("status") != "PREDECLARED_BEFORE_TRAINING"
        or payload.get("model")
        != "nnUNetTrainerResEncS6PanCancerPositiveUnlabeled"
        or payload.get("checkpoint_epochs") != EXPECTED_EPOCHS
        or names != list(base.ANALYSIS_TO_POLICY.values())
        or payload.get("threshold_search_after_public_read") is not False
        or payload.get("public_used_for_training")
        or payload.get("hidden_read")
    ):
        raise ValueError("invalid positive-unlabeled ResEnc public policy")
    return payload


def select(
    policy_path: Path,
    analyses: Mapping[int, Path],
    checkpoints: Mapping[int, Path],
) -> dict:
    """Run the shared selector with this branch's policy validator.

    Passing the loader explicitly avoids mutating the base selector module and
    keeps base/positive selection deterministic regardless of import order.
    """
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
