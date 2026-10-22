#!/usr/bin/env python3
"""Read-only real-data smoke for the matched component-box data pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_pancancer_stunet  # noqa: E402,F401 - installs trainer resolver
from nnunetv2.run.run_training import get_trainer_from_args  # noqa: E402


def main() -> None:
    trainer = get_trainer_from_args(
        "Dataset501_PanCancer",
        "3d_fullres_STUNetB_PartialBox",
        "all",
        "STUNetTrainerPanCancerComponentBox",
        "nnUNetPlans",
        device=torch.device("cpu"),
    )
    # initialize() constructs the 58.5M-parameter network, which is irrelevant
    # to this data-only smoke. nnU-Net normally sets this value there.
    trainer.batch_size = trainer.configuration_manager.batch_size
    train, validation = trainer.get_dataloaders()
    try:
        batch = next(train)
        converted = trainer._partial_targets(batch["target"])
        full_target = converted[0] if isinstance(converted, list) else converted
        payload = {
            "status": "PASS",
            "dataset_role": "Train-Labeled preprocessed data only",
            "train_augmenter_parent": type(train).__mro__[1].__name__,
            "train_seeds": train.seeds,
            "validation_seeds": validation.seeds,
            "data_shape": list(batch["data"].shape),
            "target_shapes": [list(item.shape) for item in batch["target"]],
            "trusted_voxels_full": int((full_target != 2).sum()),
            "positive_voxels_full": int((full_target == 1).sum()),
            "public_read": False,
            "hidden_read": False,
        }
        expected_train = [2_026_082_600 + index for index in range(len(train.seeds))]
        expected_validation = [
            2_026_092_600 + index for index in range(len(validation.seeds))
        ]
        if train.seeds != expected_train or validation.seeds != expected_validation:
            raise RuntimeError(f"unexpected matched seeds: {payload}")
        if payload["train_augmenter_parent"] != "MultiThreadedAugmenter":
            raise RuntimeError(f"unordered augmenter: {payload}")
        if payload["positive_voxels_full"] <= 0 or payload["trusted_voxels_full"] <= 0:
            raise RuntimeError(f"empty transformed positive batch: {payload}")
        print(json.dumps(payload, indent=2), flush=True)
    finally:
        train._finish()
        validation._finish()


if __name__ == "__main__":
    main()
