"""Prospective ResEnc-S6 continuation with shape-following 12-mm supervision."""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np
import torch
import torch.distributed as dist
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunet.partial_region_targets import shape_linf_band_target
from nnunet.resenc_s6_partialbox import nnUNetTrainerResEncS6PanCancerPartialBox
from fault_tolerant_wandb import safe_wandb_init, stable_wandb_run_id


BAND_MM = 12.0
BAND_STAGE_END_EPOCHS = (25, 50, 100)
BAND_TRAINING_HORIZON = 100
BAND_INITIAL_LR = 1e-4
ALLOWED_PARENT_EPOCHS = (100, 200, 300, 400, 500)


def validated_parent_contract(environment: Mapping[str, str]) -> tuple[int, str]:
    try:
        epoch = int(environment["RESENC_BAND12_PARENT_EPOCH"])
        sha256 = environment["RESENC_BAND12_PARENT_SHA256"].lower()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("shape-band continuation requires a parent epoch and SHA256") from error
    if epoch not in ALLOWED_PARENT_EPOCHS:
        raise ValueError(f"invalid shape-band parent epoch: {epoch}")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError("shape-band parent SHA256 must be 64 lowercase hex characters")
    return epoch, sha256


class nnUNetTrainerResEncS6PanCancerPartialBand12(
    nnUNetTrainerResEncS6PanCancerPartialBox
):
    """Fine-tune a selected trained S6 parent with a 12-mm lesion-shape band."""

    landmark_epochs = (25, 50)
    band_stage_end_epochs = BAND_STAGE_END_EPOCHS
    band_training_horizon = BAND_TRAINING_HORIZON
    trusted_band_mm = BAND_MM

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = BAND_INITIAL_LR
        stage_end_epoch = int(
            os.environ.get("RESENC_BAND12_STAGE_END_EPOCH", BAND_TRAINING_HORIZON)
        )
        if stage_end_epoch not in self.band_stage_end_epochs:
            raise ValueError(
                "RESENC_BAND12_STAGE_END_EPOCH must be one of "
                f"{self.band_stage_end_epochs}, found {stage_end_epoch}"
            )
        self.num_epochs = stage_end_epoch
        self.lr_schedule_horizon_epochs = self.band_training_horizon
        self.save_every = 1
        self.inference_checkpoint_start_epoch = self.band_training_horizon + 1
        self.bbox_context_mm = self.trusted_band_mm
        spacing = tuple(float(value) for value in self.configuration_manager.spacing)
        self.full_margin_voxels = tuple(
            max(1, int(np.ceil(self.trusted_band_mm / value))) for value in spacing
        )

    def _partial_targets(self, target):
        targets = target if isinstance(target, list) else [target]
        full_shape = tuple(int(value) for value in targets[0].shape[2:])
        converted = [
            shape_linf_band_target(item, full_shape, self.full_margin_voxels)
            for item in targets
        ]
        return converted if isinstance(target, list) else converted[0]

    def on_train_start(self) -> None:
        # Bypass the base W&B identity while retaining its architecture,
        # source/size sampler, component-uniform crop centers, and loss.
        nnUNetTrainer.on_train_start(self)
        parent_epoch, parent_sha256 = validated_parent_contract(os.environ)
        if not self.is_ddp or dist.get_world_size() != 2:
            raise RuntimeError("shape-band S6 continuation requires exactly 2 GPUs")
        if self.configuration_manager.batch_size != self.expected_global_batch_size:
            raise RuntimeError("shape-band S6 global batch-size contract changed")
        parameter_source = self.network.module if self.is_ddp else self.network
        parameter_source = getattr(parameter_source, "_orig_mod", parameter_source)
        parameter_count = sum(parameter.numel() for parameter in parameter_source.parameters())
        if parameter_count != self.expected_parameter_count:
            raise RuntimeError("shape-band S6 parameter count changed")
        if tuple(self.full_margin_voxels) != (5, 8, 8):
            raise RuntimeError(
                f"shape-band S6 margin changed: {self.full_margin_voxels}"
            )
        if self.local_rank != 0:
            return
        run_name = f"resenc_s6_partialband12_parent{parent_epoch}_100ep_20260827"
        run_id = stable_wandb_run_id(self.output_folder, run_name)
        self._resenc_wandb_run = safe_wandb_init(
            project=os.environ.get("WANDB_PROJECT", "pancancer-ct"),
            name=run_name,
            id=run_id,
            resume="allow",
            mode=os.environ.get("WANDB_MODE", "online"),
            config={
                "architecture": self.architecture_name,
                "expected_parameters": self.expected_parameter_count,
                "global_batch_size": self.expected_global_batch_size,
                "initialization": "selected_base_parent_weights_fresh_optimizer",
                "parent_epoch": parent_epoch,
                "parent_checkpoint_sha256": parent_sha256,
                "configuration": self.configuration_name,
                "spacing_mm": list(self.configuration_manager.spacing),
                "patch_size": list(self.configuration_manager.patch_size),
                "initial_lr": self.initial_lr,
                "training_horizon_epochs": self.band_training_horizon,
                "staged_endpoint_epochs": list(self.band_stage_end_epochs),
                "optimizer": "SGD_Nesterov_momentum_0.99_fresh",
                "lr_schedule": "polynomial_100_epochs",
                "partial_label_policy": "shape_linf_band_12mm",
                "full_margin_voxels_zyx": list(self.full_margin_voxels),
                "loss": "trusted_shape_band_CE_plus_Tversky_a025_b075",
                "case_sampling": "75pct_uniform_plus_25pct_source_size_balanced",
                "component_center_fraction": self.component_center_fraction,
                "foreground_oversample_fraction": self.oversample_foreground_percent,
                "training_cases": self.expected_case_count,
                "training_sources": self.expected_source_count,
                "public_used_for_training": False,
                "hidden_used_for_training": False,
            },
        )
