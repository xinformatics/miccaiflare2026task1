"""Partial-label-safe nnU-Net ResEnc S6 trainer for pan-cancer CT.

The historical S6 experiment used stock complete-mask supervision even though
the challenge masks annotate primarily one lesion.  This trainer retains the
planned 107M-parameter ResidualEncoderUNet, but imports the audited component-
box positive-unlabeled target, bounded source/size sampler, and mildly recall-
weighted Tversky+CE loss from the successful STU-Net training path.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANCANCER_ROOT = PROJECT_ROOT / "external" / "STU-Net" / "pancancer"
GF_ROOT = PROJECT_ROOT / "external" / "GF-Screen"
for source_root in (PANCANCER_ROOT, GF_ROOT):
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

from STUNetTrainerPanCancerSourceBalanced import (  # noqa: E402
    STUNetTrainerPanCancerComponentBoxE160SourceStratified,
)
from fault_tolerant_wandb import (  # noqa: E402
    safe_wandb_finish,
    safe_wandb_init,
    safe_wandb_log,
    stable_wandb_run_id,
)


class nnUNetTrainerResEncS6PanCancerPartialBox(
    STUNetTrainerPanCancerComponentBoxE160SourceStratified
):
    """Train ResEnc S6 from scratch without using missing labels as negatives."""

    architecture_name = "ResidualEncoderUNet-S6-W32"
    expected_parameter_count = 107_467_050
    expected_global_batch_size = 16
    landmark_epochs = (100, 200, 300, 400)
    stage_end_epochs = (100, 200, 300, 400, 500)
    training_horizon_epochs = 500

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=torch.device("cuda"),
    ):
        if configuration != "3d_fullres_ResEncS6":
            raise ValueError(
                "partial-box ResEnc trainer requires configuration "
                "3d_fullres_ResEncS6"
            )
        plans = copy.deepcopy(plans)
        plans["configurations"][configuration]["batch_size"] = (
            self.expected_global_batch_size
        )
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = 1e-3
        stage_end_epoch = int(
            os.environ.get("RESENC_STAGE_END_EPOCH", self.training_horizon_epochs)
        )
        if stage_end_epoch not in self.stage_end_epochs:
            raise ValueError(
                "RESENC_STAGE_END_EPOCH must be one of "
                f"{self.stage_end_epochs}, found {stage_end_epoch}"
            )
        # End the current process at a predeclared public endpoint so the GPUs
        # can be released for inference. The LR scheduler still spans the full
        # 500-epoch training horizon and resume checkpoints retain optimizer
        # momentum, so staging does not restart optimization.
        self.num_epochs = stage_end_epoch
        self.lr_schedule_horizon_epochs = self.training_horizon_epochs
        self.save_every = 1
        self.oversample_foreground_percent = 1.0
        # The inherited STU trainer otherwise writes a 430 MB inference-only
        # snapshot after every epoch. Full resumable latest checkpoints are
        # retained every epoch; immutable landmarks are limited to the fixed
        # 100-epoch endpoints below.
        self.inference_checkpoint_start_epoch = self.training_horizon_epochs + 1
        self._resenc_wandb_run = None

    def configure_optimizers(self):
        """Preserve the 500-epoch polynomial schedule across staged exits."""

        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = PolyLRScheduler(
            optimizer,
            self.initial_lr,
            self.lr_schedule_horizon_epochs,
        )
        return optimizer, scheduler

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Use the architecture encoded in nnUNetPlans, not STU-Net."""

        if not architecture_class_name.endswith("ResidualEncoderUNet"):
            raise ValueError(
                "ResEnc S6 trainer received a non-residual architecture: "
                f"{architecture_class_name}"
            )
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

    def on_train_start(self) -> None:
        # Call nnU-Net directly. Dynamic dispatch still uses this class's
        # source-stratified loader, partial targets, loss, and architecture,
        # while avoiding every continuation-specific STU W&B identity.
        nnUNetTrainer.on_train_start(self)
        if not self.is_ddp or dist.get_world_size() != 2:
            raise RuntimeError("ResEnc S6 production training requires exactly 2 GPUs")
        if self.configuration_manager.batch_size != self.expected_global_batch_size:
            raise RuntimeError("ResEnc S6 global batch-size contract changed")
        parameter_source = self.network.module if self.is_ddp else self.network
        parameter_source = getattr(parameter_source, "_orig_mod", parameter_source)
        parameter_count = sum(parameter.numel() for parameter in parameter_source.parameters())
        if parameter_count != self.expected_parameter_count:
            raise RuntimeError(
                f"ResEnc S6 parameter count changed: {parameter_count} != "
                f"{self.expected_parameter_count}"
            )
        if self.local_rank != 0:
            return
        run_name = "resenc_s6_partialbox_sourcesize_a025b075_500ep_20260827"
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
                "initialization": "from_scratch",
                "configuration": self.configuration_name,
                "spacing_mm": list(self.configuration_manager.spacing),
                "patch_size": list(self.configuration_manager.patch_size),
                "initial_lr": self.initial_lr,
                "training_horizon_epochs": self.training_horizon_epochs,
                "staged_endpoint_epochs": list(self.stage_end_epochs),
                "optimizer": "SGD_Nesterov_momentum_0.99",
                "lr_schedule": "polynomial_500_epochs",
                "partial_label_policy": (
                    "ignore_outside_union_of_16mm_component_boxes"
                ),
                "loss": "trusted_component_box_CE_plus_Tversky_a025_b075",
                "case_sampling": "75pct_uniform_plus_25pct_source_size_balanced",
                "staged_augmentation_seed_policy": (
                    "base_2026082600_plus_stage_start_epoch_times_1000000_"
                    "plus_ddp_rank_times_100000"
                ),
                "component_center_fraction": self.component_center_fraction,
                "foreground_oversample_fraction": (
                    self.oversample_foreground_percent
                ),
                "training_cases": self.expected_case_count,
                "training_sources": self.expected_source_count,
                "public_used_for_training": False,
                "hidden_used_for_training": False,
            },
        )

    def on_epoch_end(self) -> None:
        completed_epoch = self.current_epoch + 1
        metrics = self.logger.my_fantastic_logging
        payload = {
            "epoch": completed_epoch,
            "train_loss": float(metrics["train_losses"][-1]),
            "val_loss": float(metrics["val_losses"][-1]),
            "ema_fg_dice": float(metrics["ema_fg_dice"][-1]),
            "learning_rate": float(metrics["lrs"][-1]),
        }
        if completed_epoch in self.landmark_epochs:
            self.save_checkpoint(
                os.path.join(
                    self.output_folder,
                    f"checkpoint_epoch_{completed_epoch:04d}.pth",
                )
            )
        # Bypass STU's every-ten-epoch archival and inference-only snapshots.
        # nnU-Net writes checkpoint_latest every epoch and maintains best.
        nnUNetTrainer.on_epoch_end(self)
        if self.local_rank == 0:
            self._resenc_wandb_run = safe_wandb_log(
                self._resenc_wandb_run,
                payload,
                step=completed_epoch,
            )

    def on_train_end(self) -> None:
        nnUNetTrainer.on_train_end(self)
        if self.local_rank == 0:
            safe_wandb_finish(self._resenc_wandb_run)
            self._resenc_wandb_run = None
