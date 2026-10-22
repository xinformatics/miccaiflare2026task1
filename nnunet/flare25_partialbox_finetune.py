"""Partial-label-safe fine-tuning of the official FLARE25 whole-volume model.

The released checkpoint was trained with a six-stage PlainConvUNet, despite
the legacy custom configuration name in its plans.  Dataset501 already has
the images preprocessed at the checkpoint's 2.4 x 1.6 x 1.6 mm spacing and
128 x 160 x 160 patch geometry.  This trainer injects the released network
architecture while reusing that existing cache, so no second 200-GB
preprocessing copy is required.
"""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler


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


class WarmupCosineScheduler(_LRScheduler):
    """Five-epoch linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer,
        peak_lr: float,
        total_epochs: int,
        warmup_epochs: int = 5,
        start_lr: float = 1e-5,
        final_lr: float = 3e-5,
    ):
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.start_lr = float(start_lr)
        self.final_lr = float(final_lr)
        self.ctr = 0
        super().__init__(optimizer, -1)

    def _lr_at(self, epoch: int) -> float:
        epoch = max(0, min(int(epoch), self.total_epochs))
        if epoch < self.warmup_epochs:
            fraction = epoch / max(1, self.warmup_epochs - 1)
            return self.start_lr + fraction * (self.peak_lr - self.start_lr)
        denominator = max(1, self.total_epochs - self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / denominator
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr + (self.peak_lr - self.final_lr) * cosine

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        new_lr = self._lr_at(int(current_step))
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = new_lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


class nnUNetTrainerFLARE25PartialBoxFineTune(
    STUNetTrainerPanCancerComponentBoxE160SourceStratified
):
    """Adapt the official FLARE25 network to all 17,575 partial-label cases."""

    configuration_name = "3d_fullres_FLARE25FT"
    source_cache_configuration = "3d_fullres_S4D2W32"
    expected_parameter_count = 31_195_594
    expected_global_batch_size = 4
    training_horizon_epochs = 50
    landmark_epochs = (10, 25, 50)
    pretrained_checkpoint_sha256 = (
        "c3f136cf916ea135025dcd43742e8e42ce6fcb6c9aab7802522bd8fb3e3b6941"
    )

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=torch.device("cuda"),
    ):
        if configuration != self.configuration_name:
            raise ValueError(
                f"FLARE25 fine-tuning requires {self.configuration_name}, "
                f"found {configuration}"
            )
        plans = copy.deepcopy(plans)
        fullres = copy.deepcopy(plans["configurations"]["3d_fullres"])
        cache = plans["configurations"][self.source_cache_configuration]
        for key in (
            "data_identifier",
            "preprocessor_name",
            "patch_size",
            "median_image_size_in_voxels",
            "spacing",
            "normalization_schemes",
            "use_mask_for_norm",
            "resampling_fn_data",
            "resampling_fn_seg",
            "resampling_fn_data_kwargs",
            "resampling_fn_seg_kwargs",
            "resampling_fn_probabilities",
            "resampling_fn_probabilities_kwargs",
        ):
            fullres[key] = copy.deepcopy(cache[key])
        # Dataset501's ordinary 3d_fullres plan uses anisotropic kernels and
        # strides. The released FLARE25 checkpoint instead used six isotropic
        # stages; preserve those exact tensor shapes while retaining the
        # already-cached 2.4 x 1.6 x 1.6-mm arrays.
        architecture = fullres["architecture"]["arch_kwargs"]
        architecture["kernel_sizes"] = [[3, 3, 3]] * 6
        architecture["strides"] = [[1, 1, 1]] + [[2, 2, 2]] * 5
        fullres["batch_size"] = self.expected_global_batch_size
        # The released checkpoint uses non-batch Dice in its original plan.
        fullres["batch_dice"] = False
        plans["configurations"][configuration] = fullres

        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = 3e-4
        self.num_epochs = self.training_horizon_epochs
        self.save_every = 1
        self.oversample_foreground_percent = 1.0
        self.inference_checkpoint_start_epoch = self.num_epochs + 1
        self._flare25_wandb_run = None

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if not architecture_class_name.endswith("PlainConvUNet"):
            raise ValueError(
                "FLARE25 fine-tuning received an incompatible architecture: "
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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            peak_lr=self.initial_lr,
            total_epochs=self.num_epochs,
            warmup_epochs=5,
            start_lr=1e-5,
            final_lr=3e-5,
        )
        return optimizer, scheduler

    def on_train_start(self) -> None:
        # Keep this class's loader, partial targets and loss, but avoid the STU
        # continuation's W&B identity and checkpoint side effects.
        nnUNetTrainer.on_train_start(self)
        if not self.is_ddp or dist.get_world_size() != 2:
            raise RuntimeError("FLARE25 production fine-tuning requires exactly 2 GPUs")
        if self.configuration_manager.batch_size != self.expected_global_batch_size:
            raise RuntimeError("FLARE25 global batch-size contract changed")
        parameter_source = self.network.module if self.is_ddp else self.network
        parameter_source = getattr(parameter_source, "_orig_mod", parameter_source)
        parameter_count = sum(
            parameter.numel() for parameter in parameter_source.parameters()
        )
        if parameter_count != self.expected_parameter_count:
            raise RuntimeError(
                f"FLARE25 parameter count changed: {parameter_count} != "
                f"{self.expected_parameter_count}"
            )
        if self.local_rank != 0:
            return
        run_name = "flare25_partialbox_all17575_warmcos50_20260828"
        run_id = stable_wandb_run_id(self.output_folder, run_name)
        self._flare25_wandb_run = safe_wandb_init(
            project=os.environ.get("WANDB_PROJECT", "pancancer-ct"),
            name=run_name,
            id=run_id,
            resume="allow",
            mode=os.environ.get("WANDB_MODE", "online"),
            config={
                "architecture": "FLARE25_PlainConvUNet_6stage",
                "parameters": parameter_count,
                "initialization": "official_FLARE25_checkpoint_network_all_layers",
                "pretrained_checkpoint_sha256": self.pretrained_checkpoint_sha256,
                "training_cases": self.expected_case_count,
                "global_batch_size": self.expected_global_batch_size,
                "spacing_mm": list(self.configuration_manager.spacing),
                "patch_size": list(self.configuration_manager.patch_size),
                "optimizer": "SGD_Nesterov_momentum_0.99",
                "weight_decay": self.weight_decay,
                "peak_lr": self.initial_lr,
                "lr_schedule": "linear_warmup5_1e-5_to3e-4_cosine_to3e-5",
                "training_horizon_epochs": self.num_epochs,
                "landmark_epochs": list(self.landmark_epochs),
                "partial_label_policy": (
                    "ignore_outside_union_of_16mm_component_boxes"
                ),
                "loss": "trusted_component_box_CE_plus_Tversky_a025_b075",
                "case_sampling": "75pct_uniform_plus25pct_source_size_balanced",
                "component_center_fraction": self.component_center_fraction,
                "public_used_for_training": False,
                "hidden_used_for_training": False,
            },
        )

    def on_epoch_end(self) -> None:
        completed_epoch = self.current_epoch + 1
        metrics = self.logger.my_fantastic_logging
        if completed_epoch in self.landmark_epochs:
            self.save_checkpoint(
                os.path.join(
                    self.output_folder,
                    f"checkpoint_epoch_{completed_epoch:04d}.pth",
                )
            )
        nnUNetTrainer.on_epoch_end(self)
        if self.local_rank == 0:
            self._flare25_wandb_run = safe_wandb_log(
                self._flare25_wandb_run,
                {
                    "epoch": completed_epoch,
                    "train_loss": float(metrics["train_losses"][-1]),
                    "val_loss": float(metrics["val_losses"][-1]),
                    "ema_fg_dice": float(metrics["ema_fg_dice"][-1]),
                    "learning_rate": float(metrics["lrs"][-1]),
                },
                step=completed_epoch,
            )

    def on_train_end(self) -> None:
        nnUNetTrainer.on_train_end(self)
        if self.local_rank == 0:
            safe_wandb_finish(self._flare25_wandb_run)
            self._flare25_wandb_run = None
