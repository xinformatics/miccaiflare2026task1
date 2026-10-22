"""Low-drift FLARE25 adaptation with independent Train-Unlabel evidence."""

from __future__ import annotations

import importlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.optim.lr_scheduler import _LRScheduler

from nnunet.flare25_partialbox_finetune import nnUNetTrainerFLARE25PartialBoxFineTune
from nnunet.resenc_s6_positive_unlabeled import (
    positive_unlabeled_loader_factory,
    unwrap_loader,
)
from STUNetTrainerPanCancer import PartialBoxTverskyCELoss, STUNetTrainerPanCancer
from STUNetTrainerPanCancerSourceBalanced import (
    foreground_proxy_counts_from_dataset,
    identifier_inventory_sha256,
    source_from_identifier,
)
from fault_tolerant_wandb import (
    safe_wandb_finish,
    safe_wandb_init,
    safe_wandb_log,
    stable_wandb_run_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "Dataset504_FLARE25MultiTeacher"
BUILD_MARKER = (
    PROJECT_ROOT / "nnUNet_preprocessed" / DATASET_NAME / "DATASET_BUILD_COMPLETE.json"
)
PSEUDO_PREFIX = "Pseudo-"
PSEUDO_EXPOSURE = 0.10


def load_dataset_contract(path: Path = BUILD_MARKER) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "COMPLETE",
        "target_dataset": DATASET_NAME,
        "labeled_cases": 17_575,
        "public_read": False,
        "hidden_read": False,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"Dataset504 contract mismatch for {key}")
    pseudo_cases = int(payload.get("pseudo_cases", 0))
    if pseudo_cases < 500 or payload.get("total_cases") != 17_575 + pseudo_cases:
        raise ValueError("Dataset504 pseudo inventory is implausible")
    manifest = payload.get("manifest", {})
    manifest_path = Path(manifest.get("path", ""))
    if not manifest_path.is_file():
        raise ValueError("Dataset504 manifest is missing")
    details = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = details.get("training", [])
    if len(rows) != pseudo_cases:
        raise ValueError("Dataset504 manifest count changed")
    payload["pseudo_identifiers"] = tuple(sorted(row["identifier"] for row in rows))
    return payload


class FrozenEncoderCosineScheduler(_LRScheduler):
    """Decoder-only stabilization followed by differential cosine decay."""

    def __init__(
        self,
        optimizer,
        total_epochs: int = 100,
        frozen_epochs: int = 10,
        encoder_peak_lr: float = 1e-5,
        decoder_peak_lr: float = 3e-5,
        encoder_final_lr: float = 1e-6,
        decoder_final_lr: float = 3e-6,
    ):
        if len(optimizer.param_groups) != 2:
            raise ValueError("differential schedule requires encoder and decoder groups")
        self.total_epochs = int(total_epochs)
        self.frozen_epochs = int(frozen_epochs)
        self.encoder_peak_lr = float(encoder_peak_lr)
        self.decoder_peak_lr = float(decoder_peak_lr)
        self.encoder_final_lr = float(encoder_final_lr)
        self.decoder_final_lr = float(decoder_final_lr)
        self.ctr = 0
        super().__init__(optimizer, -1)

    def rates(self, epoch: int) -> tuple[float, float]:
        epoch = max(0, min(int(epoch), self.total_epochs))
        if epoch < self.frozen_epochs:
            return 0.0, 1e-5
        progress = (epoch - self.frozen_epochs) / max(
            1, self.total_epochs - self.frozen_epochs
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        encoder = self.encoder_final_lr + (
            self.encoder_peak_lr - self.encoder_final_lr
        ) * cosine
        decoder = self.decoder_final_lr + (
            self.decoder_peak_lr - self.decoder_final_lr
        ) * cosine
        return encoder, decoder

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        rates = self.rates(int(current_step))
        for group, rate in zip(self.optimizer.param_groups, rates, strict=True):
            group["lr"] = rate
        self._last_lr = list(rates)

    def get_last_lr(self):
        return self._last_lr


class nnUNetTrainerFLARE25MultiTeacherDistill(
    nnUNetTrainerFLARE25PartialBoxFineTune
):
    """Adapt the released FLARE25 model without treating missing lesions negative."""

    configuration_name = "3d_fullres_FLARE25MT"
    expected_global_batch_size = 8
    training_horizon_epochs = 100
    landmark_epochs = (10, 25, 50, 75, 100)
    stage_end_epochs = landmark_epochs
    component_center_fraction = 0.25

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        contract = load_dataset_contract()
        self._multiteacher_contract = contract
        self.expected_case_count = int(contract["total_cases"])
        self.expected_pseudo_cases = int(contract["pseudo_cases"])
        self.expected_pseudo_identifiers = tuple(contract["pseudo_identifiers"])
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        stage_end = int(
            os.environ.get("FLARE25_MT_STAGE_END_EPOCH", self.training_horizon_epochs)
        )
        if stage_end not in self.stage_end_epochs:
            raise ValueError(
                f"FLARE25_MT_STAGE_END_EPOCH must be one of {self.stage_end_epochs}"
            )
        self.num_epochs = stage_end
        self.save_every = 1
        self.oversample_foreground_percent = 1.0
        self.inference_checkpoint_start_epoch = self.num_epochs + 1
        self._flare25_wandb_run = None

    def configure_optimizers(self):
        encoder_parameters = list(self.network.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        decoder_parameters = [
            parameter
            for parameter in self.network.parameters()
            if id(parameter) not in encoder_ids
        ]
        if not encoder_parameters or not decoder_parameters:
            raise ValueError("FLARE25 encoder/decoder parameter split failed")
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": 0.0, "name": "encoder"},
                {"params": decoder_parameters, "lr": 1e-5, "name": "decoder"},
            ],
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        return optimizer, FrozenEncoderCosineScheduler(
            optimizer, total_epochs=self.training_horizon_epochs
        )

    def _build_loss(self):
        loss = PartialBoxTverskyCELoss(alpha=0.35, beta=0.65)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.asarray([1.0 / (2**index) for index in range(len(scales))])
            weights[-1] = 1e-6 if self.is_ddp else 0.0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _validated_counts(self, data) -> np.ndarray:
        identifiers = tuple(str(item) for item in data.identifiers)
        if len(identifiers) != self.expected_case_count:
            raise ValueError("Dataset504 case count changed")
        pseudo = tuple(sorted(item for item in identifiers if item.startswith(PSEUDO_PREFIX)))
        if pseudo != self.expected_pseudo_identifiers:
            raise ValueError("Dataset504 pseudo identifier inventory changed")
        labeled = [item for item in identifiers if not item.startswith(PSEUDO_PREFIX)]
        if len(labeled) != 17_575:
            raise ValueError("Dataset504 labeled inventory changed")
        return foreground_proxy_counts_from_dataset(data)

    def get_dataloaders(self):
        trainer_module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        original_loader = trainer_module.nnUNetDataLoader
        trainer_module.nnUNetDataLoader = positive_unlabeled_loader_factory(
            original_loader,
            self.component_center_fraction,
            self._validated_counts,
            PSEUDO_EXPOSURE,
        )
        try:
            loaders = STUNetTrainerPanCancer.get_dataloaders(self)
        finally:
            trainer_module.nnUNetDataLoader = original_loader
        training_loader = unwrap_loader(loaders[0])
        probabilities = np.asarray(training_loader.sampling_probabilities)
        identifiers = tuple(training_loader._data.identifiers)
        pseudo = np.asarray(
            [item.startswith(PSEUDO_PREFIX) for item in identifiers], dtype=bool
        )
        if not np.isclose(probabilities[pseudo].sum(), PSEUDO_EXPOSURE, atol=1e-12):
            raise ValueError("Dataset504 pseudo exposure changed")
        return loaders

    def on_train_start(self) -> None:
        nnUNetTrainer.on_train_start(self)
        if not self.is_ddp or dist.get_world_size() != 2:
            raise RuntimeError("FLARE25 multi-teacher training requires exactly 2 GPUs")
        if self.configuration_manager.batch_size != self.expected_global_batch_size:
            raise RuntimeError("FLARE25 multi-teacher global batch size changed")
        parameter_source = self.network.module if self.is_ddp else self.network
        parameter_source = getattr(parameter_source, "_orig_mod", parameter_source)
        parameter_count = sum(parameter.numel() for parameter in parameter_source.parameters())
        if parameter_count != self.expected_parameter_count:
            raise RuntimeError("FLARE25 architecture parameter count changed")
        if self.local_rank != 0:
            return
        run_name = "flare25_multiteacher_lowdrift_100ep_20260830"
        self._flare25_wandb_run = safe_wandb_init(
            project=os.environ.get("WANDB_PROJECT", "pancancer-ct"),
            name=run_name,
            id=stable_wandb_run_id(self.output_folder, run_name),
            resume="allow",
            mode=os.environ.get("WANDB_MODE", "online"),
            config={
                "architecture": "official_FLARE25_PlainConvUNet_31M",
                "initialization": "exact_official_FLARE25_all_tensors",
                "training_cases": self.expected_case_count,
                "labeled_cases": 17_575,
                "multiteacher_pseudo_cases": self.expected_pseudo_cases,
                "pseudo_exposure": PSEUDO_EXPOSURE,
                "global_batch_size": self.expected_global_batch_size,
                "optimizer": "AdamW_betas_0.9_0.999_weight_decay_1e-4",
                "schedule": "encoder_lr0_10ep_then_1e-5;decoder_1e-5_then_3e-5;cosine",
                "loss": "component_box_CE_plus_Tversky_a035_b065",
                "unknown_policy": "outside_component_boxes_and_disputed_regions_ignored",
                "teacher_policy": self._multiteacher_contract["policy"],
                "landmark_epochs": list(self.landmark_epochs),
                "public_used_for_training": False,
                "hidden_used_for_training": False,
            },
        )

    def on_epoch_end(self) -> None:
        completed_epoch = self.current_epoch + 1
        metrics = self.logger.my_fantastic_logging
        if completed_epoch in self.landmark_epochs:
            self.save_checkpoint(
                os.path.join(self.output_folder, f"checkpoint_epoch_{completed_epoch:04d}.pth")
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
                    "encoder_lr": float(self.optimizer.param_groups[0]["lr"]),
                    "decoder_lr": float(self.optimizer.param_groups[1]["lr"]),
                },
                step=completed_epoch,
            )

    def on_train_end(self) -> None:
        nnUNetTrainer.on_train_end(self)
        if self.local_rank == 0:
            safe_wandb_finish(self._flare25_wandb_run)
            self._flare25_wandb_run = None


class nnUNetTrainerFLARE25MultiTeacherSmoke(
    nnUNetTrainerFLARE25MultiTeacherDistill
):
    """Two-step DDP, loader, loss, and batch-8 memory preflight."""

    configuration_name = "3d_fullres_FLARE25MTSmoke"
    landmark_epochs = ()

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
        self.save_every = 1

    def on_train_start(self) -> None:
        nnUNetTrainer.on_train_start(self)
        if not self.is_ddp or dist.get_world_size() != 2:
            raise RuntimeError("FLARE25 multi-teacher smoke requires two GPUs")
        if self.configuration_manager.batch_size != self.expected_global_batch_size:
            raise RuntimeError("FLARE25 multi-teacher smoke batch size changed")

    def on_epoch_end(self) -> None:
        nnUNetTrainer.on_epoch_end(self)

    def on_train_end(self) -> None:
        nnUNetTrainer.on_train_end(self)
