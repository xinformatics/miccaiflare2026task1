"""Partial-label-safe STU-Net-B trainer for FLARE pan-cancer CT.

The architecture is loaded directly from the pinned upstream STU-Net source so
the network remains byte-for-byte faithful to commit
0652eaf5cafda6bc269942238c3695994f2dc4f0. The trainer interface is adapted to
nnU-Net 2.6.x without editing the installed package.

This is intentionally a recall/proposal model. Every sampled patch is centred
on annotated foreground. Background is supervised only inside a 16-mm context
box around the annotation; the rest is ignored because Train-Labeled masks may
omit secondary lesions.
"""

from __future__ import annotations

import importlib.util
import importlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from scipy import ndimage
from torch import nn


UPSTREAM_COMMIT = "0652eaf5cafda6bc269942238c3695994f2dc4f0"
IGNORE_LABEL = 2


def matched_augmentation_seed_base(
    augmentation_seed_offset: int,
    local_rank: int,
    augmenter_index: int,
) -> int:
    """Return a reproducible seed base unique to stage, DDP rank and loader."""

    values = {
        "augmentation_seed_offset": int(augmentation_seed_offset),
        "local_rank": int(local_rank),
        "augmenter_index": int(augmenter_index),
    }
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    seed = (
        2_026_082_600
        + values["augmentation_seed_offset"]
        + values["local_rank"] * 100_000
        + values["augmenter_index"] * 10_000
    )
    if seed > np.iinfo(np.uint32).max:
        raise ValueError("augmentation seed exceeds the uint32 range")
    return seed


def _load_pinned_stunet_class():
    source = (
        Path(__file__).resolve().parents[1]
        / "nnUNet-2.2"
        / "nnunetv2"
        / "training"
        / "nnUNetTrainer"
        / "STUNetTrainer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pinned_upstream_stunet_architecture", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned STU-Net source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STUNet


STUNet = _load_pinned_stunet_class()


class PartialBoxTverskyCELoss(nn.Module):
    """Case-balanced Tversky+CE over trusted component-box voxels only."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, ignore_label: int = 2):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ignore_label = int(ignore_label)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or logits.shape[1] != 2:
            raise ValueError("STU-Net pan-cancer logits must be [B,2,D,H,W]")
        if target.ndim != 5 or target.shape[1] != 1:
            raise ValueError("STU-Net pan-cancer target must be [B,1,D,H,W]")
        target_labels = target[:, 0].long()
        valid = target_labels != self.ignore_label
        binary = target_labels.clamp(min=0, max=1)

        ce_voxels = F.cross_entropy(logits, binary, reduction="none")
        valid_count = valid.sum()
        if valid_count.item() > 0:
            ce = (ce_voxels * valid).sum() / valid_count
        else:
            ce = logits.sum() * 0.0

        probability = logits.softmax(dim=1)[:, 1]
        foreground = binary.float()
        axes = tuple(range(1, probability.ndim))
        valid_float = valid.float()
        true_positive = (probability * foreground * valid_float).sum(axes)
        false_positive = (probability * (1.0 - foreground) * valid_float).sum(axes)
        false_negative = ((1.0 - probability) * foreground * valid_float).sum(axes)
        denominator = (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
        )
        tversky = (true_positive + 1e-6) / (denominator + 1e-6)
        valid_samples = valid.flatten(1).any(1)
        if valid_samples.any():
            tversky_loss = 1.0 - tversky[valid_samples].mean()
        else:
            tversky_loss = logits.sum() * 0.0
        return ce + tversky_loss


def partial_box_target(
    target: torch.Tensor,
    full_shape: tuple[int, int, int],
    full_margin_voxels: tuple[int, int, int],
) -> torch.Tensor:
    """Set unknown background outside an expanded foreground box to ignore."""

    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError("target must be [B,1,D,H,W]")
    spatial_shape = target.shape[2:]
    scaled_margin = tuple(
        max(1, int(np.ceil(margin * size / full)))
        for margin, size, full in zip(
            full_margin_voxels, spatial_shape, full_shape, strict=True
        )
    )
    result = torch.full_like(target, IGNORE_LABEL)
    for batch_index in range(target.shape[0]):
        source = target[batch_index, 0]
        coordinates = torch.where(source > 0)
        if coordinates[0].numel() == 0:
            continue
        lower = [
            max(0, int(axis.min().item()) - margin)
            for axis, margin in zip(coordinates, scaled_margin, strict=True)
        ]
        upper = [
            min(size, int(axis.max().item()) + margin + 1)
            for axis, margin, size in zip(
                coordinates, scaled_margin, spatial_shape, strict=True
            )
        ]
        region = tuple(slice(start, stop) for start, stop in zip(lower, upper, strict=True))
        values = source[region]
        result[batch_index, 0][region] = torch.where(
            (values == 0) | (values == 1), values, IGNORE_LABEL
        )
    return result


def component_box_target(
    target: torch.Tensor,
    full_shape: tuple[int, int, int],
    full_margin_voxels: tuple[int, int, int],
) -> torch.Tensor:
    """Trust the union of expanded boxes around individual foreground components.

    A single box around all foreground can span unrelated lesions and turn the
    intervening, potentially unannotated tissue into a hard negative. Components
    are therefore identified with 26-connectivity and expanded independently.
    The operation is intentionally performed on the already augmented target so
    its geometry exactly matches the patch presented to the network.
    """

    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError("target must be [B,1,D,H,W]")
    spatial_shape = target.shape[2:]
    scaled_margin = tuple(
        max(1, int(np.ceil(margin * size / full)))
        for margin, size, full in zip(
            full_margin_voxels, spatial_shape, full_shape, strict=True
        )
    )
    result = torch.full_like(target, IGNORE_LABEL)
    connectivity = np.ones((3, 3, 3), dtype=np.uint8)
    for batch_index in range(target.shape[0]):
        source = target[batch_index, 0]
        foreground = source == 1
        coordinates = torch.where(foreground)
        if coordinates[0].numel() == 0:
            continue

        foreground_lower = [int(axis.min().item()) for axis in coordinates]
        foreground_upper = [int(axis.max().item()) + 1 for axis in coordinates]
        foreground_region = tuple(
            slice(start, stop)
            for start, stop in zip(foreground_lower, foreground_upper, strict=True)
        )
        foreground_crop = (
            foreground[foreground_region].detach().to(device="cpu").numpy()
        )
        labeled, component_count = ndimage.label(
            foreground_crop, structure=connectivity
        )
        trusted = torch.zeros_like(source, dtype=torch.bool)
        for component_slice in ndimage.find_objects(labeled, component_count):
            if component_slice is None:
                continue
            lower = [
                max(0, foreground_start + int(axis.start) - margin)
                for foreground_start, axis, margin in zip(
                    foreground_lower, component_slice, scaled_margin, strict=True
                )
            ]
            upper = [
                min(
                    size,
                    foreground_start + int(axis.stop) + margin,
                )
                for foreground_start, axis, margin, size in zip(
                    foreground_lower,
                    component_slice,
                    scaled_margin,
                    spatial_shape,
                    strict=True,
                )
            ]
            region = tuple(
                slice(start, stop) for start, stop in zip(lower, upper, strict=True)
            )
            trusted[region] = True

        known_binary = (source == 0) | (source == 1)
        result[batch_index, 0] = torch.where(
            trusted & known_binary, source, IGNORE_LABEL
        )
    return result


class STUNetTrainerPanCancer(nnUNetTrainer):
    """STU-Net-B proposal model with partial-label-safe positive-box training."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = 3e-4
        self.num_epochs = 300
        self.save_every = 5
        self.oversample_foreground_percent = 1.0
        self.bbox_context_mm = 16.0
        # Preserve every post-fork model state for retrospective public
        # inference. Full optimizer checkpoints remain every ten epochs.
        self.inference_checkpoint_start_epoch = 101
        spacing = tuple(float(value) for value in self.configuration_manager.spacing)
        self.full_margin_voxels = tuple(
            max(1, int(np.ceil(self.bbox_context_mm / value))) for value in spacing
        )
        self._wandb_run = None

    def get_dataloaders(self):
        """Use matched, seeded, ordered augmentation after an optimizer resume.

        nnU-Net 2.6 uses ``NonDetMultiThreadedAugmenter`` with entropy-derived
        worker seeds and completion-order delivery. Two otherwise identical
        one-GPU branches would therefore receive different batches. Replacing
        that constructor process-locally preserves nnU-Net's transforms while
        assigning fixed worker seeds and round-robin result ordering.
        """

        trainer_module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        original_augmenter = trainer_module.NonDetMultiThreadedAugmenter
        augmenter_index = 0
        augmentation_seed_offset = int(
            os.environ.get("PANCANCER_AUGMENTATION_SEED_OFFSET", "0")
        )
        local_rank = int(getattr(self, "local_rank", 0))

        class MatchedOrderedAugmenter(MultiThreadedAugmenter):
            def __init__(
                self,
                data_loader,
                transform,
                num_processes,
                num_cached=2,
                seeds=None,
                pin_memory=False,
                wait_time=0.02,
            ):
                nonlocal augmenter_index
                if seeds is not None:
                    raise ValueError("matched augmenter requires centrally assigned seeds")
                seed_base = matched_augmentation_seed_base(
                    augmentation_seed_offset,
                    local_rank,
                    augmenter_index,
                )
                augmenter_index += 1
                matched_seeds = [seed_base + index for index in range(num_processes)]
                super().__init__(
                    data_loader=data_loader,
                    transform=transform,
                    num_processes=num_processes,
                    num_cached_per_queue=max(1, num_cached // num_processes),
                    seeds=matched_seeds,
                    pin_memory=pin_memory,
                    timeout=120,
                    wait_time=wait_time,
                )

        trainer_module.NonDetMultiThreadedAugmenter = MatchedOrderedAugmenter
        try:
            return super().get_dataloaders()
        finally:
            trainer_module.NonDetMultiThreadedAugmenter = original_augmenter

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        del architecture_class_name, arch_init_kwargs_req_import
        strides = [list(value) for value in arch_init_kwargs["strides"]][1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        return STUNet(
            num_input_channels,
            num_output_channels,
            depth=[1] * 6,
            dims=[32, 64, 128, 256, 512, 512],
            pool_op_kernel_sizes=strides,
            conv_kernel_sizes=[[3, 3, 3]] * 6,
            enable_deep_supervision=enable_deep_supervision,
        )

    def _build_loss(self):
        loss = PartialBoxTverskyCELoss(alpha=0.3, beta=0.7)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.asarray([1.0 / (2**index) for index in range(len(scales))])
            weights[-1] = 1e-6 if self.is_ddp else 0.0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _partial_targets(self, target):
        targets = target if isinstance(target, list) else [target]
        full_shape = tuple(int(value) for value in targets[0].shape[2:])
        converted = [
            partial_box_target(item, full_shape, self.full_margin_voxels)
            for item in targets
        ]
        return converted if isinstance(target, list) else converted[0]

    def train_step(self, batch: dict) -> dict:
        batch = dict(batch)
        batch["target"] = self._partial_targets(batch["target"])
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        original_target = batch["target"]
        if isinstance(original_target, list):
            original_target = [item.to(self.device, non_blocking=True) for item in original_target]
        else:
            original_target = original_target.to(self.device, non_blocking=True)
        target = self._partial_targets(original_target)
        context = (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            output = self.network(data)
            loss = self.loss(output, target)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
        prediction = output.argmax(1)
        truth = target[:, 0]
        valid = truth != IGNORE_LABEL
        truth = truth == 1
        prediction = prediction == 1
        axes = tuple(range(prediction.ndim))
        tp = (prediction & truth & valid).sum(axes).detach().cpu().numpy()[None]
        fp = (prediction & ~truth & valid).sum(axes).detach().cpu().numpy()[None]
        fn = (~prediction & truth & valid).sum(axes).detach().cpu().numpy()[None]
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp,
            "fp_hard": fp,
            "fn_hard": fn,
        }

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0 and self.__class__.__name__ == "STUNetTrainerPanCancer":
            self._initialize_wandb(
                name="stunetb_partialbox_300ep_20260825",
                run_id="stunetbpartial20260825",
                partial_label_policy="ignore_outside_expanded_positive_bbox",
            )

    def _initialize_wandb(
        self,
        *,
        name: str,
        run_id: str,
        partial_label_policy: str,
        loss_name: str = "trusted_box_CE_plus_Tversky_a03_b07",
        extra_config: dict | None = None,
    ) -> None:
        """Start one rank-zero W&B run; failure never interrupts training."""

        try:
            import wandb

            config = {
                "architecture": "STU-Net-B",
                "upstream_commit": UPSTREAM_COMMIT,
                "pretraining": "TotalSegmentator-105-class-4000-epoch",
                "initial_lr": self.initial_lr,
                "num_epochs": self.num_epochs,
                "bbox_context_mm": self.bbox_context_mm,
                "oversample_foreground_percent": self.oversample_foreground_percent,
                "partial_label_policy": partial_label_policy,
                "loss": loss_name,
                "augmentation_delivery": "ordered_round_robin",
                "augmentation_seed_base": 2_026_082_600,
            }
            if extra_config:
                config.update(extra_config)
            self._wandb_run = wandb.init(
                project="pancancer-ct",
                name=name,
                id=run_id,
                resume="allow",
                mode=os.environ.get("WANDB_MODE", "online"),
                config=config,
            )
        except Exception as error:
            self.print_to_log_file(f"W&B initialization failed safely: {error}")
            self._wandb_run = None

    def on_epoch_end(self):
        epoch = self.current_epoch
        metrics = self.logger.my_fantastic_logging
        payload = {
            "epoch": epoch,
            "train_loss": float(metrics["train_losses"][-1]),
            "val_loss": float(metrics["val_losses"][-1]),
            "ema_fg_dice": float(metrics["ema_fg_dice"][-1]),
            "learning_rate": float(metrics["lrs"][-1]),
        }
        if (epoch + 1) % 10 == 0:
            self.save_checkpoint(
                os.path.join(self.output_folder, f"checkpoint_epoch_{epoch + 1:04d}.pth")
            )
        if epoch + 1 >= self.inference_checkpoint_start_epoch:
            self.save_inference_checkpoint(
                os.path.join(
                    self.output_folder,
                    f"checkpoint_inference_epoch_{epoch + 1:04d}.pth",
                )
            )
        super().on_epoch_end()
        if self.local_rank == 0 and self._wandb_run is not None:
            self._wandb_run.log(payload, step=epoch)

    def save_inference_checkpoint(self, filename: str) -> None:
        """Atomically save only the fields required by nnU-Net prediction.

        Optimizer, gradient-scaler, and logger histories approximately double
        the size of a full resume checkpoint but are not read by
        ``nnUNetPredictor.initialize_from_trained_model_folder``. The compact
        snapshots therefore let us retain every epoch without sacrificing the
        ordinary ten-epoch resume checkpoints.
        """

        if self.local_rank != 0:
            return
        if self.disable_checkpointing:
            self.print_to_log_file(
                "No inference checkpoint written, checkpointing is disabled"
            )
            return
        model = self.network.module if self.is_ddp else self.network
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        logging = self.logger.my_fantastic_logging

        def last_float(key: str) -> float | None:
            values = logging.get(key, ())
            return float(values[-1]) if values else None

        internal_metrics = {
            "mean_fg_dice": last_float("mean_fg_dice"),
            "ema_fg_dice": last_float("ema_fg_dice"),
            "train_loss": last_float("train_losses"),
            "val_loss": last_float("val_losses"),
            "learning_rate": last_float("lrs"),
        }
        checkpoint = {
            "network_weights": model.state_dict(),
            "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            "current_epoch": self.current_epoch + 1,
            "checkpoint_kind": "inference_only",
            "internal_metrics": internal_metrics,
        }
        destination = Path(filename)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        metadata = {
            "schema_version": 1,
            "checkpoint": destination.name,
            "checkpoint_bytes": destination.stat().st_size,
            "current_epoch": self.current_epoch + 1,
            "trainer_name": self.__class__.__name__,
            "checkpoint_kind": "inference_only",
            "internal_partial_label_metrics": internal_metrics,
            "selection_warning": (
                "trusted-box internal diagnostic only; fixed-public DSC/NSD is required "
                "for promotion"
            ),
        }
        sidecar = destination.with_suffix(".json")
        sidecar_temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
        try:
            sidecar_temporary.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(sidecar_temporary, sidecar)
        finally:
            if sidecar_temporary.exists():
                sidecar_temporary.unlink()

    def on_train_end(self):
        super().on_train_end()
        if self.local_rank == 0 and self._wandb_run is not None:
            self._wandb_run.finish()

    def perform_actual_validation(self, save_probabilities: bool = False):
        del save_probabilities
        self.print_to_log_file(
            "perform_actual_validation skipped: fold=all would score partial training "
            "labels. Use the immutable public-50 evaluator."
        )


class STUNetTrainerPanCancerComponentBox(STUNetTrainerPanCancer):
    """Ablation that trusts a separate context box for each connected lesion."""

    def _partial_targets(self, target):
        targets = target if isinstance(target, list) else [target]
        full_shape = tuple(int(value) for value in targets[0].shape[2:])
        converted = [
            component_box_target(item, full_shape, self.full_margin_voxels)
            for item in targets
        ]
        return converted if isinstance(target, list) else converted[0]

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            self._initialize_wandb(
                name="stunetb_componentbox_branch_20260826",
                run_id="stunetbcomponentbox20260826",
                partial_label_policy="ignore_outside_union_of_component_bboxes",
            )


class STUNetTrainerPanCancerComponentBoxE160Balanced(
    STUNetTrainerPanCancerComponentBox
):
    """Precision-preserving 80-epoch continuation from component epoch 160.

    The network weights are loaded as pretrained weights, so SGD momentum and
    the polynomial schedule restart instead of inheriting the trajectory that
    regressed between epochs 160 and 200. All data, augmentation, trusted-box,
    architecture, and sampling choices remain unchanged.
    """

    parent_checkpoint_sha256 = (
        "81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74"
    )

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = 3e-5
        self.num_epochs = 80
        # The host is under intermittent system-wide memory pressure. Persist
        # optimizer state after every epoch so an external OOM kill loses at
        # most the in-flight epoch rather than rolling back several epochs.
        self.save_every = 1
        self.inference_checkpoint_start_epoch = 1

    def _build_loss(self):
        loss = PartialBoxTverskyCELoss(alpha=0.5, beta=0.5)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.asarray([1.0 / (2**index) for index in range(len(scales))])
            weights[-1] = 1e-6 if self.is_ddp else 0.0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def on_train_start(self):
        # Bypass ComponentBox.on_train_start so this branch does not attach to
        # the completed epoch-100-to-300 W&B run.
        STUNetTrainerPanCancer.on_train_start(self)
        if self.local_rank == 0:
            self._initialize_wandb(
                name="stunetb_componentbox_e160_balanced_80ep_20260826",
                run_id="stunetbcompe160balanced20260826",
                partial_label_policy="ignore_outside_union_of_component_bboxes",
                loss_name="trusted_component_box_CE_plus_Tversky_a05_b05",
                extra_config={
                    "parent_epoch": 160,
                    "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                    "optimizer_restart": True,
                    "optimizer": "SGD_Nesterov_momentum_0.99",
                    "lr_schedule": "polynomial_80_epochs",
                },
            )


class STUNetTrainerPanCancerSmoke(STUNetTrainerPanCancer):
    """Two-iteration DDP/data/loss/VRAM preflight; never a model candidate."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
