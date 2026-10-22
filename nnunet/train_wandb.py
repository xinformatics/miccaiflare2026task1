#!/usr/bin/env python3
"""Configurable one/two-GPU nnU-Net training with W&B and epoch inference."""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from functools import partial
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.run_training import (
    cleanup_ddp,
    find_free_network_port,
    maybe_load_checkpoint,
    setup_ddp,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as BaseTrainer
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from nnunet.tversky import MemoryEfficientSoftTverskyLoss


DATASET_NAME = os.environ.get("NNUNET_DATASET", "Dataset501_PanCancer")
PLANS_NAME = os.environ.get("NNUNET_PLANS", "nnUNetPlans")
CONFIGURATION = os.environ.get("NNUNET_CONFIGURATION", "3d_fullres_ResEncS6")
FOLD = os.environ.get("NNUNET_FOLD", "all")
WORLD_SIZE = int(os.environ.get("NNUNET_WORLD_SIZE", "2"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# Keep this class name identical to the stock trainer because the architecture,
# checkpoints, and inference behavior are unchanged. This lets standard
# nnUNetv2_predict load the model without installing custom code site-wide.
class nnUNetTrainer(BaseTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        # Keep this signature explicit: nnU-Net records constructor arguments by
        # introspecting the subclass signature and cannot handle *args/**kwargs.
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        if self._env_flag("NNUNET_DISABLE_DEEP_SUPERVISION", False):
            self.enable_deep_supervision = False
        self.num_epochs = int(os.environ.get("NNUNET_NUM_EPOCHS", "500"))
        self.initial_lr = float(os.environ.get("NNUNET_INITIAL_LR", "1e-3"))
        self.oversample_foreground_percent = float(
            os.environ.get(
                "NNUNET_OVERSAMPLE_FOREGROUND",
                str(self.oversample_foreground_percent),
            )
        )
        if not 0 <= self.oversample_foreground_percent <= 1:
            raise ValueError("NNUNET_OVERSAMPLE_FOREGROUND must be between 0 and 1")
        self.gradient_accumulation_steps = int(
            os.environ.get("NNUNET_GRADIENT_ACCUMULATION_STEPS", "1")
        )
        if self.gradient_accumulation_steps < 1:
            raise ValueError("NNUNET_GRADIENT_ACCUMULATION_STEPS must be >= 1")
        # A resumable checkpoint must correspond to every set of exported
        # epoch predictions. Keep checkpoint_latest for nnU-Net resume behavior;
        # on_epoch_end also archives an immutable epoch-numbered copy.
        self.save_every = 1

    def configure_optimizers(self):
        optimizer_name = os.environ.get("NNUNET_OPTIMIZER", "sgd").lower()
        if optimizer_name == "sgd":
            return super().configure_optimizers()
        if optimizer_name != "adamw":
            raise ValueError(f"Unsupported NNUNET_OPTIMIZER: {optimizer_name}")
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def set_deep_supervision_enabled(self, enabled: bool):
        if self._env_flag("NNUNET_DISABLE_DEEP_SUPERVISION", False):
            return
        return super().set_deep_supervision_enabled(enabled)

    def _tversky_parameters(self) -> tuple[float, float] | None:
        alpha_value = os.environ.get("NNUNET_TVERSKY_ALPHA")
        beta_value = os.environ.get("NNUNET_TVERSKY_BETA")
        if alpha_value is None and beta_value is None:
            return None
        if alpha_value is None or beta_value is None:
            raise ValueError(
                "NNUNET_TVERSKY_ALPHA and NNUNET_TVERSKY_BETA must be set together"
            )
        alpha, beta = float(alpha_value), float(beta_value)
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError("Invalid Tversky alpha/beta values")
        return alpha, beta

    def _build_loss(self):
        parameters = self._tversky_parameters()
        if parameters is None:
            return super()._build_loss()
        alpha, beta = parameters

        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            tversky = partial(
                MemoryEfficientSoftTverskyLoss,
                alpha=alpha,
                beta=beta,
            )
            loss = DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=tversky,
            )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(scales))])
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def train_step(self, batch: dict) -> dict:
        """Optionally accumulate microbatches without changing optimizer updates."""
        if self.gradient_accumulation_steps == 1:
            return super().train_step(batch)
        if self.is_ddp:
            raise RuntimeError("Gradient accumulation is only enabled for single-GPU runs")

        batches = [batch]
        batches.extend(
            next(self.dataloader_train)
            for _ in range(self.gradient_accumulation_steps - 1)
        )
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        for microbatch in batches:
            data = microbatch["data"].to(self.device, non_blocking=True)
            target = microbatch["target"]
            if isinstance(target, list):
                target = [item.to(self.device, non_blocking=True) for item in target]
            else:
                target = target.to(self.device, non_blocking=True)
            amp = (
                torch.autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with amp:
                output = self.network(data)
                loss = self.loss(output, target)
            losses.append(loss.detach())
            scaled_loss = loss / self.gradient_accumulation_steps
            if self.grad_scaler is not None:
                self.grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        mean_loss = torch.stack(losses).mean()
        return {"loss": mean_loss.cpu().numpy()}

    @staticmethod
    def _env_flag(name: str, default: bool = True) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _inference_network(self) -> torch.nn.Module:
        """Return the rank-local network without its DDP wrapper."""
        network = self.network
        if isinstance(network, DistributedDataParallel):
            network = network.module
        # Preserve nnU-Net's existing OptimizedModule. If it were unwrapped,
        # nnUNetPredictor would torch.compile the same 107M model a second time.
        return network

    def _score_public_predictions(self, epoch_dir: Path) -> dict[str, float]:
        combined = epoch_dir / "predictions"
        combined.mkdir(parents=True, exist_ok=True)
        for part_dir in sorted(epoch_dir.glob("part_*")):
            for prediction in part_dir.glob("*.nii.gz"):
                destination = combined / prediction.name
                if not destination.exists():
                    destination.symlink_to(prediction.resolve())

        expected = len(list(Path(os.environ["NNUNET_PUBLIC_LABELS"]).glob("*.nii.gz")))
        produced = len(list(combined.glob("*.nii.gz")))
        if produced != expected:
            raise RuntimeError(
                f"Epoch prediction set is incomplete: produced {produced}, expected {expected}"
            )

        metrics_csv = epoch_dir / "challenge_metrics.csv"
        evaluator = PROJECT_ROOT / "handoff_bundle" / "code" / "evaluate.py"
        command = [
            sys.executable,
            str(evaluator),
            "--pred",
            str(combined),
            "--gt",
            os.environ["NNUNET_PUBLIC_LABELS"],
            "--csv",
            str(metrics_csv),
        ]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        (epoch_dir / "challenge_evaluation.log").write_text(
            result.stdout + result.stderr
        )

        with metrics_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        dsc = np.asarray([float(row["Lesion_DSC"]) for row in rows])
        nsd = np.asarray([float(row["Lesion_NSD"]) for row in rows])
        summary = {
            "case_count": int(len(rows)),
            "mean_dsc": float(dsc.mean()),
            "median_dsc": float(np.median(dsc)),
            "mean_nsd": float(nsd.mean()),
            "median_nsd": float(np.median(nsd)),
            "zero_dsc_cases": int((dsc == 0).sum()),
        }
        (epoch_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        return summary

    def _predict_public_validation(self, epoch: int) -> dict[str, float] | None:
        if not self._env_flag("NNUNET_EPOCH_PREDICTIONS", True):
            return None
        every = int(os.environ.get("NNUNET_EPOCH_PREDICTION_EVERY", "1"))
        if every < 1 or (epoch + 1) % every:
            return None

        source = Path(os.environ["NNUNET_PUBLIC_IMAGES"])
        output_root = Path(os.environ["NNUNET_EPOCH_PREDICTION_DIR"])
        epoch_dir = output_root / f"epoch_{epoch:04d}"
        part_dir = epoch_dir / f"part_{self.local_rank}"
        part_dir.mkdir(parents=True, exist_ok=True)
        workers = int(os.environ.get("NNUNET_EPOCH_PREDICTION_WORKERS", "2"))

        if self.is_ddp:
            dist.barrier()
        started = time.time()
        self.print_to_log_file(
            f"Epoch {epoch}: predicting public validation part "
            f"{self.local_rank + 1}/{dist.get_world_size() if self.is_ddp else 1}",
            also_print_to_console=True,
        )
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=True,
                device=self.device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
            inference_network = self._inference_network()
            parameter_source = getattr(inference_network, "_orig_mod", inference_network)
            predictor.manual_initialization(
                inference_network,
                self.plans_manager,
                self.configuration_manager,
                [parameter_source.state_dict()],
                self.dataset_json,
                self.__class__.__name__,
                self.inference_allowed_mirroring_axes,
            )
            world_size = dist.get_world_size() if self.is_ddp else 1
            predictor.predict_from_files(
                str(source),
                str(part_dir),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=workers,
                num_processes_segmentation_export=workers,
                num_parts=world_size,
                part_id=self.local_rank,
            )
            del predictor
            if self.is_ddp:
                dist.barrier()

            summary = None
            if self.local_rank == 0:
                summary = self._score_public_predictions(epoch_dir)
                summary["duration_seconds"] = float(time.time() - started)
                (epoch_dir / "metrics_summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                self.print_to_log_file(
                    f"Epoch {epoch} public validation: DSC={summary['mean_dsc']:.4f}, "
                    f"NSD={summary['mean_nsd']:.4f}, n={summary['case_count']}",
                    also_print_to_console=True,
                )
            if self.is_ddp:
                dist.barrier()
            return summary
        finally:
            self.set_deep_supervision_enabled(True)
            self.network.train()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _log_public_metrics(self, epoch: int, metrics: dict[str, float] | None) -> None:
        if self.local_rank != 0 or self.wandb_run is None or not metrics:
            return
        self.wandb_run.log(
            {
                "epoch": epoch,
                "public_val/mean_dsc": metrics["mean_dsc"],
                "public_val/median_dsc": metrics["median_dsc"],
                "public_val/mean_nsd": metrics["mean_nsd"],
                "public_val/median_nsd": metrics["median_nsd"],
                "public_val/zero_dsc_cases": metrics["zero_dsc_cases"],
                "public_val/prediction_seconds": metrics["duration_seconds"],
            }
        )

    def _init_wandb(self) -> None:
        if self.local_rank != 0:
            self.wandb_run = None
            return

        import wandb

        run_id_file = Path(self.output_folder) / "wandb_run_id.txt"
        run_id = os.environ.get("WANDB_RUN_ID")
        if not run_id and run_id_file.is_file():
            run_id = run_id_file.read_text().strip() or None

        config = {
            "dataset": self.plans_manager.dataset_name,
            "dataset_id": 501,
            "configuration": self.configuration_name,
            "plans": self.plans_manager.plans_name,
            "fold": self.fold,
            "trainer": "nnUNetTrainer+WandB+EpochPredictions",
            "world_size": dist.get_world_size() if self.is_ddp else 1,
            "microbatch_size_per_rank": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_global_batch_size": (
                self.batch_size
                * self.gradient_accumulation_steps
                * (dist.get_world_size() if self.is_ddp else 1)
            ),
            "rank0_batch_size": self.batch_size,
            "patch_size": list(self.configuration_manager.patch_size),
            "spacing": list(self.configuration_manager.spacing),
            "epochs": self.num_epochs,
            "iterations_per_epoch": self.num_iterations_per_epoch,
            "val_iterations_per_epoch": self.num_val_iterations_per_epoch,
            "initial_lr": self.initial_lr,
            "weight_decay": self.weight_decay,
            "optimizer": os.environ.get("NNUNET_OPTIMIZER", "sgd").lower(),
            "foreground_oversampling": self.oversample_foreground_percent,
            "loss": (
                "Tversky+CE" if self._tversky_parameters() is not None else "Dice+CE"
            ),
            "tversky_alpha": (
                self._tversky_parameters()[0]
                if self._tversky_parameters() is not None
                else None
            ),
            "tversky_beta": (
                self._tversky_parameters()[1]
                if self._tversky_parameters() is not None
                else None
            ),
            "training_labels_are_partial": True,
            "public_validation_used_for_training": False,
            "pretrained_weights": os.environ.get("NNUNET_PRETRAINED_SOURCE") or None,
            "full_pretrained_state_loaded": self._env_flag(
                "NNUNET_LOAD_FULL_PRETRAINED", False
            ),
        }
        self.wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "miccai26-ct-segmentation"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=os.environ.get("WANDB_NAME", f"resenc-s6-fold-{self.fold}"),
            id=run_id,
            resume="allow" if run_id else None,
            group=os.environ.get("WANDB_GROUP", self.configuration_name),
            job_type="train",
            dir=os.environ.get("WANDB_DIR"),
            config={key: _json_scalar(value) for key, value in config.items()},
            tags=[
                "nnunetv2",
                "ct",
                "lesion-segmentation",
                "partial-labels",
                f"{WORLD_SIZE}gpu",
                self.configuration_name,
            ],
        )
        run_id_file.write_text(self.wandb_run.id + "\n")
        self.wandb_run.define_metric("epoch")
        self.wandb_run.define_metric("train/*", step_metric="epoch")
        self.wandb_run.define_metric("internal_val/*", step_metric="epoch")
        self.wandb_run.define_metric("public_val/*", step_metric="epoch")
        self.wandb_run.define_metric("system_epoch/*", step_metric="epoch")
        self.print_to_log_file(
            f"W&B run {self.wandb_run.id} ({self.wandb_run.settings.mode}): "
            f"{self.wandb_run.url or self.wandb_run.dir}",
            also_print_to_console=True,
        )

    def on_train_start(self) -> None:
        super().on_train_start()
        parameters = self._tversky_parameters()
        if parameters is not None:
            self.print_to_log_file(
                f"Tversky+CE enabled: alpha={parameters[0]}, beta={parameters[1]}, "
                f"lr={self.initial_lr}, oversample_fg={self.oversample_foreground_percent}",
                also_print_to_console=True,
            )
        self._init_wandb()
        if self.current_epoch == 0 and self._env_flag("NNUNET_EVAL_BEFORE_TRAIN", False):
            baseline_summary = (
                Path(os.environ["NNUNET_EPOCH_PREDICTION_DIR"])
                / "epoch_-001"
                / "metrics_summary.json"
            )
            if not baseline_summary.exists():
                self._log_public_metrics(-1, self._predict_public_validation(-1))
        # A run may be resumed from a checkpoint created before epoch exports
        # were enabled. Backfill the most recently completed epoch first.
        completed_epoch = self.current_epoch - 1
        if completed_epoch >= 0:
            summary = (
                Path(os.environ["NNUNET_EPOCH_PREDICTION_DIR"])
                / f"epoch_{completed_epoch:04d}"
                / "metrics_summary.json"
            )
            if not summary.exists():
                self._log_public_metrics(
                    completed_epoch,
                    self._predict_public_validation(completed_epoch),
                )

    def on_epoch_end(self) -> None:
        epoch = self.current_epoch
        super().on_epoch_end()
        if self.local_rank == 0 and self._env_flag(
            "NNUNET_ARCHIVE_EPOCH_CHECKPOINTS", True
        ):
            latest = Path(self.output_folder) / "checkpoint_latest.pth"
            archived = Path(self.output_folder) / f"checkpoint_epoch_{epoch:04d}.pth"
            temporary = archived.with_name(f".{archived.name}.tmp")
            if not latest.is_file():
                raise FileNotFoundError(
                    f"Cannot archive epoch {epoch}: missing {latest}"
                )
            shutil.copy2(latest, temporary)
            os.replace(temporary, archived)
            self.print_to_log_file(
                f"Archived full epoch checkpoint: {archived.name}",
                also_print_to_console=True,
            )
        log = self.logger.my_fantastic_logging
        if self.local_rank == 0 and self.wandb_run is not None:
            metrics = {
                "epoch": epoch,
                "train/loss": float(log["train_losses"][-1]),
                "internal_val/loss": float(log["val_losses"][-1]),
                "internal_val/mean_foreground_dice": float(log["mean_fg_dice"][-1]),
                "internal_val/ema_foreground_dice": float(log["ema_fg_dice"][-1]),
                "train/learning_rate": float(log["lrs"][-1]),
                "system_epoch/duration_seconds": float(
                    log["epoch_end_timestamps"][-1] - log["epoch_start_timestamps"][-1]
                ),
            }
            for index, value in enumerate(log["dice_per_class_or_region"][-1]):
                metrics[f"internal_val/dice_class_{index + 1}"] = float(value)
            if self.device.type == "cuda":
                metrics["system_epoch/rank0_peak_memory_gib"] = (
                    torch.cuda.max_memory_allocated(self.device) / 2**30
                )
                torch.cuda.reset_peak_memory_stats(self.device)
            self.wandb_run.log(metrics)

        public_metrics = self._predict_public_validation(epoch)
        self._log_public_metrics(epoch, public_metrics)

    def on_train_end(self) -> None:
        super().on_train_end()
        if self.local_rank == 0 and self.wandb_run is not None:
            self.wandb_run.summary["internal_val/best_ema_foreground_dice"] = float(
                self._best_ema
            )
            self.wandb_run.finish()

    def perform_actual_validation(self, save_probabilities: bool = False) -> None:
        # With fold=all nnU-Net sets validation keys equal to all 17,575 training
        # cases. Its automatic post-training validation would therefore add many
        # hours of training-set inference. Challenge validation is run separately.
        self.print_to_log_file(
            "Skipping automatic fold-all validation; use the held-out challenge "
            "validation pipeline instead.",
            also_print_to_console=True,
        )


def _worker(
    rank: int,
    continue_training: bool,
    validation_only: bool,
    val_best: bool,
    save_probabilities: bool,
    epochs: int | None,
    pretrained_weights: str | None,
    monai_swin_pretrained: str | None,
) -> None:
    trainer = None
    if WORLD_SIZE > 1:
        setup_ddp(rank, WORLD_SIZE)
        torch.cuda.set_device(torch.device("cuda", dist.get_rank()))
    else:
        torch.cuda.set_device(torch.device("cuda", 0))
    try:
        base = Path(nnUNet_preprocessed) / DATASET_NAME
        plans = load_json(str(base / f"{PLANS_NAME}.json"))
        batch_size = os.environ.get("NNUNET_BATCH_SIZE")
        if batch_size is not None:
            plans["configurations"][CONFIGURATION]["batch_size"] = int(batch_size)
        trainer = nnUNetTrainer(
            plans=plans,
            configuration=CONFIGURATION,
            fold=FOLD,
            dataset_json=load_json(str(base / "dataset.json")),
        )
        if epochs is not None:
            trainer.num_epochs = epochs
        load_full_pretrained = nnUNetTrainer._env_flag(
            "NNUNET_LOAD_FULL_PRETRAINED", False
        )
        if monai_swin_pretrained:
            trainer.initialize()
            network = trainer.network
            if isinstance(network, DistributedDataParallel):
                network = network.module
            network = getattr(network, "_orig_mod", network)
            loader = getattr(network, "load_pretrained_encoder", None)
            if not callable(loader):
                raise TypeError("Configured network cannot load MONAI Swin weights")
            tensor_count = loader(monai_swin_pretrained)
            print(
                f"Loaded {tensor_count} official MONAI pretrained Swin encoder "
                f"tensors from {monai_swin_pretrained}"
            )
        elif pretrained_weights and load_full_pretrained:
            trainer.initialize()
            checkpoint = torch.load(
                pretrained_weights,
                map_location=trainer.device,
                weights_only=False,
            )
            network = trainer.network
            if isinstance(network, DistributedDataParallel):
                network = network.module
            network = getattr(network, "_orig_mod", network)
            network.load_state_dict(checkpoint["network_weights"], strict=True)
            print(
                "Loaded the complete pretrained network state, including the "
                f"binary segmentation head, from {pretrained_weights}"
            )
        else:
            maybe_load_checkpoint(
                trainer,
                continue_training,
                validation_only,
                pretrained_weights_file=pretrained_weights,
            )
        cudnn.deterministic = False
        cudnn.benchmark = True
        if not validation_only:
            trainer.run_training()
        if val_best:
            trainer.load_checkpoint(join(trainer.output_folder, "checkpoint_best.pth"))
        trainer.perform_actual_validation(save_probabilities)
    except BaseException:
        # nnU-Net's augmenter processes can outlive the trainer after a CUDA
        # exception and retain a large CUDA context. Shut them down explicitly
        # before re-raising so a retry starts on a genuinely clean GPU.
        if trainer is not None:
            for loader_name in ("dataloader_train", "dataloader_val"):
                loader = getattr(trainer, loader_name, None)
                finish = getattr(loader, "_finish", None)
                if callable(finish):
                    try:
                        finish()
                    except Exception:
                        pass
            wandb_run = getattr(trainer, "wandb_run", None)
            if wandb_run is not None:
                try:
                    wandb_run.finish(exit_code=1)
                except Exception:
                    pass
        raise
    finally:
        if WORLD_SIZE > 1:
            cleanup_ddp()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", action="store_true", help="Continue from the latest checkpoint")
    parser.add_argument("--val", action="store_true", help="Only run final validation")
    parser.add_argument("--val_best", action="store_true", help="Validate checkpoint_best")
    parser.add_argument("--npz", action="store_true", help="Save validation probabilities")
    parser.add_argument(
        "--pretrained-weights",
        default=None,
        help="Initialize a fresh run from an nnU-Net checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override nnU-Net's number of training epochs (for example, --epochs 1).",
    )
    parser.add_argument(
        "--monai-swin-pretrained",
        default=None,
        help="Initialize the SwinUNETR encoder from an official MONAI SSL checkpoint.",
    )
    args = parser.parse_args()
    if args.c and args.val:
        parser.error("--c and --val cannot be used together")
    if args.c and args.pretrained_weights:
        parser.error("--c and --pretrained-weights cannot be used together")
    if args.c and args.monai_swin_pretrained:
        parser.error("--c and --monai-swin-pretrained cannot be used together")
    if args.pretrained_weights and args.monai_swin_pretrained:
        parser.error("Choose only one pretrained initialization source")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")

    worker_args = (
        args.c,
        args.val,
        args.val_best,
        args.npz,
        args.epochs,
        args.pretrained_weights,
        args.monai_swin_pretrained,
    )
    if WORLD_SIZE > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ.setdefault("MASTER_PORT", str(find_free_network_port()))
        mp.spawn(_worker, args=worker_args, nprocs=WORLD_SIZE, join=True)
    else:
        _worker(0, *worker_args)


if __name__ == "__main__":
    main()
