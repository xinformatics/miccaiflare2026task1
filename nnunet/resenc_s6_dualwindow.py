"""Function-preserving dual-window continuation for pan-cancer ResEnc S6.

The preprocessed CT channel is retained unchanged. Two deterministic views are
derived in the first convolution from the normalized tensor: the soft-tissue
window used by GF Focus and a broad lung window. The original one-channel stem
is copied into channel zero and the additional kernels are initialized to zero,
so loading a parent checkpoint preserves its initial predictions up to ordinary
convolution floating-point tolerance.

No additional preprocessed image cache is required.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import Tensor, nn

from nnunet.resenc_s6_partialbox import (
    nnUNetTrainerResEncS6PanCancerPartialBox,
)
from nnunet.resenc_s6_positive_unlabeled import (
    EXPECTED_LABELED_CASES,
    EXPECTED_PSEUDO_CASES,
    PSEUDO_EXPOSURE,
    nnUNetTrainerResEncS6PanCancerPositiveUnlabeled,
)
from fault_tolerant_wandb import (
    safe_wandb_init,
    stable_wandb_run_id,
)


CT_MEAN_HU = -44.66527557373047
CT_STD_HU = 245.26239013671875
SOFT_TISSUE_WINDOW_HU = (-175.0, 250.0)
LUNG_WINDOW_HU = (-1000.0, 400.0)
STEM_WEIGHT_KEY = "encoder.stem.convs.0.conv.weight"
STEM_ALIAS_WEIGHT_KEY = "encoder.stem.convs.0.all_modules.0.weight"
STEM_WEIGHT_KEYS = (STEM_WEIGHT_KEY, STEM_ALIAS_WEIGHT_KEY)
EXPECTED_PARENT_TRAINER = "nnUNetTrainerResEncS6PanCancerPositiveUnlabeled"
EXPECTED_PARENT_EPOCH = 100


def normalized_ct_to_dual_window_channels(normalized_ct: Tensor) -> Tensor:
    """Return normalized CT, soft-tissue and lung channels.

    nnU-Net's CT normalization is inverted with the immutable Dataset501/502
    foreground statistics. Window channels follow the established GF scaling
    convention and lie in [0, 1].
    """

    if normalized_ct.ndim != 5 or normalized_ct.shape[1] != 1:
        raise ValueError("dual-window input must have shape [B,1,D,H,W]")
    hu = normalized_ct * CT_STD_HU + CT_MEAN_HU
    soft_low, soft_high = SOFT_TISSUE_WINDOW_HU
    lung_low, lung_high = LUNG_WINDOW_HU
    soft = hu.clamp(soft_low, soft_high).sub(soft_low).div(soft_high - soft_low)
    lung = hu.clamp(lung_low, lung_high).sub(lung_low).div(lung_high - lung_low)
    return torch.cat((normalized_ct, soft, lung), dim=1)


class DualWindowInputConv3d(nn.Conv3d):
    """Three-channel Conv3d with an external one-channel CT interface."""

    def forward(self, input: Tensor) -> Tensor:
        expanded = normalized_ct_to_dual_window_channels(input)
        return F.conv3d(
            expanded,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def inflate_one_channel_stem(source: nn.Conv3d) -> DualWindowInputConv3d:
    """Inflate a one-channel stem while preserving its initial function."""

    if not isinstance(source, nn.Conv3d) or source.in_channels != 1:
        raise TypeError("ResEnc source stem must be a one-channel Conv3d")
    if source.groups != 1:
        raise ValueError("grouped source stems cannot be inflated safely")
    result = DualWindowInputConv3d(
        in_channels=3,
        out_channels=source.out_channels,
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=1,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        result.weight.zero_()
        result.weight[:, 0].copy_(source.weight[:, 0])
        if source.bias is not None:
            result.bias.copy_(source.bias)
    return result


def replace_resenc_stem_with_dual_window(network: nn.Module) -> nn.Module:
    """Replace the known ResEnc-S6 first convolution in place."""

    try:
        stem = network.encoder.stem.convs[0]
        source = stem.conv
        forward_source = stem.all_modules[0]
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("unexpected ResEnc stem structure") from error
    if forward_source is not source:
        raise ValueError("ResEnc stem convolution aliases unexpectedly diverged")
    inflated = inflate_one_channel_stem(source)
    # ConvDropoutNormReLU exposes the convolution directly and also stores it
    # as the first module used by forward(). Both names must retain one shared
    # object or the old one-channel convolution remains active and trainable.
    stem.conv = inflated
    stem.all_modules[0] = inflated
    return network


def adapt_parent_network_weights(
    parent_weights: Mapping[str, Tensor], target_weights: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """Adapt only the one-to-three-channel stem and validate every other key."""

    parent_keys = set(parent_weights)
    target_keys = set(target_weights)
    if parent_keys != target_keys:
        missing = sorted(target_keys - parent_keys)
        unexpected = sorted(parent_keys - target_keys)
        raise ValueError(
            f"parent/target key mismatch; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    adapted: dict[str, Tensor] = {}
    for key, target in target_weights.items():
        parent = parent_weights[key]
        if key in STEM_WEIGHT_KEYS:
            if (
                parent.ndim != 5
                or target.ndim != 5
                or parent.shape[1] != 1
                or target.shape[1] != 3
                or parent.shape[0] != target.shape[0]
                or parent.shape[2:] != target.shape[2:]
            ):
                raise ValueError(
                    f"invalid stem inflation shapes: {tuple(parent.shape)} -> "
                    f"{tuple(target.shape)}"
                )
            value = torch.zeros_like(target, device="cpu")
            value[:, 0].copy_(parent[:, 0].to(device="cpu", dtype=target.dtype))
            adapted[key] = value
        else:
            if parent.shape != target.shape:
                raise ValueError(
                    f"parent tensor shape changed for {key}: "
                    f"{tuple(parent.shape)} != {tuple(target.shape)}"
                )
            adapted[key] = parent.to(device="cpu", dtype=target.dtype)
    return adapted


def load_function_preserving_parent(
    network: nn.Module,
    checkpoint_path: str | os.PathLike[str],
    expected_trainer: str = EXPECTED_PARENT_TRAINER,
    expected_epoch: int = EXPECTED_PARENT_EPOCH,
) -> str:
    """Strictly load a positive-only ResEnc parent and return its SHA256."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if checkpoint.get("trainer_name") != expected_trainer:
        raise ValueError("dual-window parent trainer mismatch")
    if int(checkpoint.get("current_epoch", -1)) != int(expected_epoch):
        raise ValueError("dual-window parent epoch mismatch")
    parent_weights = checkpoint.get("network_weights")
    if not isinstance(parent_weights, Mapping):
        raise ValueError("dual-window parent has no network weights")
    target_weights = network.state_dict()
    adapted = adapt_parent_network_weights(parent_weights, target_weights)
    network.load_state_dict(adapted, strict=True)
    del adapted, parent_weights, checkpoint
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class nnUNetTrainerResEncS6PanCancerDualWindow(
    nnUNetTrainerResEncS6PanCancerPositiveUnlabeled
):
    """Continue Dataset502 training with function-preserving CT windows."""

    architecture_name = "ResidualEncoderUNet-S6-W32-dynamic-dual-window"
    expected_parameter_count = 107_468_778
    landmark_epochs = (25, 50)
    dual_window_stage_end_epochs = (25, 50, 100)

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = 1e-4
        stage_end_epoch = int(os.environ.get("RESENC_DUALWINDOW_STAGE_END_EPOCH", "100"))
        if stage_end_epoch not in self.dual_window_stage_end_epochs:
            raise ValueError(
                "RESENC_DUALWINDOW_STAGE_END_EPOCH must be one of "
                f"{self.dual_window_stage_end_epochs}, found {stage_end_epoch}"
            )
        self.num_epochs = stage_end_epoch
        self.lr_schedule_horizon_epochs = 100
        self.inference_checkpoint_start_epoch = 101

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if num_input_channels != 1:
            raise ValueError("dynamic dual-window ResEnc requires one preprocessed CT channel")
        network = nnUNetTrainerResEncS6PanCancerPartialBox.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return replace_resenc_stem_with_dual_window(network)

    def on_train_start(self) -> None:
        # Bypass the parent trainer's W&B identity and parameter-count check,
        # while preserving its Dataset502 dataloader and partial-label loss.
        nnUNetTrainer.on_train_start(self)
        if not self.is_ddp or dist.get_world_size() != 2:
            raise RuntimeError("dual-window ResEnc training requires exactly 2 GPUs")
        if self.configuration_manager.batch_size != self.expected_global_batch_size:
            raise RuntimeError("dual-window ResEnc global batch-size contract changed")
        parameter_source = self.network.module if self.is_ddp else self.network
        parameter_source = getattr(parameter_source, "_orig_mod", parameter_source)
        parameter_count = sum(parameter.numel() for parameter in parameter_source.parameters())
        if parameter_count != self.expected_parameter_count:
            raise RuntimeError(
                f"dual-window parameter count changed: {parameter_count} != "
                f"{self.expected_parameter_count}"
            )
        if self.current_epoch == 0:
            parent = os.environ.get("RESENC_DUALWINDOW_PARENT_CHECKPOINT")
            if not parent:
                raise RuntimeError("RESENC_DUALWINDOW_PARENT_CHECKPOINT is required")
            parent_sha256 = load_function_preserving_parent(parameter_source, parent)
            self.print_to_log_file(
                f"Loaded function-preserving dual-window parent SHA256 {parent_sha256}"
            )
        if self.local_rank != 0:
            return
        run_name = "resenc_s6_dualwindow_dataset502_f020_100ep_20260827"
        run_id = stable_wandb_run_id(self.output_folder, run_name)
        self._resenc_wandb_run = safe_wandb_init(
            project=os.environ.get("WANDB_PROJECT", "pancancer-ct"),
            name=run_name,
            id=run_id,
            resume="allow",
            mode=os.environ.get("WANDB_MODE", "online"),
            config={
                "architecture": self.architecture_name,
                "parameters": self.expected_parameter_count,
                "initialization": "function_preserving_positive_resenc_epoch100",
                "input_views": [
                    "nnUNet_normalized_CT",
                    "soft_tissue_-175_250_HU",
                    "lung_-1000_400_HU",
                ],
                "extra_stem_kernel_initialization": "zeros",
                "global_batch_size": self.expected_global_batch_size,
                "initial_lr": self.initial_lr,
                "training_horizon_epochs": 100,
                "optimizer": "fresh_SGD_Nesterov_momentum_0.99",
                "lr_schedule": "polynomial_100_epochs",
                "loss": "component-box_CE_plus_Tversky_a025_b075",
                "labeled_cases": EXPECTED_LABELED_CASES,
                "positive_unlabeled_cases": EXPECTED_PSEUDO_CASES,
                "pseudo_exposure": PSEUDO_EXPOSURE,
                "public_used_for_training": False,
                "hidden_used_for_training": False,
            },
        )
