"""Pretrained GF Focus contour model for candidate-centred ROI refinement."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from torch.nn import functional as F


FOCUS_PREFIX = "segmentation_model."
FOCUS_ENCODER_PREFIXES = (
    "swinViT.",
    "encoder1.",
    "encoder2.",
    "encoder3.",
    "encoder4.",
    "encoder10.",
)
FOCUS_INPUT_WEIGHT_KEYS = (
    "swinViT.patch_embed.proj.weight",
    "encoder1.layer.conv1.conv.weight",
    "encoder1.layer.conv3.conv.weight",
)


# This matches the segmentation objective used by the released GF trainer
# when --squared_dice is not set. It has no trainable state and follows the
# input device, so one shared instance is safe.
RELEASE_DICE_CE = DiceCELoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def focus_state_dict(payload: dict) -> dict[str, torch.Tensor]:
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state dictionary")
    selected = {
        str(key)[len(FOCUS_PREFIX) :]: value
        for key, value in state.items()
        if str(key).startswith(FOCUS_PREFIX)
    }
    if not selected:
        # A contour-only checkpoint stores the unprefixed model directly.
        selected = {
            str(key): value
            for key, value in state.items()
            if isinstance(value, torch.Tensor)
        }
    if not selected or not all(isinstance(value, torch.Tensor) for value in selected.values()):
        raise ValueError("checkpoint contains no valid Focus tensors")
    return selected


def build_focus_model(
    use_checkpoint: bool = True, in_channels: int = 1
) -> SwinUNETR:
    if in_channels < 1:
        raise ValueError("Focus input channels must be positive")
    kwargs = dict(
        in_channels=in_channels,
        out_channels=2,
        feature_size=48,
        use_checkpoint=use_checkpoint,
        use_v2=True,
    )
    try:
        return SwinUNETR(**kwargs)
    except TypeError as error:
        if "img_size" not in str(error):
            raise
        return SwinUNETR(img_size=(64, 64, 64), **kwargs)


def expand_focus_input_channels(
    state: dict[str, torch.Tensor], in_channels: int
) -> dict[str, torch.Tensor]:
    """Zero-initialize extra prompt channels while preserving released logits."""

    if in_channels < 1:
        raise ValueError("Focus input channels must be positive")
    expanded = dict(state)
    for key in FOCUS_INPUT_WEIGHT_KEYS:
        if key not in state:
            raise ValueError(f"Focus state has no input projection: {key}")
        weight = state[key]
        if weight.ndim != 5:
            raise ValueError(
                f"unexpected Focus input projection shape: {key} {tuple(weight.shape)}"
            )
        if weight.shape[1] == in_channels:
            continue
        if weight.shape[1] != 1:
            raise ValueError(
                f"cannot adapt {key} from {weight.shape[1]} channels to {in_channels}"
            )
        value = weight.new_zeros((weight.shape[0], in_channels, *weight.shape[2:]))
        value[:, 0:1].copy_(weight)
        expanded[key] = value
    return expanded


def load_pretrained_focus(
    path: Path, use_checkpoint: bool = True, in_channels: int = 1
) -> SwinUNETR:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_focus_model(
        use_checkpoint=use_checkpoint, in_channels=in_channels
    )
    state = expand_focus_input_channels(focus_state_dict(payload), in_channels)
    model.load_state_dict(state, strict=True)
    return model


def set_focus_encoder_trainable(model: SwinUNETR, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith(FOCUS_ENCODER_PREFIXES):
            parameter.requires_grad = bool(trainable)


def set_focus_prompt_adapter_trainable(
    model: SwinUNETR, trainable: bool
) -> None:
    """Control the input projection independently of the frozen encoder."""

    found = False
    for name, parameter in model.named_parameters():
        if any(name == key for key in (
            "swinViT.patch_embed.proj.weight",
            "swinViT.patch_embed.proj.bias",
            "encoder1.layer.conv1.conv.weight",
            "encoder1.layer.conv3.conv.weight",
        )):
            parameter.requires_grad = bool(trainable)
            found = True
    if not found:
        raise ValueError("Focus model has no prompt input projection")


def focus_discriminative_parameter_groups(
    model: SwinUNETR,
    encoder_lr_scale: float,
) -> list[dict]:
    """Split trainable Focus tensors into decoder and encoder LR groups.

    Decoder/output tensors retain the requested peak learning rate.  The
    pretrained Swin and convolutional encoder tensors receive a smaller,
    explicitly recorded multiplier.  Returning the decoder group first keeps
    the historical ``lr`` metric equal to the user-facing peak rate.
    """

    scale = float(encoder_lr_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("encoder_lr_scale must be in (0, 1]")
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Focus model has no trainable parameters")
    if scale == 1.0:
        return [{"params": [parameter for _, parameter in trainable], "lr_scale": 1.0}]
    encoder = [
        parameter
        for name, parameter in trainable
        if name.startswith(FOCUS_ENCODER_PREFIXES)
    ]
    decoder = [
        parameter
        for name, parameter in trainable
        if not name.startswith(FOCUS_ENCODER_PREFIXES)
    ]
    if not encoder or not decoder:
        raise ValueError(
            "discriminative Focus tuning requires trainable encoder and decoder tensors"
        )
    return [
        {"params": decoder, "lr_scale": 1.0},
        {"params": encoder, "lr_scale": scale},
    ]


def foreground_logit(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5 or logits.shape[1] != 2:
        raise ValueError("Focus logits must have shape (B,2,D,H,W)")
    return logits[:, 1:2] - logits[:, 0:1]


def focus_contour_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    negative_weight: float = 0.1,
    boundary_voxel_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dice/BCE contour loss with bounded negative-candidate suppression."""

    binary_logits = foreground_logit(logits)
    targets = targets.float()
    labels = labels.float().reshape(-1)
    if targets.shape != binary_logits.shape or len(labels) != len(targets):
        raise ValueError("Focus target/label shape mismatch")
    positive = labels > 0.5
    if bool(positive.any()):
        positive_logits = binary_logits[positive]
        positive_targets = targets[positive]
        probabilities = torch.sigmoid(positive_logits)
        intersection = (probabilities * positive_targets).sum(dim=(1, 2, 3, 4))
        denominator = probabilities.sum(dim=(1, 2, 3, 4)) + positive_targets.sum(
            dim=(1, 2, 3, 4)
        )
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        foreground = positive_targets.sum()
        background = positive_targets.numel() - foreground
        positive_weight = torch.clamp(
            background / torch.clamp(foreground, min=1.0), 1.0, 30.0
        )
        bce_map = F.binary_cross_entropy_with_logits(
            positive_logits,
            positive_targets,
            pos_weight=positive_weight,
            reduction="none",
        )
        dilated = F.max_pool3d(positive_targets, 3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool3d(
            1.0 - positive_targets, 3, stride=1, padding=1
        )
        boundary = (dilated - eroded).clamp(0.0, 1.0)
        voxel_weights = 1.0 + float(boundary_voxel_weight) * boundary
        bce = (bce_map * voxel_weights).sum() / voxel_weights.sum()
    else:
        dice = binary_logits.sum() * 0.0
        bce = binary_logits.sum() * 0.0
    negative = ~positive
    negative_loss = (
        F.softplus(binary_logits[negative]).mean()
        if bool(negative.any())
        else binary_logits.sum() * 0.0
    )
    total = dice + bce + float(negative_weight) * negative_loss
    return total, {
        "dice": dice.detach(),
        "voxel_bce": bce.detach(),
        "negative": negative_loss.detach(),
    }


def focus_release_dice_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Released GF two-class Dice+CE objective on the audited ROI batch."""

    if logits.ndim != 5 or logits.shape[1] != 2:
        raise ValueError("Focus logits must have shape (B,2,D,H,W)")
    if targets.shape != logits[:, :1].shape or len(labels.reshape(-1)) != len(targets):
        raise ValueError("Focus target/label shape mismatch")
    # DiceCELoss requires integer class indices in (B,1,D,H,W). Keeping both
    # positive and trusted-negative patches exactly mirrors the released
    # complete-label training semantics.
    total = RELEASE_DICE_CE(logits, targets.long())
    zero = total.detach() * 0.0
    return total, {"release_dice_ce": total.detach(), "negative": zero}


def focus_training_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    recipe: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if recipe == "boundary_balanced":
        return focus_contour_loss(logits, targets, labels)
    if recipe == "release_dice_ce":
        return focus_release_dice_ce_loss(logits, targets, labels)
    raise ValueError(f"unknown Focus loss recipe: {recipe}")
