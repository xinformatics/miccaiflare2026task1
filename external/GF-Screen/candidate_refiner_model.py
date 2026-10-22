"""Prompt-conditioned candidate classification and voxel refinement model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def downsample_prompt_weights(
    prompt: torch.Tensor, output_shape: tuple[int, int, int]
) -> torch.Tensor:
    """Map prompt occupancy to the encoder grid without losing tiny seeds."""

    if prompt.ndim != 5 or prompt.shape[1] != 1 or len(output_shape) != 3:
        raise ValueError("prompt pooling requires Bx1xDxHxW input and a 3-D shape")
    if any(int(value) < 1 for value in output_shape):
        raise ValueError("prompt pooling output shape must be positive")
    # Coordinate sampling can entirely miss a tiny prompt. Adaptive averaging
    # integrates every input voxel and retains fractional occupancy.
    return F.adaptive_avg_pool3d(prompt.float().clamp_min(0.0), output_shape)


def groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups(out_channels), out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = self.activation(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return self.activation(value + residual)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(
            value, size=skip.shape[2:], mode="trilinear", align_corners=False
        )
        return self.block(torch.cat((value, skip), dim=1))


class CandidateRefiner(nn.Module):
    """Small residual 3-D U-Net with a shared candidate-presence head."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
        prompt_masked_classifier: bool = False,
        prompt_channel: int = 2,
    ):
        super().__init__()
        if in_channels < 1 or base_channels < 4:
            raise ValueError("invalid channel count")
        if prompt_masked_classifier and not 0 <= prompt_channel < in_channels:
            raise ValueError("prompt-masked classifier requires a valid prompt channel")
        self.prompt_masked_classifier = bool(prompt_masked_classifier)
        self.prompt_channel = int(prompt_channel)
        self.encoder0 = ResidualBlock(in_channels, base_channels)
        self.encoder1 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.encoder2 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.encoder3 = ResidualBlock(base_channels * 4, base_channels * 8, stride=2)
        self.context = ResidualBlock(base_channels * 8, base_channels * 8)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(
                base_channels * (32 if self.prompt_masked_classifier else 16), 1
            ),
        )
        self.decoder2 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder1 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder0 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        self.segmenter = nn.Conv3d(base_channels, 1, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = (
            value[:, self.prompt_channel : self.prompt_channel + 1]
            if self.prompt_masked_classifier
            else None
        )
        skip0 = self.encoder0(value)
        skip1 = self.encoder1(skip0)
        skip2 = self.encoder2(skip1)
        encoded = self.context(self.encoder3(skip2))
        global_average = F.adaptive_avg_pool3d(encoded, 1).flatten(1)
        global_maximum = F.adaptive_max_pool3d(encoded, 1).flatten(1)
        pooled_parts = [global_average, global_maximum]
        if prompt is not None:
            prompt_weights = downsample_prompt_weights(prompt, encoded.shape[2:])
            weight_sum = prompt_weights.sum(dim=(2, 3, 4), keepdim=True)
            valid_prompt = weight_sum.flatten(1).squeeze(1) > 1e-6
            prompt_average = (
                (encoded * prompt_weights).sum(dim=(2, 3, 4))
                / weight_sum.flatten(1).clamp_min(1e-6)
            )
            prompt_mask = prompt_weights > 1e-6
            prompt_maximum = encoded.masked_fill(
                ~prompt_mask.expand_as(encoded), -torch.inf
            ).amax(dim=(2, 3, 4))
            # Training data are independently audited to contain nonempty
            # prompts. The fallback keeps the opt-in head finite on malformed
            # external inputs without changing the normal path.
            prompt_average = torch.where(
                valid_prompt[:, None], prompt_average, global_average
            )
            prompt_maximum = torch.where(
                valid_prompt[:, None], prompt_maximum, global_maximum
            )
            pooled_parts.extend((prompt_average, prompt_maximum))
        pooled = torch.cat(pooled_parts, dim=1)
        classification = self.classifier(pooled).flatten()
        decoded = self.decoder2(encoded, skip2)
        decoded = self.decoder1(decoded, skip1)
        decoded = self.decoder0(decoded, skip0)
        return classification, self.segmenter(decoded)


def candidate_refiner_loss(
    classification_logits: torch.Tensor,
    segmentation_logits: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
    classification_weight: float = 1.0,
    classification_negative_weight: float = 1.0,
    segmentation_weight: float = 1.0,
    negative_segmentation_weight: float = 0.1,
    boundary_voxel_weight: float = 2.0,
    small_lesion_loss: bool = False,
    small_lesion_positive_weight_cap: float = 100.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = labels.float().reshape(-1)
    targets = targets.float()
    if not 0 < float(classification_negative_weight) <= 1:
        raise ValueError("classification_negative_weight must be in (0, 1]")
    if not 0 <= float(negative_segmentation_weight) <= 1:
        raise ValueError("negative_segmentation_weight must be in [0, 1]")
    if float(classification_negative_weight) == 1.0:
        # Preserve the historical reduction kernel exactly for primary and
        # already-running checkpoints; do not route the default through a
        # mathematically equivalent but numerically different custom sum.
        classification = F.binary_cross_entropy_with_logits(
            classification_logits.reshape(-1), labels
        )
    else:
        classification_map = F.binary_cross_entropy_with_logits(
            classification_logits.reshape(-1), labels, reduction="none"
        )
        classification_weights = torch.where(
            labels > 0.5,
            torch.ones_like(labels),
            torch.full_like(labels, float(classification_negative_weight)),
        )
        # Normalize by effective weight so this intervention changes the
        # relative positive/unlabeled penalty without silently shrinking the
        # entire classification objective against the segmentation objective.
        classification = (
            classification_map * classification_weights
        ).sum() / classification_weights.sum().clamp_min(1e-12)
    positive = labels > 0.5
    if bool(positive.any()):
        logits_positive = segmentation_logits[positive]
        targets_positive = targets[positive]
        probabilities = torch.sigmoid(logits_positive)
        intersection = (probabilities * targets_positive).sum(dim=(1, 2, 3, 4))
        denominator = probabilities.sum(dim=(1, 2, 3, 4)) + targets_positive.sum(
            dim=(1, 2, 3, 4)
        )
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        if small_lesion_loss:
            # A batch-global foreground ratio is dominated by the larger
            # lesions and gives tiny positives almost no BCE signal. Compute
            # one ratio per positive patch for the controlled small-lesion
            # ablation. The explicit cap and gradient clipping in training
            # keep this branch bounded; the default branch is bit-for-bit the
            # historical recipe.
            spatial_axes = tuple(range(1, targets_positive.ndim))
            foreground = targets_positive.sum(dim=spatial_axes)
            voxels_per_sample = targets_positive[0].numel()
            background = float(voxels_per_sample) - foreground
            positive_weight = torch.clamp(
                background / torch.clamp(foreground, min=1.0),
                1.0,
                float(small_lesion_positive_weight_cap),
            ).reshape((-1,) + (1,) * (targets_positive.ndim - 1))
        else:
            foreground = targets_positive.sum()
            background = targets_positive.numel() - foreground
            positive_weight = torch.clamp(
                background / torch.clamp(foreground, min=1.0), 1.0, 30.0
            )
        voxel_bce_map = F.binary_cross_entropy_with_logits(
            logits_positive,
            targets_positive,
            pos_weight=positive_weight,
            reduction="none",
        )
        dilated = F.max_pool3d(targets_positive, 3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool3d(
            1.0 - targets_positive, 3, stride=1, padding=1
        )
        boundary = (dilated - eroded).clamp(0.0, 1.0)
        voxel_weights = 1.0 + float(boundary_voxel_weight) * boundary
        voxel_bce = (voxel_bce_map * voxel_weights).sum() / voxel_weights.sum()
        positive_segmentation = dice + voxel_bce
    else:
        dice = segmentation_logits.sum() * 0.0
        voxel_bce = segmentation_logits.sum() * 0.0
        positive_segmentation = segmentation_logits.sum() * 0.0
    negative = ~positive
    if bool(negative.any()):
        negative_segmentation = F.softplus(segmentation_logits[negative]).mean()
    else:
        negative_segmentation = segmentation_logits.sum() * 0.0
    segmentation = positive_segmentation + negative_segmentation_weight * negative_segmentation
    total = classification_weight * classification + segmentation_weight * segmentation
    return total, {
        "classification": classification.detach(),
        "dice": dice.detach(),
        "voxel_bce": voxel_bce.detach(),
        "negative_segmentation": negative_segmentation.detach(),
    }
