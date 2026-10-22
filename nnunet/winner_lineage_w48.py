"""Efficient winner-lineage W48 network for pan-cancer CT segmentation.

The network preserves the four-resolution-stage PlainConvUNet used by the
provided FLARE25 checkpoint. It adds capacity and context where they are
cheapest (the bottleneck) and learns a patch-level lesion-presence gate to
suppress foreground predictions on lesion-free inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dynamic_network_architectures.architectures.unet import PlainConvUNet


class DepthwiseResidualBottleneck3D(nn.Module):
    """Inverted-cost residual context block using a depthwise 3D convolution."""

    def __init__(self, channels: int, bottleneck_ratio: float = 0.5) -> None:
        super().__init__()
        hidden = max(16, int(round(channels * bottleneck_ratio / 8)) * 8)
        groups = min(16, hidden)
        while hidden % groups:
            groups -= 1
        self.reduce = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        output_groups = min(16, channels)
        while channels % output_groups:
            output_groups -= 1
        self.expand = nn.Sequential(
            nn.Conv3d(hidden, channels, 1, bias=False),
            nn.GroupNorm(output_groups, channels),
        )
        # A small initial residual scale keeps optimization close to the proven
        # winner topology while allowing the new branch to grow useful context.
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 1e-3))
        self.activation = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.expand(self.depthwise(self.reduce(x)))
        return self.activation(x + self.layer_scale * residual)


class PyramidContext3D(nn.Module):
    """Low-cost global/multi-scale context at the smallest feature map."""

    def __init__(
        self,
        channels: int,
        pool_sizes: Sequence[int] = (1, 2, 4),
        branch_channels: int = 40,
    ) -> None:
        super().__init__()
        self.pool_sizes = tuple(int(size) for size in pool_sizes)
        branch_groups = min(8, branch_channels)
        while branch_channels % branch_groups:
            branch_groups -= 1
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool3d(size),
                    nn.Conv3d(channels, branch_channels, 1, bias=False),
                    nn.GroupNorm(branch_groups, branch_channels),
                    nn.LeakyReLU(negative_slope=1e-2, inplace=True),
                )
                for size in self.pool_sizes
            ]
        )
        fused_channels = channels + len(self.pool_sizes) * branch_channels
        output_groups = min(16, channels)
        while channels % output_groups:
            output_groups -= 1
        self.fuse = nn.Sequential(
            nn.Conv3d(fused_channels, channels, 1, bias=False),
            nn.GroupNorm(output_groups, channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        pooled = [
            F.interpolate(branch(x), size=size, mode="trilinear", align_corners=False)
            for branch in self.branches
        ]
        return x + self.layer_scale * self.fuse(torch.cat([x, *pooled], dim=1))


class WinnerLineageW48UNet(PlainConvUNet):
    """Four-stage W48 U-Net with efficient context and lesion-presence gating."""

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage,
        conv_op,
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes: int,
        n_conv_per_stage_decoder,
        conv_bias: bool = True,
        norm_op=None,
        norm_op_kwargs: dict | None = None,
        dropout_op=None,
        dropout_op_kwargs: dict | None = None,
        nonlin=None,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        pyramid_pool_sizes: Sequence[int] = (1, 2, 4),
        pyramid_branch_channels: int = 40,
        context_blocks: int = 2,
        bottleneck_ratio: float = 0.5,
        presence_gate: bool = True,
    ) -> None:
        if conv_op is not nn.Conv3d:
            raise ValueError("WinnerLineageW48UNet is a 3D-only architecture")
        if n_stages != 4:
            raise ValueError("WinnerLineageW48UNet must retain four resolution stages")
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            nonlin_first=nonlin_first,
        )
        bottleneck_channels = int(self.encoder.output_channels[-1])
        self.pyramid_context = PyramidContext3D(
            bottleneck_channels,
            pool_sizes=pyramid_pool_sizes,
            branch_channels=pyramid_branch_channels,
        )
        self.context_blocks = nn.Sequential(
            *[
                DepthwiseResidualBottleneck3D(
                    bottleneck_channels,
                    bottleneck_ratio=bottleneck_ratio,
                )
                for _ in range(int(context_blocks))
            ]
        )
        self.presence_gate_enabled = bool(presence_gate and num_classes == 2)
        if self.presence_gate_enabled:
            self.presence_pool = nn.AdaptiveAvgPool3d(1)
            self.presence_projection = nn.Conv3d(
                bottleneck_channels, 1, 1, bias=False
            )
            # Start nearly open (sigmoid(2)=0.88), then learn suppression from
            # the many foreground-free patches and healthy scans.
            self.presence_bias = nn.Parameter(torch.tensor(2.0))

    def _apply_presence_gate(
        self, output: torch.Tensor, gate_log_probability: torch.Tensor
    ) -> torch.Tensor:
        background = output[:, :1]
        foreground = output[:, 1:] + gate_log_probability
        return torch.cat((background, foreground), dim=1)

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        bottleneck = self.context_blocks(self.pyramid_context(skips[-1]))
        skips[-1] = bottleneck
        output = self.decoder(skips)
        if not self.presence_gate_enabled:
            return output
        gate_logits = self.presence_projection(self.presence_pool(bottleneck))
        gate_log_probability = F.logsigmoid(gate_logits + self.presence_bias)
        if isinstance(output, (tuple, list)):
            return [
                self._apply_presence_gate(item, gate_log_probability)
                for item in output
            ]
        return self._apply_presence_gate(output, gate_log_probability)

    def compute_conv_feature_map_size(self, input_size):
        base = super().compute_conv_feature_map_size(input_size)
        bottleneck_size = list(input_size)
        for stride in self.encoder.strides:
            bottleneck_size = [
                int(size // step) for size, step in zip(bottleneck_size, stride)
            ]
        bottleneck_voxels = int(np.prod(bottleneck_size))
        channels = int(self.encoder.output_channels[-1])
        # Conservative activation estimate for context fusion and residual blocks.
        context = bottleneck_voxels * channels * (2 + len(self.pyramid_context.branches))
        return base + np.int64(context)
