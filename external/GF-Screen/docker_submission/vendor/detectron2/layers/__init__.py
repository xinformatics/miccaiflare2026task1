"""Minimal Detectron2 layers used by the frozen BiomedParse graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as functional


class Conv2d(nn.Conv2d):
    """Detectron2-compatible Conv2d with optional norm and activation."""

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = functional.conv2d(
            value,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            value = self.norm(value)
        if self.activation is not None:
            value = self.activation(value)
        return value


class DeformConv(nn.Conv2d):
    """Import-compatible sentinel; absent from the selected inference graph.

    The frozen configuration instantiates ``TransformerEncoderPixelDecoder``,
    which only uses :class:`Conv2d`.  Calling this class is therefore a contract
    violation rather than silently substituting ordinary convolution.
    """

    def forward(self, value: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("DeformConv is not authorized in the frozen BiomedParse graph")


@dataclass
class ShapeSpec:
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


def get_norm(
    norm: Optional[Union[str, Callable[[int], nn.Module]]], channels: int
):
    if norm is None or norm == "":
        return None
    if callable(norm):
        return norm(channels)
    choices = {
        "BN": nn.BatchNorm2d,
        "SyncBN": nn.SyncBatchNorm,
        "GN": lambda count: nn.GroupNorm(32, count),
    }
    if norm not in choices:
        raise KeyError(f"unsupported inference normalization: {norm}")
    return choices[norm](channels)


__all__ = ["Conv2d", "DeformConv", "ShapeSpec", "get_norm"]
