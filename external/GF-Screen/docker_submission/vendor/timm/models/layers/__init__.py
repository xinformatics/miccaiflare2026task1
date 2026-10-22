"""The three timm layer helpers used by the frozen BiomedParse model."""

from __future__ import annotations

from itertools import repeat
from typing import Iterable

import torch
from torch import nn


def to_ntuple(count: int):
    def parse(value):
        if isinstance(value, Iterable) and not isinstance(value, str):
            return value
        return tuple(repeat(value, count))

    return parse


to_2tuple = to_ntuple(2)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def drop_path(value, drop_probability=0.0, training=False, scale_by_keep=True):
    if drop_probability == 0.0 or not training:
        return value
    keep_probability = 1.0 - drop_probability
    shape = (value.shape[0],) + (1,) * (value.ndim - 1)
    random = value.new_empty(shape).bernoulli_(keep_probability)
    if keep_probability > 0.0 and scale_by_keep:
        random.div_(keep_probability)
    return value * random


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, value):
        return drop_path(value, self.drop_prob, self.training, self.scale_by_keep)


__all__ = ["DropPath", "drop_path", "to_2tuple", "to_ntuple", "trunc_normal_"]

