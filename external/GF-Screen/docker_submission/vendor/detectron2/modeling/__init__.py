"""Minimal Detectron2 modeling API used by BiomedParse."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

from torch import nn

from detectron2.layers import ShapeSpec


class Registry:
    def __init__(self, name: str):
        self.name = name
        self.objects = {}

    def register(self, value=None):
        def decorate(item):
            if item.__name__ in self.objects:
                raise KeyError(f"duplicate {self.name} registration: {item.__name__}")
            self.objects[item.__name__] = item
            return item

        return decorate if value is None else decorate(value)

    def get(self, name: str):
        return self.objects[name]


class Backbone(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, value):
        raise NotImplementedError

    @property
    def size_divisibility(self) -> int:
        return 0


BACKBONE_REGISTRY = Registry("BACKBONE")
SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")

__all__ = [
    "BACKBONE_REGISTRY",
    "SEM_SEG_HEADS_REGISTRY",
    "Backbone",
    "Registry",
    "ShapeSpec",
]

