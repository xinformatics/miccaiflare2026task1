#!/usr/bin/env python3
"""Launch the conditional source-stratified alpha=0.20/beta=0.80 trainer."""

from __future__ import annotations

import run_pancancer_stunet as base
from STUNetTrainerPanCancerSourceAggressive import (
    STUNetTrainerPanCancerComponentBoxE160SourceAggressive,
)


TRAINER_NAME = "STUNetTrainerPanCancerComponentBoxE160SourceAggressive"


def resolve_trainer(folder, class_name, current_module):
    if class_name == TRAINER_NAME:
        return STUNetTrainerPanCancerComponentBoxE160SourceAggressive
    return base.resolve_trainer(folder, class_name, current_module)


base.run_training_module.recursive_find_python_class = resolve_trainer
base.run_training_module.load_pretrained_weights = base.load_stunet_pretrained_weights


if __name__ == "__main__":
    base.run_training_module.run_training_entry()
