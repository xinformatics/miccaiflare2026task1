#!/usr/bin/env python3
"""Add a six-stage STU-Net-B config that reuses the existing 2.4x1.6x1.6 cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


CONFIGURATION = "3d_fullres_STUNetB_PartialBox"
SOURCE_CONFIGURATION = "3d_fullres_S4D2W32"
DATA_IDENTIFIER = "nnUNetPlans_3d_fullres_S4D2W32"
OFFICIAL_STRIDES = [
    [1, 1, 1],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [1, 1, 2],
]


def build_configuration(source: dict) -> dict:
    configuration = json.loads(json.dumps(source))
    configuration["data_identifier"] = DATA_IDENTIFIER
    configuration["batch_size"] = 2
    configuration["architecture"] = {
        # The custom trainer replaces this class with the pinned upstream
        # STU-Net implementation. These kwargs carry the six-stage geometry
        # through nnU-Net's plan/checkpoint contract.
        "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        "arch_kwargs": {
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 512, 512],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [[3, 3, 3]] * 6,
            "strides": OFFICIAL_STRIDES,
            "n_conv_per_stage": [1] * 6,
            "n_conv_per_stage_decoder": [1] * 5,
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
    }
    return configuration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plans",
        type=Path,
        default=Path(os.environ["nnUNet_preprocessed"])
        / "Dataset501_PanCancer"
        / "nnUNetPlans.json",
    )
    args = parser.parse_args()
    plans = json.loads(args.plans.read_text(encoding="utf-8"))
    candidate = build_configuration(plans["configurations"][SOURCE_CONFIGURATION])
    existing = plans["configurations"].get(CONFIGURATION)
    if existing is not None and existing != candidate:
        raise ValueError(f"existing {CONFIGURATION} differs from the pinned recipe")
    plans["configurations"][CONFIGURATION] = candidate
    temporary = args.plans.with_name(f".{args.plans.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(plans, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, args.plans)
    cache = args.plans.parent / DATA_IDENTIFIER
    if not cache.is_dir():
        raise FileNotFoundError(f"preprocessed cache is missing: {cache}")
    print(
        json.dumps(
            {
                "configuration": CONFIGURATION,
                "data_identifier": DATA_IDENTIFIER,
                "patch_size": candidate["patch_size"],
                "spacing": candidate["spacing"],
                "batch_size": candidate["batch_size"],
                "strides": OFFICIAL_STRIDES,
                "cache": str(cache.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
