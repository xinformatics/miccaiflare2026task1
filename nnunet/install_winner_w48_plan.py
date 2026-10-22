#!/usr/bin/env python3
"""Install the W48 winner-lineage configuration using the existing cache."""

from __future__ import annotations

import copy
import json
from pathlib import Path


CONFIGURATION = "3d_fullres_WinnerW48"


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    plans_path = project / "nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans.json"
    plans = json.loads(plans_path.read_text())
    source = plans["configurations"]["3d_fullres_S4D2W32"]
    config = copy.deepcopy(source)
    config["batch_size"] = 8
    config["architecture"] = {
        "network_class_name": "nnunet.winner_lineage_w48.WinnerLineageW48UNet",
        "arch_kwargs": {
            "n_stages": 4,
            "features_per_stage": [48, 96, 192, 320],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [[3, 3, 3]] * 4,
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
            "n_conv_per_stage": [2, 2, 2, 2],
            "n_conv_per_stage_decoder": [2, 2, 2],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
            "pyramid_pool_sizes": [1, 2, 4],
            "pyramid_branch_channels": 40,
            "context_blocks": 2,
            "bottleneck_ratio": 0.5,
            "presence_gate": True,
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
    }
    plans["configurations"][CONFIGURATION] = config
    plans_path.write_text(json.dumps(plans, indent=4) + "\n")
    print(f"Installed {CONFIGURATION} in {plans_path}")
    print(f"Cache: {config['data_identifier']}")
    print(f"Spacing: {config['spacing']}; patch: {config['patch_size']}; batch: {config['batch_size']}")


if __name__ == "__main__":
    main()
