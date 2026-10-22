#!/usr/bin/env python3
"""Install a SwinUNETR config that reuses the existing 2.4 x 1.6 x 1.6 cache."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path


root = Path(
    os.environ.get(
        "nnUNet_preprocessed",
        Path(__file__).parent.parent / "nnUNet_preprocessed",
    )
)
plans_path = root / "Dataset501_PanCancer" / "nnUNetPlans.json"
plans = json.loads(plans_path.read_text())
source = copy.deepcopy(plans["configurations"]["3d_fullres_S4D2W32_ft"])

source["batch_size"] = 1
source["patch_size"] = [96, 96, 96]
source["batch_dice"] = False
source["architecture"] = {
    "network_class_name": "nnunet.monai_swin.SwinUNETRForNNUNet",
    "arch_kwargs": {
        "patch_size": 2,
        "depths": [2, 2, 2, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "feature_size": 48,
        "norm_name": "instance",
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "dropout_path_rate": 0.0,
        "normalize": True,
        "patch_norm": False,
        "use_checkpoint": True,
        "spatial_dims": 3,
        "downsample": "merging",
        "use_v2": False,
    },
    "_kw_requires_import": [],
}

name = "3d_fullres_MONAI_SwinUNETR_f48_pretrained"
plans["configurations"][name] = source
temporary = plans_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(plans, indent=4) + "\n")
os.replace(temporary, plans_path)
print(f"Installed {name} in {plans_path}")
