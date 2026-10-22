#!/usr/bin/env python3
"""Install isolated Tversky continuation configs without new preprocessing."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path


root = Path(os.environ.get("nnUNet_preprocessed", Path(__file__).parent.parent / "nnUNet_preprocessed"))
plans_path = root / "Dataset501_PanCancer" / "nnUNetPlans.json"
plans = json.loads(plans_path.read_text())
source_name = "3d_fullres_S4D2W32_ft"
source = plans["configurations"][source_name]

for name in (
    "3d_fullres_S4D2W32_tversky_b070",
    "3d_fullres_S4D2W32_tversky_b085",
    "3d_fullres_S4D2W32_tversky_b070_bs2",
    "3d_fullres_S4D2W32_tversky_b085_bs2",
    "3d_fullres_S4D2W32_tversky_repro_b070",
    "3d_fullres_S4D2W32_tversky_repro_b085",
):
    plans["configurations"][name] = copy.deepcopy(source)

temporary = plans_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(plans, indent=4) + "\n")
os.replace(temporary, plans_path)
print(f"Installed isolated Tversky configs in {plans_path}")
