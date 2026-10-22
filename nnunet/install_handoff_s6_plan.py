#!/usr/bin/env python3
"""Install the handoff's exact 107M ResEnc S6 plan for the local Dataset 501."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path


DATASET = "Dataset501_PanCancerCT"
CONFIGURATION = "3d_fullres_ResEncS6"
PLANS_NAME = "nnUNetS6Plans"
DATA_IDENTIFIER = "nnUNetS6Plans_3d_fullres_2p4_1p6"


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    supplied = project / "handoff_bundle" / "plans" / "nnUNetPlans.json"
    local_dir = Path(os.environ["nnUNet_preprocessed"]) / DATASET
    local_plan = local_dir / "nnUNetResEncUNetLPlans.json"
    output = local_dir / f"{PLANS_NAME}.json"

    source = json.loads(supplied.read_text())
    local = json.loads(local_plan.read_text())
    config = copy.deepcopy(source["configurations"][CONFIGURATION])
    config["data_identifier"] = DATA_IDENTIFIER

    # Keep the handoff geometry/architecture verbatim, while using this machine's
    # already fingerprinted intensity statistics and local dataset identity.
    plans = copy.deepcopy(local)
    plans["dataset_name"] = DATASET
    plans["plans_name"] = PLANS_NAME
    plans["experiment_planner_used"] = "HandoffResEncS6"
    plans["configurations"] = {CONFIGURATION: config}
    output.write_text(json.dumps(plans, indent=4) + "\n")

    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    manager = PlansManager(plans).get_configuration(CONFIGURATION)
    kwargs = manager.network_arch_init_kwargs
    network = get_network_from_plans(
        manager.network_arch_class_name,
        kwargs,
        manager.network_arch_init_kwargs_req_import,
        1,
        2,
        True,
    )
    parameters = sum(parameter.numel() for parameter in network.parameters())
    assert manager.spacing == [2.4, 1.6, 1.6]
    assert manager.patch_size == [128, 160, 160]
    assert kwargs["n_stages"] == 6
    assert kwargs["features_per_stage"] == [32, 64, 128, 256, 320, 320]
    assert kwargs["n_blocks_per_stage"] == [1, 3, 4, 6, 6, 6]
    assert 107_000_000 < parameters < 108_000_000

    print(f"Installed: {output}")
    print(f"Configuration: {CONFIGURATION}")
    print(f"Cache: {local_dir / DATA_IDENTIFIER}")
    print(f"Spacing: {manager.spacing}; patch: {manager.patch_size}; batch: {manager.batch_size}")
    print(f"Parameters: {parameters:,}")


if __name__ == "__main__":
    main()
