"""Inject the S4D2W32 configuration into the auto-generated nnUNetPlans.json.

Run AFTER `nnUNetv2_plan_experiment -d 501`. Writes the config in nnU-Net 2.6's new
`architecture` format (see arch.py for why the published old-format version is a
silent no-op) and asserts the resulting network really is the 5.6M 4-stage model.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arch import CONFIG_NAME, build_config  # noqa: E402

PLANS = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_PanCancer/nnUNetPlans.json"


def main():
    plans = json.loads(PLANS.read_text())
    cfgs = plans["configurations"]
    if "3d_fullres" not in cfgs:
        sys.exit("no 3d_fullres in plans -- run nnUNetv2_plan_experiment first")

    cfgs[CONFIG_NAME] = build_config(cfgs["3d_fullres"])
    PLANS.write_text(json.dumps(plans, indent=4))
    print(f"added '{CONFIG_NAME}' to {PLANS}")

    # verify what nnU-Net will actually build
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    cm = PlansManager(json.loads(PLANS.read_text())).get_configuration(CONFIG_NAME)
    ak = cm.network_arch_init_kwargs
    net = get_network_from_plans(cm.network_arch_class_name, ak,
                                 cm.network_arch_init_kwargs_req_import, 1, 2, True)
    n = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"resolved: n_stages={ak['n_stages']} feats={ak['features_per_stage']}")
    print(f"          patch={cm.patch_size} spacing={cm.spacing} batch={cm.batch_size}")
    print(f"          data_identifier={cm.data_identifier}")
    print(f"          PARAMS={n:.2f}M")
    assert ak["n_stages"] == 4, f"expected 4 stages, got {ak['n_stages']}"
    assert 5.0 < n < 6.5, f"expected ~5.6M params, got {n:.2f}M"
    print("OK: 4-stage 5.6M network confirmed")

    print("\nfg intensity properties (drives CTNormalization):")
    print(json.dumps(plans["foreground_intensity_properties_per_channel"], indent=1))


if __name__ == "__main__":
    main()
