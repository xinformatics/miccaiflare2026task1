"""Add a ResidualEncoderUNet config sized to the efficiency budget we actually have.

Why: the S4D2W32 lineage we inherited runs 5.6M parameters while using only 60% of
the 4096MB serving budget and 15% of the 60s budget. The hidden leaderboard tops out
at 0.71 DSC and the 0.59 solution used a 104M-parameter SwinUNETR. We were bringing
a 19x smaller model to that fight.

Measured on this box (code/measure_train_cost.py, GPU idle):

    config              params   train VRAM  s/epoch   inference (budget 4096MB/60s)
    Plain S4 W32 (old)   5.60M      6994MB     165s     2461MB /  9.3s
    ResEnc S6 W32       107.47M    10271MB     276s     3296MB / 15.0s

So 19x the capacity costs +3.3GB training VRAM (fits 16GB at batch 2) and 1.67x the
epoch time, and still serves inside both limits even after doubling inference time
for the Quadro RTX 5000 eval machine.

Spacing and patch are unchanged from the S4D2W32 config, so this REUSES the existing
203GB preprocessed cache -- no second preprocessing pass.

LR: 1e-3 rather than nnU-Net's default 1e-2. Both the FLARE24 and FLARE25 winning
teams independently deviated to 1e-3 on this task; the FLARE24 README lists it
explicitly as one of only four changes from stock nnU-Net.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import arch  # noqa: E402

CONFIG_NAME = "3d_fullres_ResEncS6"
N_STAGES = 6
FEATURES = [32, 64, 128, 256, 320, 320]
BLOCKS_PER_STAGE = [1, 3, 4, 6, 6, 6]          # nnU-Net ResEnc preset depths
PLANS = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_PanCancer/nnUNetPlans.json"


def main():
    plans = json.loads(PLANS.read_text())
    cfg = arch.build_config(plans["configurations"]["3d_fullres"])
    cfg["architecture"] = {
        "network_class_name":
            "dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet",
        "arch_kwargs": {
            "n_stages": N_STAGES,
            "features_per_stage": FEATURES,
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [[3, 3, 3]] * N_STAGES,
            "strides": [[1, 1, 1]] + [[2, 2, 2]] * (N_STAGES - 1),
            "n_blocks_per_stage": BLOCKS_PER_STAGE,
            "n_conv_per_stage_decoder": [2] * (N_STAGES - 1),
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None, "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
    }
    # same spacing as S4D2W32 -> same preprocessed data, no second cache
    cfg["data_identifier"] = "nnUNetPlans_3d_fullres_S4D2W32"
    plans["configurations"][CONFIG_NAME] = cfg
    PLANS.write_text(json.dumps(plans, indent=4))
    print(f"added '{CONFIG_NAME}' to {PLANS}")

    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    cm = PlansManager(json.loads(PLANS.read_text())).get_configuration(CONFIG_NAME)
    ak = cm.network_arch_init_kwargs
    net = get_network_from_plans(cm.network_arch_class_name, ak,
                                 cm.network_arch_init_kwargs_req_import, 1, 2, True)
    n = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"resolved: {cm.network_arch_class_name.split('.')[-1]}")
    print(f"          n_stages={ak['n_stages']} feats={ak['features_per_stage']}")
    print(f"          blocks={ak.get('n_blocks_per_stage')}")
    print(f"          patch={cm.patch_size} spacing={cm.spacing} batch={cm.batch_size}")
    print(f"          data_identifier={cm.data_identifier}")
    print(f"          PARAMS={n:.2f}M")
    assert 100 < n < 115, f"expected ~107M params, got {n:.2f}M"
    cache = PLANS.parent / cm.data_identifier
    assert cache.is_dir(), f"preprocessed cache missing: {cache}"
    print(f"OK: 107M ResEnc confirmed, reusing cache at {cache}")


if __name__ == "__main__":
    main()
