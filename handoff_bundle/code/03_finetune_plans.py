"""Add a config whose architecture matches the FLARE25 baseline, so its weights load.

The organisers' baseline (Dataset025_FLARE25, 3d_fullres, 5.47M params) measures
DSC 0.4725 / NSD 0.3671 on our public validation set -- a far better starting point
than random init, and it was trained on only 5,328 cases against our 17,575.

nnUNetv2_train -pretrained_weights requires the architecture to match exactly. The
baseline differs from our S4D2W32 in exactly one place:

    baseline strides : [[1,1,1],[2,2,2],[2,2,2],[1,2,2]]
    ours             : [[1,1,1],[2,2,2],[2,2,2],[2,2,2]]

Everything else (4 stages, [32,64,128,256], 3x3x3 kernels, 2 convs/stage,
PlainConvUNet) is identical, and strides do not change parameter counts -- so this
one edit makes the checkpoint loadable.

What we deliberately do NOT match:
  * spacing      baseline 3.0/2.0/2.0  vs ours 2.4/1.6/1.6
  * normalization  baseline fingerprint (mean 12.12, std 221.86)
                   vs ours (mean -44.67, std 245.26)
Matching those would mean re-preprocessing all 17,575 cases (~6h, ~150GB). The
mismatch is a domain shift that fine-tuning absorbs, and our finer spacing is the
better target anyway -- it is what the FLARE25 winner moved to. We keep our cache.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arch import build_config  # noqa: E402

CONFIG_NAME = "3d_fullres_S4D2W32_ft"
BASELINE_STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
PLANS = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_PanCancer/nnUNetPlans.json"
BASELINE = Path("/mnt/yadav/PancancerCTSeg/baseline_model/workspace/nnUNet_results/"
                "Dataset025_FLARE25/nnUNetTrainer__nnUNetPlans__3d_fullres")


def main():
    plans = json.loads(PLANS.read_text())
    cfg = build_config(plans["configurations"]["3d_fullres"])
    cfg["architecture"]["arch_kwargs"]["strides"] = [list(s) for s in BASELINE_STRIDES]
    # reuse the SAME preprocessed data as the from-scratch config: identical spacing,
    # so no second 203GB cache is needed
    cfg["data_identifier"] = "nnUNetPlans_3d_fullres_S4D2W32"
    plans["configurations"][CONFIG_NAME] = cfg
    PLANS.write_text(json.dumps(plans, indent=4))
    print(f"added '{CONFIG_NAME}' to {PLANS}")

    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    cm = PlansManager(json.loads(PLANS.read_text())).get_configuration(CONFIG_NAME)
    ak = cm.network_arch_init_kwargs
    net = get_network_from_plans(cm.network_arch_class_name, ak,
                                 cm.network_arch_init_kwargs_req_import, 1, 2, True)
    n = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"resolved: n_stages={ak['n_stages']} feats={ak['features_per_stage']}")
    print(f"          strides={ak['strides']}")
    print(f"          patch={cm.patch_size} spacing={cm.spacing}")
    print(f"          data_identifier={cm.data_identifier}  PARAMS={n:.2f}M")

    # the real test: does the baseline checkpoint load into it cleanly?
    ck = torch.load(BASELINE / "fold_all/checkpoint_final.pth",
                    map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict(ck["network_weights"], strict=False)
    print(f"\nbaseline checkpoint -> missing {len(missing)}, unexpected {len(unexpected)}")
    if missing or unexpected:
        print("  missing   :", missing[:5])
        print("  unexpected:", unexpected[:5])
        sys.exit("architecture does not match -- pretrained_weights would not transfer")
    print("OK: baseline weights load cleanly; -pretrained_weights will transfer")
    print(f"\nfine-tune with:\n  nnUNetv2_train 501 {CONFIG_NAME} all "
          f"-tr nnUNetTrainer_Epoch500_Lr1e3 \\\n"
          f"    -pretrained_weights {BASELINE}/fold_all/checkpoint_final.pth")


if __name__ == "__main__":
    main()
