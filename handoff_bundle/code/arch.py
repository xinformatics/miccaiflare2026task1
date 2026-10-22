"""The S4D2W32 architecture, expressed in nnU-Net 2.6's *new* plans format.

The FLARE24/25 winners publish this config using pre-2.5 keys
(UNet_base_num_features / n_conv_per_stage_encoder / pool_op_kernel_sizes) nested
under `inherits_from: 3d_fullres`. On nnU-Net >=2.5 those keys are DEAD: inheritance
copies the parent's `architecture` dict, and plans_handler only reconstructs from old
keys when `architecture` is absent. Loading their plans verbatim yields a 31.2M-param
6-stage net instead of the intended 5.6M 4-stage one -- verified by parameter count.

So we emit an explicit `architecture` block.

Geometry follows the FLARE25 winner's *preprocessing* (patch 128x160x160 @
2.4x1.6x1.6mm, which they show expands spatial coverage from 96x126x126mm to
307x256x256mm) but keeps the genuinely 4-stage 5.6M network they believed they were
training. Their reported 45.2M parameter count is itself evidence they hit the bug
above and trained the deep net by accident.

Measured on this box (code/sweep_configs.py): 6971MB train / 1737MB inference,
versus 7167MB / 1984MB for the 6-stage at identical geometry.
"""

PATCH_SIZE = [128, 160, 160]       # (z, y, x)
SPACING = [2.4, 1.6, 1.6]          # (z, y, x)
BATCH_SIZE = 2
N_STAGES = 4
FEATURES = [32, 64, 128, 256]

ARCHITECTURE = {
    "network_class_name":
        "dynamic_network_architectures.architectures.unet.PlainConvUNet",
    "arch_kwargs": {
        "n_stages": N_STAGES,
        "features_per_stage": FEATURES,
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[3, 3, 3]] * N_STAGES,
        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "n_conv_per_stage": [2] * N_STAGES,
        "n_conv_per_stage_decoder": [2] * (N_STAGES - 1),
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

CONFIG_NAME = "3d_fullres_S4D2W32"


def build_config(parent: dict) -> dict:
    """Full standalone config dict (no inherits_from, so nothing can be shadowed)."""
    import copy
    cfg = copy.deepcopy(parent)
    cfg.update({
        "data_identifier": f"nnUNetPlans_{CONFIG_NAME}",
        "patch_size": list(PATCH_SIZE),
        "spacing": list(SPACING),
        "batch_size": BATCH_SIZE,
        "architecture": copy.deepcopy(ARCHITECTURE),
        # Deliberate, not inherited. nnU-Net ties 3d_fullres batch_dice to whether a
        # 3d_lowres config happened to be generated (default_experiment_planner.py
        # :477-480) -- incidental for us. Foreground here is ~0.07% of voxels (val
        # profile), so many sampled patches contain no lesion at all and per-sample
        # Dice degenerates on them. Pooling the Dice over the batch keeps the
        # gradient meaningful.
        "batch_dice": True,
    })
    for dead in ("UNet_class_name", "UNet_base_num_features",
                 "n_conv_per_stage_encoder", "n_conv_per_stage_decoder",
                 "num_pool_per_axis", "pool_op_kernel_sizes", "conv_kernel_sizes",
                 "unet_max_num_features", "inherits_from", "next_stage",
                 "previous_stage"):
        cfg.pop(dead, None)
    return cfg
