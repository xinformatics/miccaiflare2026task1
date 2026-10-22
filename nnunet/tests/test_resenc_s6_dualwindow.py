from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from nnunet.resenc_s6_dualwindow import (
    CT_MEAN_HU,
    CT_STD_HU,
    STEM_ALIAS_WEIGHT_KEY,
    STEM_WEIGHT_KEY,
    DualWindowInputConv3d,
    adapt_parent_network_weights,
    inflate_one_channel_stem,
    load_function_preserving_parent,
    nnUNetTrainerResEncS6PanCancerDualWindow,
    normalized_ct_to_dual_window_channels,
    replace_resenc_stem_with_dual_window,
)
from nnunet.resenc_s6_partialbox import nnUNetTrainerResEncS6PanCancerPartialBox


def normalized(hu: torch.Tensor) -> torch.Tensor:
    return (hu - CT_MEAN_HU) / CT_STD_HU


class ToyResEnc(nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.stem = nn.Module()
        block = nn.Module()
        block.conv = nn.Conv3d(input_channels, 2, 3, padding=1, bias=True)
        block.all_modules = nn.Sequential(block.conv, nn.Identity())
        self.encoder.stem.convs = nn.ModuleList([block])
        self.head = nn.Conv3d(2, 2, 1)

    def forward(self, x):
        x = self.encoder.stem.convs[0].all_modules(x)
        return self.head(x)


class DualWindowTests(unittest.TestCase):
    def test_window_values_match_gf_convention(self):
        hu = torch.tensor([-1000.0, -175.0, 0.0, 250.0, 400.0]).reshape(1, 1, 1, 1, 5)
        channels = normalized_ct_to_dual_window_channels(normalized(hu))
        self.assertEqual(tuple(channels.shape), (1, 3, 1, 1, 5))
        torch.testing.assert_close(channels[:, 0], normalized(hu)[:, 0])
        expected_soft = torch.tensor([0.0, 0.0, 175 / 425, 1.0, 1.0]).reshape(1, 1, 1, 5)
        expected_lung = torch.tensor([0.0, 825 / 1400, 1000 / 1400, 1250 / 1400, 1.0]).reshape(1, 1, 1, 5)
        torch.testing.assert_close(channels[:, 1], expected_soft, atol=1e-6, rtol=0)
        torch.testing.assert_close(channels[:, 2], expected_lung, atol=1e-6, rtol=0)

    def test_input_contract_rejects_non_ct_shapes(self):
        with self.assertRaisesRegex(ValueError, "\[B,1,D,H,W\]"):
            normalized_ct_to_dual_window_channels(torch.zeros(1, 2, 3, 3, 3))

    def test_zero_extra_kernels_preserve_stem_output(self):
        torch.manual_seed(7)
        source = nn.Conv3d(1, 4, 3, padding=1, bias=True)
        inflated = inflate_one_channel_stem(source)
        self.assertIsInstance(inflated, DualWindowInputConv3d)
        self.assertEqual(tuple(inflated.weight.shape), (4, 3, 3, 3, 3))
        self.assertEqual(torch.count_nonzero(inflated.weight[:, 1:]).item(), 0)
        x = torch.randn(2, 1, 5, 6, 7)
        torch.testing.assert_close(inflated(x), source(x), atol=2e-6, rtol=1e-5)

    def test_replace_stem_keeps_state_key_and_parameter_delta(self):
        model = ToyResEnc()
        before = sum(parameter.numel() for parameter in model.parameters())
        replace_resenc_stem_with_dual_window(model)
        after = sum(parameter.numel() for parameter in model.parameters())
        self.assertIn(STEM_WEIGHT_KEY, model.state_dict())
        self.assertIn(STEM_ALIAS_WEIGHT_KEY, model.state_dict())
        self.assertIs(
            model.encoder.stem.convs[0].conv,
            model.encoder.stem.convs[0].all_modules[0],
        )
        self.assertEqual(after - before, 2 * 2 * 3 * 3 * 3)

    def test_parent_adaptation_changes_only_stem_shape(self):
        parent = ToyResEnc(1)
        target = replace_resenc_stem_with_dual_window(ToyResEnc(1))
        adapted = adapt_parent_network_weights(parent.state_dict(), target.state_dict())
        self.assertEqual(tuple(adapted[STEM_WEIGHT_KEY].shape), (2, 3, 3, 3, 3))
        torch.testing.assert_close(
            adapted[STEM_WEIGHT_KEY][:, 0], parent.state_dict()[STEM_WEIGHT_KEY][:, 0]
        )
        self.assertEqual(torch.count_nonzero(adapted[STEM_WEIGHT_KEY][:, 1:]).item(), 0)
        self.assertEqual(
            torch.count_nonzero(adapted[STEM_ALIAS_WEIGHT_KEY][:, 1:]).item(), 0
        )
        for key in adapted:
            if key not in (STEM_WEIGHT_KEY, STEM_ALIAS_WEIGHT_KEY):
                torch.testing.assert_close(adapted[key], parent.state_dict()[key])

    def test_parent_adaptation_fails_closed_on_other_shape_change(self):
        parent = ToyResEnc(1)
        target = replace_resenc_stem_with_dual_window(ToyResEnc(1))
        weights = dict(parent.state_dict())
        weights["head.weight"] = torch.zeros(3, 2, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "head.weight"):
            adapt_parent_network_weights(weights, target.state_dict())

    def test_checkpoint_loader_validates_metadata_and_loads_all_heads(self):
        torch.manual_seed(11)
        parent = ToyResEnc(1)
        target = replace_resenc_stem_with_dual_window(ToyResEnc(1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.pth"
            torch.save(
                {
                    "trainer_name": "nnUNetTrainerResEncS6PanCancerPositiveUnlabeled",
                    "current_epoch": 100,
                    "network_weights": parent.state_dict(),
                },
                path,
            )
            digest = load_function_preserving_parent(target, path)
        self.assertEqual(len(digest), 64)
        torch.testing.assert_close(target.head.weight, parent.head.weight)
        torch.testing.assert_close(target.head.bias, parent.head.bias)
        x = torch.randn(1, 1, 5, 5, 5)
        torch.testing.assert_close(target(x), parent(x), atol=3e-6, rtol=1e-5)

    def test_builder_replaces_only_first_convolution(self):
        fake = ToyResEnc(1)
        with patch.object(
            nnUNetTrainerResEncS6PanCancerPartialBox,
            "build_network_architecture",
            return_value=fake,
        ):
            result = nnUNetTrainerResEncS6PanCancerDualWindow.build_network_architecture(
                "dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet",
                {},
                [],
                1,
                2,
                True,
            )
        self.assertIs(result, fake)
        self.assertIsInstance(result.encoder.stem.convs[0].conv, DualWindowInputConv3d)

    def test_builder_rejects_materialized_multichannel_dataset(self):
        with self.assertRaisesRegex(ValueError, "one preprocessed CT channel"):
            nnUNetTrainerResEncS6PanCancerDualWindow.build_network_architecture(
                "ResidualEncoderUNet", {}, [], 3, 2, True
            )

    def test_training_contract_is_material_not_short_tweak(self):
        self.assertEqual(
            nnUNetTrainerResEncS6PanCancerDualWindow.expected_parameter_count,
            107_468_778,
        )
        self.assertEqual(
            nnUNetTrainerResEncS6PanCancerDualWindow.dual_window_stage_end_epochs,
            (25, 50, 100),
        )


if __name__ == "__main__":
    unittest.main()
