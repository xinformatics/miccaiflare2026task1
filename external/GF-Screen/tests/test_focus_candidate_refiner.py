import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from focus_candidate_refiner import (
    expand_focus_input_channels,
    focus_contour_loss,
    focus_discriminative_parameter_groups,
    focus_release_dice_ce_loss,
    focus_state_dict,
    focus_training_loss,
    foreground_logit,
    set_focus_encoder_trainable,
    set_focus_prompt_adapter_trainable,
)
from train_focus_candidate_refiner import (
    FOCUS_CONFIDENCE_FEATURE_NAMES,
    FocusPatchDataset,
    evaluate,
    focus_confidence_features,
)
from torch.utils.data import DataLoader, Dataset
from monai.losses import DiceCELoss


class FocusCandidateRefinerTests(unittest.TestCase):
    def test_focus_evaluator_can_select_above_legacy_contour_ceiling(self):
        class ToyModel(torch.nn.Module):
            def forward(self, images):
                foreground = images[:, :1]
                background = torch.zeros_like(foreground)
                return torch.cat((background, foreground), dim=1)

        class ToyDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                image = torch.full((1, 8, 8, 8), -10.0)
                target = torch.zeros(1, 8, 8, 8)
                if index == 1:
                    return image, target, torch.tensor(0.0), "negative"
                target[:, 2:6, 2:6, 2:6] = 1
                image[:, 2:6, 2:6, 2:6] = 3.0
                image[:, 1:7, 1:7, 1:7] = torch.maximum(
                    image[:, 1:7, 1:7, 1:7], torch.tensor(1.5)
                )
                return image, target, torch.tensor(1.0), "positive"

        metrics, predictions = evaluate(
            ToyModel(),
            DataLoader(ToyDataset(), batch_size=1),
            torch.device("cpu"),
            segmentation_threshold_max=0.95,
        )
        self.assertGreater(metrics["segmentation_threshold"], 0.80)
        self.assertEqual(predictions["segmentation_dice_grid"].shape, (1, 16))
        self.assertEqual(predictions["negative_patch_id"].tolist(), ["negative"])
        self.assertEqual(predictions["negative_nonempty_grid"].shape, (1, 16))
        self.assertEqual(predictions["all_patch_id"].tolist(), ["positive", "negative"])
        self.assertEqual(predictions["all_label"].tolist(), [1.0, 0.0])
        self.assertEqual(
            predictions["confidence_feature_names"].tolist(),
            list(FOCUS_CONFIDENCE_FEATURE_NAMES),
        )
        self.assertEqual(predictions["confidence_features"].shape, (2, 10))
        self.assertTrue(np.isfinite(predictions["confidence_features"]).all())
        np.testing.assert_array_equal(
            predictions["negative_nonempty_grid"].sum(axis=0),
            predictions["negative_nonempty_by_threshold"],
        )
        with self.assertRaisesRegex(ValueError, "0.05 grid point"):
            evaluate(
                ToyModel(),
                DataLoader(ToyDataset(), batch_size=1),
                torch.device("cpu"),
                segmentation_threshold_max=0.83,
            )

    def test_focus_confidence_features_use_last_channel_as_seed(self):
        probabilities = torch.zeros((1, 1, 4, 4, 4))
        probabilities[:, :, 1:3, 1:3, 1:3] = 0.8
        images = torch.zeros((1, 3, 4, 4, 4))
        images[:, -1:, 1:3, 1:3, 1:3] = 1
        features = focus_confidence_features(probabilities, images)
        values = dict(zip(FOCUS_CONFIDENCE_FEATURE_NAMES, features[0].tolist()))
        self.assertAlmostEqual(values["foreground_max"], 0.8, places=6)
        self.assertAlmostEqual(values["seed_probability_mean"], 0.8, places=6)
        self.assertAlmostEqual(values["prediction_0p50_seed_precision"], 1.0)
        self.assertAlmostEqual(values["seed_coverage_0p50"], 1.0)

    def test_extracts_prefixed_focus_state(self):
        tensor = torch.ones(2)
        state = focus_state_dict(
            {"state_dict": {"segmentation_model.layer": tensor, "glance_model.layer": tensor}}
        )
        self.assertEqual(list(state), ["layer"])

    def test_foreground_logit_is_two_class_difference(self):
        logits = torch.zeros((1, 2, 2, 2, 2))
        logits[:, 1] = 3
        self.assertTrue(torch.all(foreground_logit(logits) == 3))

    def test_loss_is_finite_for_mixed_and_all_negative_batches(self):
        logits = torch.randn((2, 2, 4, 4, 4), requires_grad=True)
        targets = torch.zeros((2, 1, 4, 4, 4))
        targets[0, :, 1:3, 1:3, 1:3] = 1
        loss, parts = focus_contour_loss(logits, targets, torch.tensor([1.0, 0.0]))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"dice", "voxel_bce", "negative"})
        loss.backward()
        negative, _ = focus_contour_loss(
            torch.randn((1, 2, 4, 4, 4)),
            torch.zeros((1, 1, 4, 4, 4)),
            torch.zeros(1),
        )
        self.assertTrue(torch.isfinite(negative))

    def test_freeze_encoder_keeps_decoder_trainable(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.swinViT = torch.nn.Linear(2, 2)
                self.decoder1 = torch.nn.Linear(2, 2)

        model = Tiny()
        set_focus_encoder_trainable(model, False)
        self.assertFalse(model.swinViT.weight.requires_grad)
        self.assertTrue(model.decoder1.weight.requires_grad)

    def test_expands_released_input_with_zero_initialized_extra_channels(self):
        weights = {
            "swinViT.patch_embed.proj.weight": torch.randn(48, 1, 2, 2, 2),
            "encoder1.layer.conv1.conv.weight": torch.randn(48, 1, 3, 3, 3),
            "encoder1.layer.conv3.conv.weight": torch.randn(48, 1, 1, 1, 1),
        }
        state = dict(weights)
        expanded = expand_focus_input_channels(state, 3)
        for key, weight in weights.items():
            self.assertEqual(expanded[key].shape[1], 3)
            self.assertTrue(torch.equal(expanded[key][:, :1], weight))
            self.assertEqual(int(torch.count_nonzero(expanded[key][:, 1:])), 0)

    def test_prompt_projection_can_be_reenabled_after_encoder_freeze(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.swinViT = torch.nn.Module()
                self.swinViT.patch_embed = torch.nn.Module()
                self.swinViT.patch_embed.proj = torch.nn.Conv3d(2, 4, 2, 2)
                self.encoder1 = torch.nn.Module()
                self.encoder1.layer = torch.nn.Module()
                self.encoder1.layer.conv1 = torch.nn.Module()
                self.encoder1.layer.conv1.conv = torch.nn.Conv3d(2, 4, 3)
                self.encoder1.layer.conv3 = torch.nn.Module()
                self.encoder1.layer.conv3.conv = torch.nn.Conv3d(2, 4, 1)
                self.decoder1 = torch.nn.Conv3d(4, 2, 1)

        model = Tiny()
        set_focus_encoder_trainable(model, False)
        set_focus_prompt_adapter_trainable(model, True)
        self.assertTrue(model.swinViT.patch_embed.proj.weight.requires_grad)
        self.assertTrue(model.encoder1.layer.conv1.conv.weight.requires_grad)
        self.assertTrue(model.encoder1.layer.conv3.conv.weight.requires_grad)
        self.assertFalse(model.swinViT.patch_embed.proj.bias is None)
        self.assertTrue(model.decoder1.weight.requires_grad)

    def test_discriminative_groups_keep_decoder_first_and_scale_encoder(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.swinViT = torch.nn.Linear(2, 2)
                self.decoder1 = torch.nn.Linear(2, 2)

        model = Tiny()
        groups = focus_discriminative_parameter_groups(model, 0.1)
        self.assertEqual([group["lr_scale"] for group in groups], [1.0, 0.1])
        self.assertEqual(
            {id(parameter) for parameter in groups[0]["params"]},
            {id(model.decoder1.weight), id(model.decoder1.bias)},
        )
        self.assertEqual(
            {id(parameter) for parameter in groups[1]["params"]},
            {id(model.swinViT.weight), id(model.swinViT.bias)},
        )
        single = focus_discriminative_parameter_groups(model, 1.0)
        self.assertEqual(len(single), 1)
        self.assertEqual(single[0]["lr_scale"], 1.0)
        with self.assertRaisesRegex(ValueError, "encoder_lr_scale"):
            focus_discriminative_parameter_groups(model, 0.0)

    def test_prompt_dataset_keeps_seed_binary_and_augments_ct_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.npz"
            image = np.zeros((3, 4, 4, 4), dtype=np.float16)
            image[0] = 0.5
            image[2, 1:3, 1:3, 1:3] = 1
            np.savez_compressed(
                path,
                image=image,
                target=np.zeros((1, 4, 4, 4), dtype=np.uint8),
                label=np.uint8(0),
            )
            dataset = FocusPatchDataset(
                [{"path": str(path), "patch_id": "p"}],
                augment=False,
                input_mode="soft_seed",
            )
            value, _, _, _ = dataset[0]
            self.assertEqual(tuple(value.shape), (2, 4, 4, 4))
            self.assertTrue(torch.all(value[0] == 0.5))
            self.assertEqual(int(value[1].sum()), 8)

    def test_dual_window_prompt_dataset_preserves_channel_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.npz"
            image = np.zeros((3, 4, 4, 4), dtype=np.float16)
            image[0] = 0.25
            image[1] = 0.75
            image[2, 1:3, 1:3, 1:3] = 1
            np.savez_compressed(
                path,
                image=image,
                target=np.zeros((1, 4, 4, 4), dtype=np.uint8),
                label=np.uint8(0),
            )
            dataset = FocusPatchDataset(
                [{"path": str(path), "patch_id": "p"}],
                augment=False,
                input_mode="soft_lung_seed",
            )
            value, _, _, _ = dataset[0]
            self.assertEqual(tuple(value.shape), (3, 4, 4, 4))
            self.assertTrue(torch.all(value[0] == 0.25))
            self.assertTrue(torch.all(value[1] == 0.75))
            self.assertEqual(int(value[2].sum()), 8)

    def test_release_loss_matches_original_gf_dice_ce(self):
        logits = torch.randn((2, 2, 4, 4, 4), requires_grad=True)
        targets = torch.zeros((2, 1, 4, 4, 4))
        targets[0, :, 1:3, 1:3, 1:3] = 1
        labels = torch.tensor([1.0, 0.0])
        actual, parts = focus_release_dice_ce_loss(logits, targets, labels)
        expected = DiceCELoss(
            include_background=True, to_onehot_y=True, softmax=True
        )(logits, targets.long())
        self.assertTrue(torch.allclose(actual, expected))
        self.assertIn("release_dice_ce", parts)
        actual.backward()
        self.assertIsNotNone(logits.grad)

    def test_training_loss_rejects_unknown_recipe(self):
        with self.assertRaises(ValueError):
            focus_training_loss(
                torch.zeros((1, 2, 2, 2, 2)),
                torch.zeros((1, 1, 2, 2, 2)),
                torch.zeros(1),
                "not_a_recipe",
            )


if __name__ == "__main__":
    unittest.main()
    expand_focus_input_channels,
