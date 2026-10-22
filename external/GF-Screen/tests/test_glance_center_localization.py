import unittest

import torch

from GF_RL_trainer import (
    component_center_targets,
    glance_component_center_loss,
)
from models.glance import Glance_model
from models.GF_Screen_RL import select_center_crops


class GlanceCenterTargetTests(unittest.TestCase):
    def test_every_component_gets_a_center_cell(self):
        labels = torch.zeros((1, 1, 9, 9, 9))
        labels[0, 0, 0, 0, 0] = 1
        labels[0, 0, 4:6, 4:6, 4:6] = 1
        targets, positive_crops = component_center_targets(labels, (5, 5, 5))

        self.assertTrue(bool(positive_crops[0]))
        self.assertEqual(int(targets.sum()), 2)
        self.assertEqual(float(targets[0, 0, 0, 0, 0]), 1.0)
        self.assertEqual(float(targets[0, 0, 2, 2, 2]), 1.0)

    def test_partial_crop_has_gradient_only_at_annotated_center(self):
        labels = torch.zeros((1, 1, 9, 9, 9))
        labels[0, 0, 4, 4, 4] = 1
        logits = torch.zeros((1, 1, 5, 5, 5), requires_grad=True)
        loss, metrics = glance_component_center_loss(
            logits, labels, complete_label=torch.tensor([False])
        )
        loss.backward()

        nonzero = logits.grad.ne(0).nonzero(as_tuple=False)
        self.assertEqual(nonzero.shape[0], 1)
        self.assertEqual(nonzero[0].tolist(), [0, 0, 2, 2, 2])
        self.assertEqual(int(metrics["glance_center_positive_cells"]), 1)
        self.assertEqual(int(metrics["glance_center_negative_cells"]), 0)

    def test_verified_complete_empty_crop_supervises_background(self):
        labels = torch.zeros((1, 1, 9, 9, 9))
        logits = torch.zeros((1, 1, 3, 3, 3), requires_grad=True)
        loss, metrics = glance_component_center_loss(
            logits, labels, complete_label=torch.tensor([True])
        )
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(int(logits.grad.ne(0).sum()), logits.numel())
        self.assertEqual(int(metrics["glance_center_positive_cells"]), 0)
        self.assertEqual(int(metrics["glance_center_negative_cells"]), logits.numel())

    def test_unverified_empty_crop_is_fully_ignored(self):
        labels = torch.zeros((1, 1, 9, 9, 9))
        logits = torch.zeros((1, 1, 3, 3, 3), requires_grad=True)
        loss, metrics = glance_component_center_loss(
            logits, labels, complete_label=torch.tensor([False])
        )
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        self.assertEqual(int(metrics["glance_center_negative_cells"]), 0)


class GlanceCenterCompatibilityTests(unittest.TestCase):
    def test_dense_head_is_optional_and_legacy_state_is_unchanged(self):
        legacy = Glance_model("resnet10")
        localized = Glance_model("resnet10", center_localization=True)
        self.assertFalse(any("center_layer" in key for key in legacy.state_dict()))
        missing = localized.load_state_dict(legacy.state_dict(), strict=False)
        self.assertEqual(
            set(missing.missing_keys),
            {"center_layer.weight", "center_layer.bias"},
        )
        self.assertEqual(missing.unexpected_keys, [])

    def test_center_peak_gate_is_opt_in_and_thresholded(self):
        logits = torch.full((3, 1, 2, 2, 2), -8.0)
        logits[1, 0, 1, 1, 1] = 0.0
        logits[2, 0, 0, 0, 0] = 3.0
        selected = select_center_crops(logits, 0.90)
        self.assertEqual(selected.tolist(), [False, False, True])
        with self.assertRaisesRegex(ValueError, "strictly between"):
            select_center_crops(logits, 1.0)

    def test_forward_can_return_dense_centers_without_changing_default(self):
        model = Glance_model("resnet10", center_localization=True).eval()
        volume = torch.zeros((1, 1, 96, 96, 64))
        with torch.inference_mode():
            default = model(volume)
            augmented = model(volume, return_center_predictions=True)
        self.assertEqual(len(default), 3)
        self.assertEqual(len(augmented), 4)
        torch.testing.assert_close(default[2], augmented[2])
        self.assertEqual(augmented[3].shape, (1, 1, 24, 24, 16))


if __name__ == "__main__":
    unittest.main()
