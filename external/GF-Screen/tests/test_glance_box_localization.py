import unittest

import torch

from GF_RL_trainer import component_box_targets, glance_component_box_loss
from models.glance import Glance_model


class GlanceBoxTargetTests(unittest.TestCase):
    def test_nearest_component_to_crop_center_is_selected(self):
        labels = torch.zeros((1, 1, 9, 9, 9))
        labels[0, 0, 0, 0, 0] = 1
        labels[0, 0, 4:6, 3:5, 4:6] = 1
        targets, valid = component_box_targets(labels)

        self.assertTrue(bool(valid[0]))
        expected = torch.tensor(
            [[4 / 8, 3 / 8, 4 / 8, 5 / 8, 4 / 8, 5 / 8]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(targets, expected)

    def test_empty_crop_has_no_target(self):
        targets, valid = component_box_targets(torch.zeros((2, 1, 5, 6, 7)))
        self.assertFalse(bool(valid.any()))
        torch.testing.assert_close(targets, torch.zeros((2, 6)))

    def test_box_loss_has_gradient_only_for_annotated_positive_crop(self):
        labels = torch.zeros((2, 1, 9, 9, 9))
        labels[0, 0, 1:3, 2:5, 3:7] = 1
        logits = torch.zeros((2, 6), requires_grad=True)
        loss, metrics = glance_component_box_loss(logits, labels)
        loss.backward()

        self.assertGreater(float(logits.grad[0].abs().sum()), 0.0)
        torch.testing.assert_close(logits.grad[1], torch.zeros(6), rtol=0, atol=0)
        self.assertEqual(int(metrics["glance_box_positive_crops"]), 1)
        self.assertTrue(torch.isfinite(metrics["glance_box_iou"]))
        self.assertTrue(torch.isfinite(metrics["glance_box_loss"]))
        self.assertAlmostEqual(float(metrics["glance_box_loss"]), float(loss.detach()))


class GlanceBoxCompatibilityTests(unittest.TestCase):
    def test_default_state_dict_is_unchanged_and_auxiliary_head_is_optional(self):
        legacy = Glance_model("resnet10")
        localized = Glance_model("resnet10", box_localization=True)
        self.assertFalse(any("box_layer" in key for key in legacy.state_dict()))
        missing = localized.load_state_dict(legacy.state_dict(), strict=False)
        self.assertEqual(
            set(missing.missing_keys),
            {"box_layer.weight", "box_layer.bias"},
        )
        self.assertEqual(missing.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
