import unittest

import torch

from GF_RL_trainer import (
    glance_ignored_distillation_loss,
    teacher_safe_glance_loss,
)
from models.GF_Screen_RL import (
    masked_group_relative_advantages,
    teacher_safe_glance_masks,
)


def binary_logits(probability):
    probability = torch.as_tensor(probability, dtype=torch.float32)
    return torch.stack(
        ((1.0 - probability).clamp_min(1e-6).log(), probability.clamp_min(1e-6).log()),
        dim=1,
    )


class TeacherSafeGlanceTests(unittest.TestCase):
    def setUp(self):
        teacher_probability = torch.full((4, 2, 2, 2), 0.60)
        teacher_probability[0] = 0.02  # annotated positives remain positive
        teacher_probability[1] = 0.10
        teacher_probability[1].flatten()[:4] = 0.99  # sizeable pseudo positive
        teacher_probability[2] = 0.02  # confident empty crop
        teacher_probability[3] = 0.60  # uncertain: neither positive nor empty
        self.teacher_logits = binary_logits(teacher_probability)
        self.annotated = torch.tensor([1, 0, 0, 0])

    def test_partition_is_positive_pseudo_negative_and_ignored(self):
        annotated, pseudo, negative, valid, positive_voxels = teacher_safe_glance_masks(
            self.annotated,
            self.teacher_logits,
            pseudo_positive_threshold=0.95,
            pseudo_negative_threshold=0.50,
            pseudo_positive_min_voxels=3,
        )
        torch.testing.assert_close(annotated, torch.tensor([True, False, False, False]))
        torch.testing.assert_close(pseudo, torch.tensor([False, True, False, False]))
        torch.testing.assert_close(negative, torch.tensor([False, False, True, False]))
        torch.testing.assert_close(valid, torch.tensor([True, True, True, False]))
        self.assertEqual(int(positive_voxels[1]), 4)

    def test_ignored_crop_has_exactly_zero_classification_gradient(self):
        actions = torch.zeros((4, 2), requires_grad=True)
        loss, metrics = teacher_safe_glance_loss(
            actions,
            self.annotated,
            self.teacher_logits,
            pseudo_positive_threshold=0.95,
            pseudo_negative_threshold=0.50,
            pseudo_positive_min_voxels=3,
        )
        loss.backward()
        self.assertGreater(float(actions.grad[:3].abs().sum()), 0.0)
        torch.testing.assert_close(actions.grad[3], torch.zeros(2), rtol=0, atol=0)
        self.assertEqual(int(metrics["glance_annotated_positive_crops"]), 1)
        self.assertEqual(int(metrics["glance_pseudo_positive_crops"]), 1)
        self.assertEqual(int(metrics["glance_safe_negative_crops"]), 1)
        self.assertEqual(int(metrics["glance_ignored_crops"]), 1)

    def test_masked_group_advantage_uses_only_trusted_crops(self):
        actual = masked_group_relative_advantages(
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.tensor([True, True, False, False]),
        )
        torch.testing.assert_close(
            actual, torch.tensor([1.0, -1.0, 0.0, 0.0]), rtol=0, atol=0
        )

    def test_complete_mask_overrides_teacher_on_annotation_negative_crop(self):
        actions = torch.zeros((4, 2), requires_grad=True)
        loss, metrics = teacher_safe_glance_loss(
            actions,
            self.annotated,
            self.teacher_logits,
            pseudo_positive_threshold=0.95,
            pseudo_negative_threshold=0.50,
            pseudo_positive_min_voxels=3,
            complete_label=torch.tensor([False, True, False, True]),
        )
        loss.backward()

        # Crop 1 would be a teacher pseudo positive and crop 3 would be
        # ignored. Their complete masks make both trusted negatives instead.
        self.assertEqual(int(metrics["glance_pseudo_positive_crops"]), 0)
        self.assertEqual(int(metrics["glance_complete_safe_negative_crops"]), 2)
        self.assertEqual(int(metrics["glance_safe_negative_crops"]), 3)
        self.assertEqual(int(metrics["glance_ignored_crops"]), 0)
        self.assertGreater(float(actions.grad[1].abs().sum()), 0.0)
        self.assertGreater(float(actions.grad[3].abs().sum()), 0.0)

    def test_no_complete_metadata_preserves_exact_loss(self):
        actions_a = torch.randn((4, 2), requires_grad=True)
        actions_b = actions_a.detach().clone().requires_grad_(True)
        expected, expected_metrics = teacher_safe_glance_loss(
            actions_a,
            self.annotated,
            self.teacher_logits,
            pseudo_positive_threshold=0.95,
            pseudo_negative_threshold=0.50,
            pseudo_positive_min_voxels=3,
        )
        actual, actual_metrics = teacher_safe_glance_loss(
            actions_b,
            self.annotated,
            self.teacher_logits,
            pseudo_positive_threshold=0.95,
            pseudo_negative_threshold=0.50,
            pseudo_positive_min_voxels=3,
            complete_label=None,
        )
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(
            int(expected_metrics["glance_safe_negative_crops"]),
            int(actual_metrics["glance_safe_negative_crops"]),
        )

    def test_distillation_has_gradient_only_on_ignored_crop(self):
        actions = torch.zeros((4, 2), requires_grad=True)
        reference = binary_logits(torch.tensor([0.2, 0.3, 0.4, 0.9]))
        loss, metrics = glance_ignored_distillation_loss(
            actions,
            reference,
            self.annotated,
            self.teacher_logits,
            pseudo_positive_threshold=0.95,
            pseudo_negative_threshold=0.50,
            pseudo_positive_min_voxels=3,
        )
        loss.backward()
        torch.testing.assert_close(actions.grad[:3], torch.zeros((3, 2)), rtol=0, atol=0)
        self.assertGreater(float(actions.grad[3].abs().sum()), 0.0)
        self.assertEqual(int(metrics["glance_distilled_crops"]), 1)

    def test_complete_negative_is_not_distilled(self):
        actions = torch.zeros((4, 2), requires_grad=True)
        reference = binary_logits(torch.tensor([0.2, 0.3, 0.4, 0.9]))
        loss, metrics = glance_ignored_distillation_loss(
            actions,
            reference,
            self.annotated,
            self.teacher_logits,
            pseudo_positive_threshold=0.95,
            pseudo_negative_threshold=0.50,
            pseudo_positive_min_voxels=3,
            complete_label=torch.tensor([False, False, False, True]),
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(int(metrics["glance_distilled_crops"]), 0)


if __name__ == "__main__":
    unittest.main()
