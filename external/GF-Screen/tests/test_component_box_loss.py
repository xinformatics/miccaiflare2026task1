import unittest
from types import SimpleNamespace

import torch

from GF_RL_trainer import (
    bbox_partial_label_loss,
    component_box_trusted_mask,
    hybrid_cohort_label_loss,
)


class ComponentBoxTrustedMaskTests(unittest.TestCase):
    def test_separated_components_do_not_trust_the_bridge(self):
        target = torch.zeros((1, 10, 10, 10))
        target[0, 1, 1, 1] = 1
        target[0, 8, 8, 8] = 1
        trusted, mean_components = component_box_trusted_mask(target, (1, 1, 1))

        self.assertTrue(bool(trusted[0, 1, 1, 1]))
        self.assertTrue(bool(trusted[0, 8, 8, 8]))
        self.assertFalse(bool(trusted[0, 5, 5, 5]))
        self.assertEqual(float(mean_components), 2.0)

    def test_diagonal_voxels_are_one_26_connected_component(self):
        target = torch.zeros((1, 6, 6, 6))
        target[0, 2, 2, 2] = 1
        target[0, 3, 3, 3] = 1
        _, mean_components = component_box_trusted_mask(target, (0, 0, 0))
        self.assertEqual(float(mean_components), 1.0)

    def test_empty_crop_is_fully_trusted_by_geometry_builder(self):
        target = torch.zeros((1, 4, 5, 6))
        trusted, mean_components = component_box_trusted_mask(target, (1, 1, 1))
        self.assertTrue(bool(trusted.all()))
        self.assertEqual(float(mean_components), 0.0)


class ComponentBoxGradientTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = dict(
            space_x=1.0,
            space_y=1.0,
            space_z=1.0,
            bbox_context_mm=1.0,
            component_boxes=True,
            tversky_alpha=0.3,
            tversky_beta=0.7,
            boundary_weight=0.15,
            teacher_consistency_weight=0.15,
            pseudo_positive_threshold=0.98,
            pseudo_negative_threshold=0.05,
            bbox_pseudo_positive_weight=0.0,
            bbox_empty_pseudo_positive_weight=0.0,
            bbox_empty_pseudo_positive_min_voxels=2,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_inter_component_bridge_has_exactly_zero_gradient(self):
        logits = torch.zeros((1, 2, 10, 10, 10), requires_grad=True)
        labels = torch.zeros((1, 1, 10, 10, 10))
        labels[0, 0, 1, 1, 1] = 1
        labels[0, 0, 8, 8, 8] = 1
        args = self._args()
        loss, metrics = bbox_partial_label_loss(logits, labels, args)
        loss.backward()

        self.assertEqual(float(logits.grad[0, :, 5, 5, 5].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[0, :, 1, 1, 1].abs().sum()), 0.0)
        self.assertEqual(float(metrics["components_per_crop"]), 2.0)

    def test_pseudo_positive_term_only_promotes_teacher_foreground_outside_box(self):
        logits = torch.zeros((1, 2, 7, 7, 7), requires_grad=True)
        labels = torch.zeros((1, 1, 7, 7, 7))
        labels[0, 0, 2, 2, 2] = 1
        teacher = torch.full((1, 2, 7, 7, 7), -8.0)
        teacher[:, 0] = 8.0

        # This voxel is inside the one-voxel context box and must not receive
        # the directional pseudo-positive BCE term. The distant voxel is the
        # missing-lesion hypothesis that the term is designed to promote.
        inside = (2, 2, 3)
        outside = (5, 5, 5)
        for coordinate in (inside, outside):
            teacher[(0, 0, *coordinate)] = -8.0
            teacher[(0, 1, *coordinate)] = 8.0

        args = self._args(
            bbox_pseudo_positive_weight=0.1,
            teacher_consistency_weight=0.0,
        )
        loss, metrics = bbox_partial_label_loss(logits, labels, args, teacher)
        loss.backward()

        self.assertEqual(int(metrics["teacher_positive_inside_box_voxels"]), 1)
        self.assertEqual(int(metrics["teacher_positive_outside_box_voxels"]), 1)
        self.assertGreater(float(metrics["pseudo_positive_bce"]), 0.0)
        self.assertEqual(float(logits.grad[(0, slice(None), *inside)].abs().sum()), 0.0)
        self.assertLess(float(logits.grad[(0, 1, *outside)]), 0.0)
        self.assertGreater(float(logits.grad[(0, 0, *outside)]), 0.0)

    def test_pseudo_positive_weight_zero_is_exactly_inert_outside_box(self):
        logits = torch.zeros((1, 2, 7, 7, 7), requires_grad=True)
        labels = torch.zeros((1, 1, 7, 7, 7))
        labels[0, 0, 2, 2, 2] = 1
        teacher = torch.full((1, 2, 7, 7, 7), -8.0)
        teacher[:, 0] = 8.0
        outside = (5, 5, 5)
        teacher[(0, 0, *outside)] = -8.0
        teacher[(0, 1, *outside)] = 8.0

        args = self._args(teacher_consistency_weight=0.0)
        loss, metrics = bbox_partial_label_loss(logits, labels, args, teacher)
        loss.backward()

        self.assertEqual(int(metrics["teacher_positive_outside_box_voxels"]), 1)
        self.assertEqual(float(logits.grad[(0, slice(None), *outside)].abs().sum()), 0.0)

    def test_empty_cancer_pseudo_positive_requires_minimum_island(self):
        logits = torch.zeros((3, 2, 5, 5, 5), requires_grad=True)
        labels = torch.zeros((3, 1, 5, 5, 5))
        teacher = torch.full((3, 2, 5, 5, 5), -8.0)
        teacher[:, 0] = 8.0
        qualifying = ((1, 1, 1), (1, 1, 2))
        too_small = (2, 2, 2)
        healthy_teacher_fp = (3, 3, 3)
        for batch, coordinate in (
            *((0, coordinate) for coordinate in qualifying),
            (1, too_small),
            (2, healthy_teacher_fp),
        ):
            teacher[(batch, 0, *coordinate)] = -8.0
            teacher[(batch, 1, *coordinate)] = 8.0

        args = self._args(
            bbox_empty_pseudo_positive_weight=0.1,
            teacher_consistency_weight=0.0,
        )
        loss, metrics = bbox_partial_label_loss(
            logits,
            labels,
            args,
            teacher,
            trusted_empty=torch.tensor([False, False, True]),
        )
        loss.backward()

        self.assertEqual(int(metrics["cancer_negative_crops"]), 2)
        self.assertEqual(int(metrics["teacher_positive_empty_cancer_crops"]), 1)
        self.assertEqual(int(metrics["teacher_positive_empty_cancer_voxels"]), 2)
        self.assertLess(float(logits.grad[(0, 1, *qualifying[0])]), 0.0)
        self.assertEqual(float(logits.grad[(1, slice(None), *too_small)].abs().sum()), 0.0)
        # The proxy-healthy stream remains trusted ground-truth background,
        # regardless of a teacher false positive.
        self.assertGreater(float(logits.grad[(2, 1, *healthy_teacher_fp)]), 0.0)

    def test_positive_only_pseudo_background_requires_teacher_safety(self):
        logits = torch.zeros((1, 2, 7, 7, 7), requires_grad=True)
        labels = torch.zeros((1, 1, 7, 7, 7))
        labels[0, 0, 3, 3, 3] = 1
        teacher = torch.zeros((1, 2, 7, 7, 7))  # uncertain everywhere
        safe_background = (3, 3, 4)  # inside the geometric target box
        teacher[(0, 0, *safe_background)] = 8.0
        teacher[(0, 1, *safe_background)] = -8.0
        uncertain_background = (3, 4, 3)

        args = self._args(teacher_consistency_weight=0.0)
        loss, metrics = bbox_partial_label_loss(
            logits,
            labels,
            args,
            teacher_logits=teacher,
            trusted_empty=torch.tensor([False]),
            positive_only=torch.tensor([True]),
        )
        loss.backward()

        self.assertEqual(int(metrics["positive_only_crops"]), 1)
        self.assertGreater(
            float(logits.grad[(0, slice(None), *safe_background)].abs().sum()), 0.0
        )
        self.assertEqual(
            float(logits.grad[(0, slice(None), *uncertain_background)].abs().sum()),
            0.0,
        )
        self.assertGreater(float(logits.grad[0, :, 3, 3, 3].abs().sum()), 0.0)

    def test_positive_only_without_teacher_has_no_background_gradient(self):
        logits = torch.zeros((1, 2, 5, 5, 5), requires_grad=True)
        labels = torch.zeros((1, 1, 5, 5, 5))
        labels[0, 0, 2, 2, 2] = 1
        loss, _ = bbox_partial_label_loss(
            logits,
            labels,
            self._args(teacher_consistency_weight=0.0),
            positive_only=torch.tensor([True]),
        )
        loss.backward()
        self.assertGreater(float(logits.grad[0, :, 2, 2, 2].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[0, :, 2, 2, 3].abs().sum()), 0.0)

    def test_positive_only_empty_crop_remains_unknown_without_teacher(self):
        logits = torch.zeros((1, 2, 5, 5, 5), requires_grad=True)
        loss, metrics = bbox_partial_label_loss(
            logits,
            torch.zeros((1, 1, 5, 5, 5)),
            self._args(teacher_consistency_weight=0.0),
            trusted_empty=torch.tensor([False]),
            positive_only=torch.tensor([True]),
        )
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)
        self.assertEqual(int(metrics["positive_only_crops"]), 1)
        self.assertEqual(int(metrics["teacher_safe_negative_voxels"]), 0)


class HybridCohortLossTests(unittest.TestCase):
    @staticmethod
    def _args():
        return ComponentBoxGradientTests._args(teacher_consistency_weight=0.0)

    @staticmethod
    def _full_loss(logits, labels):
        probability = logits.softmax(1)[:, 1]
        return ((probability - labels[:, 0].float()) ** 2).mean()

    def test_complete_crop_has_full_volume_gradient_partial_crop_does_not(self):
        logits = torch.zeros((2, 2, 7, 7, 7), requires_grad=True)
        labels = torch.zeros((2, 1, 7, 7, 7))
        labels[:, 0, 2, 2, 2] = 1
        loss, metrics = hybrid_cohort_label_loss(
            logits,
            labels,
            complete_label=torch.tensor([True, False]),
            full_loss_func=self._full_loss,
            args=self._args(),
        )
        loss.backward()

        outside = (6, 6, 6)
        self.assertGreater(float(logits.grad[(0, slice(None), *outside)].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[(1, slice(None), *outside)].abs().sum()), 0.0)
        self.assertEqual(int(metrics["hybrid_complete_crops"]), 1)
        self.assertEqual(int(metrics["hybrid_partial_crops"]), 1)

    def test_all_partial_matches_existing_bbox_objective_exactly(self):
        logits_a = torch.randn((2, 2, 6, 6, 6), requires_grad=True)
        logits_b = logits_a.detach().clone().requires_grad_(True)
        labels = torch.zeros((2, 1, 6, 6, 6))
        labels[:, 0, 2, 2, 2] = 1
        args = self._args()
        expected, _ = bbox_partial_label_loss(logits_a, labels, args)
        actual, metrics = hybrid_cohort_label_loss(
            logits_b,
            labels,
            complete_label=torch.tensor([False, False]),
            full_loss_func=self._full_loss,
            args=args,
        )
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(int(metrics["hybrid_complete_crops"]), 0)
        self.assertEqual(int(metrics["hybrid_partial_crops"]), 2)


if __name__ == "__main__":
    unittest.main()
