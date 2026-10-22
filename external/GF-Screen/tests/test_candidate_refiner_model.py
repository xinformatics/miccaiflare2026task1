import unittest

import numpy as np
import torch

from candidate_refiner_model import (
    CandidateRefiner,
    candidate_refiner_loss,
    downsample_prompt_weights,
)
from torch.utils.data import DataLoader, Dataset

from train_candidate_refiner import (
    apply_case_balanced_validation_metrics,
    evaluate_with_predictions,
    grouped_fold_rows,
)


class CandidateRefinerTests(unittest.TestCase):
    def test_forward_and_loss(self):
        model = CandidateRefiner(base_channels=4)
        images = torch.rand(2, 3, 32, 32, 24)
        labels = torch.tensor([1.0, 0.0])
        targets = torch.zeros(2, 1, 32, 32, 24)
        targets[0, :, 12:20, 12:20, 8:16] = 1
        class_logits, segmentation_logits = model(images)
        self.assertEqual(class_logits.shape, (2,))
        self.assertEqual(segmentation_logits.shape, targets.shape)
        loss, parts = candidate_refiner_loss(
            class_logits, segmentation_logits, labels, targets
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(parts),
            {"classification", "dice", "voxel_bce", "negative_segmentation"},
        )
        loss.backward()

    def test_prompt_masked_classifier_is_opt_in_and_finite(self):
        legacy = CandidateRefiner(base_channels=4)
        prompted = CandidateRefiner(
            base_channels=4, prompt_masked_classifier=True
        )
        self.assertEqual(legacy.classifier[-1].in_features, 64)
        self.assertEqual(prompted.classifier[-1].in_features, 128)
        images = torch.rand(2, 3, 24, 24, 24)
        images[:, 2] = 0
        images[0, 2, 8:16, 8:16, 8:16] = 1
        # The malformed empty prompt in sample 1 exercises the finite global
        # fallback; audited training/inference patches normally never use it.
        classification, segmentation = prompted(images)
        self.assertEqual(tuple(classification.shape), (2,))
        self.assertEqual(tuple(segmentation.shape), (2, 1, 24, 24, 24))
        self.assertTrue(torch.isfinite(classification).all())
        classification.sum().backward()

    def test_prompt_masked_classifier_requires_valid_channel(self):
        with self.assertRaisesRegex(ValueError, "valid prompt channel"):
            CandidateRefiner(
                in_channels=2, base_channels=4, prompt_masked_classifier=True
            )

    def test_prompt_downsampling_cannot_drop_a_single_voxel_seed(self):
        prompt = torch.zeros(2, 1, 64, 64, 64)
        prompt[0, 0, 1, 2, 3] = 1
        prompt[1, 0, 62, 61, 60] = 1
        weights = downsample_prompt_weights(prompt, (8, 8, 8))
        self.assertEqual(tuple(weights.shape), (2, 1, 8, 8, 8))
        self.assertTrue(torch.all(weights.sum(dim=(1, 2, 3, 4)) > 0))
        self.assertTrue(torch.isfinite(weights).all())
        with self.assertRaisesRegex(ValueError, "Bx1xDxHxW"):
            downsample_prompt_weights(torch.zeros(1, 2, 8, 8, 8), (4, 4, 4))

    def test_uncertain_negative_weight_is_opt_in_and_scale_normalized(self):
        labels = torch.tensor([1.0, 0.0])
        class_logits = torch.tensor([2.0, 2.0])
        targets = torch.zeros(2, 1, 4, 4, 4)
        segmentation_logits = torch.zeros_like(targets)

        default_loss, default_parts = candidate_refiner_loss(
            class_logits, segmentation_logits, labels, targets
        )
        explicit_loss, explicit_parts = candidate_refiner_loss(
            class_logits,
            segmentation_logits,
            labels,
            targets,
            classification_negative_weight=1.0,
        )
        uncertain_loss, uncertain_parts = candidate_refiner_loss(
            class_logits,
            segmentation_logits,
            labels,
            targets,
            classification_negative_weight=0.25,
        )

        self.assertTrue(torch.equal(default_loss, explicit_loss))
        self.assertTrue(
            torch.equal(default_parts["classification"], explicit_parts["classification"])
        )
        self.assertLess(
            uncertain_parts["classification"], default_parts["classification"]
        )
        # The effective-weight normalization keeps the classification loss on
        # a comparable scale instead of multiplying the whole objective by
        # the average sample weight.
        self.assertGreater(float(uncertain_parts["classification"]), 0.4)
        with self.assertRaisesRegex(ValueError, "must be in"):
            candidate_refiner_loss(
                class_logits,
                segmentation_logits,
                labels,
                targets,
                classification_negative_weight=0.0,
            )

    def test_uncertain_negative_segmentation_penalty_is_opt_in(self):
        labels = torch.zeros(1)
        class_logits = torch.zeros(1)
        targets = torch.zeros(1, 1, 4, 4, 4)
        segmentation_logits = torch.ones_like(targets)
        default_loss, default_parts = candidate_refiner_loss(
            class_logits, segmentation_logits, labels, targets
        )
        explicit_loss, explicit_parts = candidate_refiner_loss(
            class_logits,
            segmentation_logits,
            labels,
            targets,
            negative_segmentation_weight=0.1,
        )
        uncertain_loss, uncertain_parts = candidate_refiner_loss(
            class_logits,
            segmentation_logits,
            labels,
            targets,
            negative_segmentation_weight=0.025,
        )
        self.assertTrue(torch.equal(default_loss, explicit_loss))
        self.assertTrue(
            torch.equal(
                default_parts["negative_segmentation"],
                explicit_parts["negative_segmentation"],
            )
        )
        self.assertLess(uncertain_loss, default_loss)
        with self.assertRaisesRegex(ValueError, "negative_segmentation_weight"):
            candidate_refiner_loss(
                class_logits,
                segmentation_logits,
                labels,
                targets,
                negative_segmentation_weight=-0.1,
            )

    def test_grouped_split_has_no_case_leakage(self):
        rows = []
        for cohort in ("a", "b"):
            for case_index in range(10):
                for patch in range(2):
                    rows.append(
                        {
                            "cohort": cohort,
                            "case": f"{cohort}{case_index}",
                            "patch_id": f"{cohort}{case_index}_{patch}",
                        }
                    )
        train, validation = grouped_fold_rows(rows, folds=5, fold=2, seed=7)
        train_cases = {row["case"] for row in train}
        validation_cases = {row["case"] for row in validation}
        self.assertFalse(train_cases & validation_cases)
        self.assertEqual(len(validation_cases), 4)
        self.assertEqual(len(train) + len(validation), len(rows))

    def test_grouped_split_keeps_training_only_rows_out_of_validation(self):
        rows = [
            {
                "case": f"case_{index}",
                "cohort": "labeled",
                "label": str(index % 2),
                "training_only": "0",
            }
            for index in range(10)
        ]
        rows.append(
            {
                "case": "pseudo_case",
                "cohort": "pseudo",
                "label": "1",
                "training_only": "1",
            }
        )
        train, validation = grouped_fold_rows(rows, folds=2, fold=0, seed=7)
        self.assertIn("pseudo_case", {row["case"] for row in train})
        self.assertNotIn("pseudo_case", {row["case"] for row in validation})
        self.assertEqual(len(train) + len(validation), len(rows))

    def test_boundary_weight_emphasizes_contour_errors(self):
        target = torch.zeros(1, 1, 16, 16, 16)
        target[:, :, 4:12, 4:12, 4:12] = 1
        logits = torch.where(target > 0, torch.tensor(5.0), torch.tensor(-5.0))
        dilated = torch.nn.functional.max_pool3d(target, 3, stride=1, padding=1)
        eroded = 1 - torch.nn.functional.max_pool3d(1 - target, 3, stride=1, padding=1)
        boundary = (dilated - eroded) > 0
        logits[boundary] = 0
        _, unweighted = candidate_refiner_loss(
            torch.zeros(1), logits, torch.ones(1), target, boundary_voxel_weight=0
        )
        _, weighted = candidate_refiner_loss(
            torch.zeros(1), logits, torch.ones(1), target, boundary_voxel_weight=3
        )
        self.assertGreater(weighted["voxel_bce"], unweighted["voxel_bce"])

    def test_small_lesion_loss_is_opt_in_and_increases_tiny_lesion_signal(self):
        labels = torch.ones(2)
        targets = torch.zeros(2, 1, 16, 16, 16)
        targets[0, :, 4:12, 4:12, 4:12] = 1
        targets[1, :, 8, 8, 8] = 1
        class_logits = torch.zeros(2)
        segmentation_logits = torch.zeros_like(targets)

        default_loss, default_parts = candidate_refiner_loss(
            class_logits, segmentation_logits, labels, targets
        )
        explicit_legacy_loss, explicit_legacy_parts = candidate_refiner_loss(
            class_logits,
            segmentation_logits,
            labels,
            targets,
            small_lesion_loss=False,
        )
        small_loss, _ = candidate_refiner_loss(
            class_logits,
            segmentation_logits,
            labels,
            targets,
            small_lesion_loss=True,
            small_lesion_positive_weight_cap=100.0,
        )

        self.assertTrue(torch.equal(default_loss, explicit_legacy_loss))
        self.assertTrue(
            torch.equal(default_parts["voxel_bce"], explicit_legacy_parts["voxel_bce"])
        )

        legacy_logits = torch.zeros_like(targets, requires_grad=True)
        legacy_loss, _ = candidate_refiner_loss(
            class_logits, legacy_logits, labels, targets
        )
        legacy_loss.backward()
        small_logits = torch.zeros_like(targets, requires_grad=True)
        small_loss, _ = candidate_refiner_loss(
            class_logits,
            small_logits,
            labels,
            targets,
            small_lesion_loss=True,
            small_lesion_positive_weight_cap=100.0,
        )
        small_loss.backward()
        self.assertGreater(
            abs(float(small_logits.grad[1, 0, 8, 8, 8])),
            abs(float(legacy_logits.grad[1, 0, 8, 8, 8])),
        )

    def test_evaluate_calibrates_segmentation_threshold_from_fold(self):
        class ToyModel(torch.nn.Module):
            def forward(self, images):
                return images[:, 1, 0, 0, 0], images[:, :1]

        class ToyDataset(Dataset):
            def __init__(self, images, targets, labels):
                self.images, self.targets, self.labels = images, targets, labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, index):
                return self.images[index], self.targets[index], self.labels[index], str(index)

        images = torch.full((2, 3, 8, 8, 8), -2.0)
        targets = torch.zeros(2, 1, 8, 8, 8)
        targets[0, :, 2:6, 2:6, 2:6] = 1
        images[0, 0, 2:6, 2:6, 2:6] = 2.0
        images[0, 1, 0, 0, 0] = 4.0
        images[1, 1, 0, 0, 0] = -4.0
        labels = torch.tensor([1.0, 0.0])
        loader = DataLoader(ToyDataset(images, targets, labels), batch_size=2)
        metrics, predictions = evaluate_with_predictions(
            ToyModel(), loader, torch.device("cpu")
        )
        self.assertEqual(metrics["segmentation_threshold"], 0.5)
        self.assertAlmostEqual(metrics["segmentation_dice"], 1.0)
        self.assertAlmostEqual(metrics["segmentation_dice_at_0p5"], 1.0)
        self.assertEqual(predictions["patch_id"].tolist(), ["0", "1"])
        self.assertEqual(predictions["positive_patch_id"].tolist(), ["0"])
        self.assertEqual(predictions["segmentation_dice_grid"].shape, (1, 13))

    def test_evaluate_can_extend_contour_grid_above_legacy_ceiling(self):
        class ToyModel(torch.nn.Module):
            def forward(self, images):
                return images[:, 1, 0, 0, 0], images[:, :1]

        class ToyDataset(Dataset):
            def __getitem__(self, index):
                image = torch.full((3, 8, 8, 8), -10.0)
                target = torch.zeros(1, 8, 8, 8)
                target[:, 2:6, 2:6, 2:6] = 1
                image[0, 2:6, 2:6, 2:6] = 3.0
                image[0, 1:7, 1:7, 1:7] = torch.maximum(
                    image[0, 1:7, 1:7, 1:7], torch.tensor(1.5)
                )
                image[1, 0, 0, 0] = 4.0
                return image, target, torch.tensor(1.0), "positive"

            def __len__(self):
                return 1

        metrics, predictions = evaluate_with_predictions(
            ToyModel(),
            DataLoader(ToyDataset(), batch_size=1),
            torch.device("cpu"),
            segmentation_threshold_max=0.95,
        )
        self.assertGreater(metrics["segmentation_threshold"], 0.80)
        self.assertEqual(predictions["segmentation_dice_grid"].shape, (1, 16))
        self.assertAlmostEqual(float(predictions["segmentation_thresholds"][-1]), 0.95)
        with self.assertRaisesRegex(ValueError, "0.05 grid point"):
            evaluate_with_predictions(
                ToyModel(),
                DataLoader(ToyDataset(), batch_size=1),
                torch.device("cpu"),
                segmentation_threshold_max=0.83,
            )

    def test_checkpoint_metrics_give_each_case_equal_classification_weight(self):
        rows = []
        patch_ids = []
        labels = []
        probabilities = []
        for index in range(20):
            patch_id = f"large__n{index:02d}"
            rows.append({"patch_id": patch_id, "case": "large", "cohort": "a", "label": 0})
            patch_ids.append(patch_id)
            labels.append(0)
            probabilities.append(0.60)
        for patch_id, case, label, probability in (
            ("large__p", "large", 1, 0.90),
            ("small__p", "small", 1, 0.55),
            ("small__n", "small", 0, 0.50),
        ):
            rows.append({"patch_id": patch_id, "case": case, "cohort": "a", "label": label})
            patch_ids.append(patch_id)
            labels.append(label)
            probabilities.append(probability)
        metrics = {
            "threshold": 0.65,
            "precision": 1.0,
            "recall": 0.5,
            "f0p5": 0.8,
            "average_precision": 0.5,
            "roc_auc": 0.5,
            "segmentation_threshold": 0.5,
            "segmentation_dice": 0.5,
        }
        predictions = {
            "patch_id": np.asarray(patch_ids),
            "label": np.asarray(labels, dtype=np.uint8),
            "classification_probability": np.asarray(probabilities),
            "positive_patch_id": np.asarray(["large__p", "small__p"]),
            "segmentation_thresholds": np.asarray([0.5, 0.6]),
            "segmentation_dice_grid": np.asarray([[0.8, 0.7], [0.6, 0.5]]),
        }

        result = apply_case_balanced_validation_metrics(metrics, predictions, rows)

        self.assertAlmostEqual(result["threshold"], 0.55)
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["aggregation"], "patches weighted to equal total weight per case")
        self.assertAlmostEqual(result["patch_pooled_classification_threshold"], 0.65)
        self.assertAlmostEqual(result["segmentation_dice"], 0.7)


if __name__ == "__main__":
    unittest.main()
