import os
import random
import unittest

import numpy as np
import torch

from train_candidate_refiner import (
    apply_source_group_segmentation_metrics,
    configure_reproducibility,
    epoch_sample_count,
    filter_rows_by_source_groups,
    learning_rate,
    seed_worker,
    should_validate_epoch,
    validation_batch_size,
    validate_resume_configuration,
)


class CandidateRefinerTrainingTests(unittest.TestCase):
    def test_source_group_filter_collapses_lung_cells_and_keeps_both_labels(self):
        rows = [
            {"source_name": "lung_cell0", "label": "0"},
            {"source_name": "lung_cell7", "label": "1"},
            {"source_name": "currentgf", "label": "1"},
        ]
        selected = filter_rows_by_source_groups(rows, ["lung_cells"])
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["source_name"] for row in selected}, {"lung_cell0", "lung_cell7"})
        with self.assertRaisesRegex(ValueError, "absent from rows"):
            filter_rows_by_source_groups(rows, ["autopet_weak"])

    def test_routed_segmentation_selector_uses_only_declared_sources(self):
        rows = [
            {
                "patch_id": "mature-pos",
                "source_name": "currentgf",
                "label": "1",
                "case": "mature-case",
                "cohort": "cohort-a",
            },
            {
                "patch_id": "weak-pos",
                "source_name": "autopet_weak",
                "label": "1",
                "case": "weak-case",
                "cohort": "cohort-b",
            },
            {
                "patch_id": "weak-neg",
                "source_name": "autopet_weak",
                "label": "0",
                "case": "weak-case",
                "cohort": "cohort-b",
            },
        ]
        predictions = {
            "positive_patch_id": np.asarray(["mature-pos", "weak-pos"]),
            "segmentation_thresholds": np.asarray([0.2, 0.5]),
            "segmentation_dice_grid": np.asarray([[0.95, 0.90], [0.30, 0.80]]),
        }
        result = apply_source_group_segmentation_metrics(
            {"segmentation_dice": 0.9},
            predictions,
            rows,
            ["autopet_weak"],
        )
        self.assertEqual(result["routed_segmentation_source_groups"], ["autopet_weak"])
        self.assertAlmostEqual(result["routed_segmentation_threshold"], 0.5)
        self.assertAlmostEqual(result["routed_segmentation_dice"], 0.8)
        self.assertEqual(result["routed_segmentation_positive_cases"], 1)

    def test_learning_rate_is_five_epoch_warmup_then_single_cosine_decay(self):
        steps_per_epoch = 100
        total_steps = 30 * steps_per_epoch
        warmup_steps = 5 * steps_per_epoch
        peak = 3e-4
        self.assertAlmostEqual(learning_rate(0, total_steps, warmup_steps, peak), peak / warmup_steps)
        self.assertAlmostEqual(
            learning_rate(warmup_steps - 1, total_steps, warmup_steps, peak), peak
        )
        self.assertAlmostEqual(
            learning_rate(17 * steps_per_epoch - 1, total_steps, warmup_steps, peak),
            1.5960669404956176e-4,
        )
        self.assertLess(
            learning_rate(total_steps - 1, total_steps, warmup_steps, peak), 1e-9
        )

    def test_reproducibility_configuration_resets_all_cpu_rngs(self):
        configure_reproducibility(1234)
        first = (random.random(), float(np.random.rand()), float(torch.rand(())))
        configure_reproducibility(1234)
        second = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(first, second)
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_worker_seed_resets_python_and_numpy_rngs(self):
        torch.manual_seed(5678)
        seed_worker(0)
        first = (random.random(), float(np.random.rand()))
        torch.manual_seed(5678)
        seed_worker(7)
        second = (random.random(), float(np.random.rand()))
        self.assertEqual(first, second)

    def test_bounded_epoch_preserves_zero_default(self):
        self.assertEqual(epoch_sample_count(10_315, 8, 0), 10_315)
        self.assertEqual(epoch_sample_count(70_000, 8, 8_192), 8_192)
        self.assertEqual(epoch_sample_count(20, 8, 0), 800)
        with self.assertRaises(ValueError):
            epoch_sample_count(70_000, 8, -1)

    def test_sparse_validation_schedule_keeps_fixed_endpoints(self):
        observed = [
            epoch for epoch in range(1, 31) if should_validate_epoch(epoch, 30, 3)
        ]
        self.assertEqual(observed, [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 30])
        self.assertTrue(all(should_validate_epoch(epoch, 4, 1) for epoch in range(1, 5)))
        with self.assertRaises(ValueError):
            should_validate_epoch(1, 30, 0)

    def test_validation_batch_size_preserves_zero_default(self):
        self.assertEqual(validation_batch_size(8, 0), 8)
        self.assertEqual(validation_batch_size(8, 32), 32)
        with self.assertRaises(ValueError):
            validation_batch_size(8, -1)

    def test_resume_recipe_validation_accepts_legacy_defaults_only(self):
        import argparse

        current = argparse.Namespace(
            folds=5,
            epochs=30,
            batch_size=8,
            workers=2,
            lr=3e-4,
            weight_decay=1e-4,
            base_channels=32,
            sampling_temperature=0.0,
            balance_source_groups=False,
            samples_per_epoch=0,
            seed=20260822,
            small_lesion_loss=False,
            small_lesion_positive_weight_cap=100.0,
            segmentation_threshold_max=0.80,
        )
        legacy = vars(current).copy()
        legacy.pop("balance_source_groups")
        legacy.pop("samples_per_epoch")
        validate_resume_configuration(legacy, current)
        changed = dict(legacy, lr=1e-4)
        with self.assertRaisesRegex(ValueError, "changed training option: lr"):
            validate_resume_configuration(changed, current)

    def test_resume_recipe_validation_binds_bounded_epoch(self):
        import argparse

        current = argparse.Namespace(
            folds=5,
            epochs=30,
            batch_size=8,
            workers=2,
            lr=3e-4,
            weight_decay=1e-4,
            base_channels=32,
            sampling_temperature=0.5,
            balance_source_groups=False,
            samples_per_epoch=8192,
            seed=20260823,
            small_lesion_loss=False,
            small_lesion_positive_weight_cap=100.0,
            segmentation_threshold_max=0.80,
        )
        saved = vars(current).copy()
        validate_resume_configuration(saved, current)
        saved["samples_per_epoch"] = 4096
        with self.assertRaisesRegex(
            ValueError, "changed training option: samples_per_epoch"
        ):
            validate_resume_configuration(saved, current)

    def test_resume_recipe_validation_binds_small_lesion_loss(self):
        import argparse

        current = argparse.Namespace(
            folds=5,
            epochs=30,
            batch_size=8,
            workers=2,
            lr=3e-4,
            weight_decay=1e-4,
            base_channels=32,
            sampling_temperature=0.5,
            balance_source_groups=False,
            samples_per_epoch=8192,
            source_conditioning=False,
            source_condition_groups=[],
            small_lesion_loss=True,
            small_lesion_positive_weight_cap=100.0,
            segmentation_threshold_max=0.95,
            seed=20260823,
        )
        saved = vars(current).copy()
        validate_resume_configuration(saved, current)
        saved["small_lesion_loss"] = False
        with self.assertRaisesRegex(
            ValueError, "changed training option: small_lesion_loss"
        ):
            validate_resume_configuration(saved, current)

    def test_resume_recipe_validation_binds_contour_grid_ceiling(self):
        import argparse

        current = argparse.Namespace(
            folds=5,
            epochs=30,
            batch_size=8,
            workers=2,
            lr=3e-4,
            weight_decay=1e-4,
            base_channels=32,
            sampling_temperature=0.5,
            balance_source_groups=False,
            samples_per_epoch=8192,
            source_conditioning=False,
            source_condition_groups=[],
            small_lesion_loss=False,
            small_lesion_positive_weight_cap=100.0,
            segmentation_threshold_max=0.95,
            seed=20260823,
        )
        saved = vars(current).copy()
        validate_resume_configuration(saved, current)
        saved["segmentation_threshold_max"] = 0.80
        with self.assertRaisesRegex(
            ValueError, "changed training option: segmentation_threshold_max"
        ):
            validate_resume_configuration(saved, current)

    def test_resume_recipe_validation_binds_validation_interval(self):
        import argparse

        current = argparse.Namespace(
            folds=5,
            epochs=30,
            batch_size=8,
            workers=2,
            lr=3e-4,
            weight_decay=1e-4,
            base_channels=32,
            sampling_temperature=0.5,
            balance_source_groups=False,
            samples_per_epoch=8192,
            source_conditioning=False,
            source_condition_groups=[],
            small_lesion_loss=False,
            small_lesion_positive_weight_cap=100.0,
            segmentation_threshold_max=0.95,
            validation_interval=3,
            validation_batch_size=32,
            seed=20260823,
        )
        saved = vars(current).copy()
        validate_resume_configuration(saved, current)
        saved["validation_interval"] = 1
        with self.assertRaisesRegex(
            ValueError, "changed training option: validation_interval"
        ):
            validate_resume_configuration(saved, current)

    def test_resume_recipe_validation_binds_validation_batch_size(self):
        import argparse

        current = argparse.Namespace(
            folds=5,
            epochs=30,
            batch_size=8,
            workers=2,
            lr=3e-4,
            weight_decay=1e-4,
            base_channels=32,
            sampling_temperature=0.5,
            balance_source_groups=False,
            samples_per_epoch=8192,
            source_conditioning=False,
            source_condition_groups=[],
            small_lesion_loss=False,
            small_lesion_positive_weight_cap=100.0,
            segmentation_threshold_max=0.95,
            validation_interval=3,
            validation_batch_size=32,
            seed=20260823,
        )
        saved = vars(current).copy()
        validate_resume_configuration(saved, current)
        saved["validation_batch_size"] = 16
        with self.assertRaisesRegex(
            ValueError, "changed training option: validation_batch_size"
        ):
            validate_resume_configuration(saved, current)

    def test_resume_recipe_validation_binds_uncertain_negative_weight(self):
        import argparse

        current = argparse.Namespace(
            folds=5,
            epochs=30,
            batch_size=8,
            workers=2,
            lr=3e-4,
            weight_decay=1e-4,
            base_channels=32,
            sampling_temperature=0.5,
            balance_source_groups=False,
            samples_per_epoch=8192,
            source_conditioning=False,
            source_condition_groups=[],
            small_lesion_loss=False,
            small_lesion_positive_weight_cap=100.0,
            classification_negative_weight=0.25,
            segmentation_threshold_max=0.95,
            validation_interval=3,
            validation_batch_size=32,
            seed=20260823,
        )
        saved = vars(current).copy()
        validate_resume_configuration(saved, current)
        saved["classification_negative_weight"] = 1.0
        with self.assertRaisesRegex(
            ValueError, "changed training option: classification_negative_weight"
        ):
            validate_resume_configuration(saved, current)

    def test_resume_recipe_validation_binds_prompt_masked_classifier(self):
        import argparse

        current = argparse.Namespace(
            folds=5,
            epochs=30,
            batch_size=8,
            workers=2,
            lr=3e-4,
            weight_decay=1e-4,
            base_channels=32,
            sampling_temperature=0.5,
            balance_source_groups=False,
            samples_per_epoch=8192,
            source_conditioning=False,
            source_condition_groups=[],
            prompt_masked_classifier=True,
            small_lesion_loss=False,
            small_lesion_positive_weight_cap=100.0,
            classification_negative_weight=1.0,
            negative_segmentation_weight=0.1,
            segmentation_threshold_max=0.95,
            validation_interval=3,
            validation_batch_size=32,
            seed=20260823,
        )
        saved = vars(current).copy()
        validate_resume_configuration(saved, current)
        saved["prompt_masked_classifier"] = False
        with self.assertRaisesRegex(
            ValueError, "changed training option: prompt_masked_classifier"
        ):
            validate_resume_configuration(saved, current)

    def test_resume_recipe_validation_binds_fixed_prompt_channel(self):
        import argparse

        current = argparse.Namespace(
            folds=5,
            epochs=30,
            batch_size=8,
            workers=2,
            lr=3e-4,
            weight_decay=1e-4,
            base_channels=32,
            sampling_temperature=0.5,
            balance_source_groups=False,
            samples_per_epoch=8192,
            source_conditioning=False,
            source_condition_groups=[],
            prompt_masked_classifier=True,
            prompt_channel=2,
            small_lesion_loss=False,
            small_lesion_positive_weight_cap=100.0,
            classification_negative_weight=1.0,
            negative_segmentation_weight=0.1,
            segmentation_threshold_max=0.95,
            validation_interval=3,
            validation_batch_size=32,
            seed=20260823,
        )
        saved = vars(current).copy()
        validate_resume_configuration(saved, current)
        saved["prompt_channel"] = 1
        with self.assertRaisesRegex(
            ValueError, "changed training option: prompt_channel"
        ):
            validate_resume_configuration(saved, current)


if __name__ == "__main__":
    unittest.main()
