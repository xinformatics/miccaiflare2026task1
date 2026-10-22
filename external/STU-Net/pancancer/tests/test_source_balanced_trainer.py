import importlib.util
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


PANCANCER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANCANCER))
MODULE_PATH = PANCANCER / "STUNetTrainerPanCancerSourceBalanced.py"
SPEC = importlib.util.spec_from_file_location("source_balanced_trainer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeDataset:
    def __init__(self, identifiers, foreground_counts=None):
        self.identifiers = identifiers
        self.foreground_counts = foreground_counts


class FakeLoader:
    def __init__(self):
        self._data = FakeDataset(["source_0"])


class SourceBalancedTrainerTests(unittest.TestCase):
    def test_probability_mixture_is_normalized_and_preserves_bins(self):
        ids = ["large_0", "large_1", "large_2", "small_0"]
        counts = [10_000, 10_000, 10_000, 50]
        probabilities = MODULE.stratified_source_probabilities(ids, counts, 0.25, 3.0)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(probabilities > 0))
        self.assertAlmostEqual(float(probabilities[-1]), 0.25)
        self.assertAlmostEqual(float(probabilities[:3].sum()), 0.75)

    def test_same_bin_source_balance_is_bounded(self):
        ids = ["large_0", "large_1", "large_2", "small_0"]
        counts = [500, 500, 500, 500]
        probabilities = MODULE.stratified_source_probabilities(ids, counts, 0.25, 3.0)
        self.assertGreater(probabilities[-1], probabilities[0])
        self.assertLessEqual(float(probabilities.max() * len(ids)), 3.0)

    def test_loader_factory_changes_training_but_not_validation(self):
        calls = []

        def original_loader(*args, **kwargs):
            calls.append((args, kwargs))
            return kwargs.get("sampling_probabilities")

        resolver = lambda data: np.asarray(data.foreground_counts)
        factory = MODULE.source_balanced_loader_factory(
            original_loader, 0.25, 3.0, resolver
        )
        training = factory(
            FakeDataset(
                ["large_0", "large_1", "large_2", "small_0"],
                [500, 500, 500, 500],
            ),
            sampling_probabilities=None,
        )
        validation = factory(
            FakeDataset(["large_3", "small_1"], [500, 500]),
            sampling_probabilities=None,
        )
        self.assertAlmostEqual(float(training.sum()), 1.0)
        self.assertIsNone(validation)
        self.assertEqual(len(calls), 2)

    def test_existing_training_probabilities_fail_closed(self):
        factory = MODULE.source_balanced_loader_factory(
            lambda *a, **k: None,
            0.25,
            3.0,
            lambda data: np.asarray([500]),
        )
        with self.assertRaises(ValueError):
            factory(FakeDataset(["source_0"]), sampling_probabilities=[1.0])

    def test_augmenter_unwrap_supports_both_backends(self):
        single = type("Single", (), {"data_loader": FakeLoader()})()
        multi = type("Multi", (), {"generator": FakeLoader()})()
        self.assertIs(MODULE.unwrap_augmenter_loader(single), single.data_loader)
        self.assertIs(MODULE.unwrap_augmenter_loader(multi), multi.generator)
        with self.assertRaises(ValueError):
            MODULE.unwrap_augmenter_loader(object())

    def test_contract_and_recipe_inherit_balanced_branch(self):
        cls = MODULE.STUNetTrainerPanCancerComponentBoxE160SourceStratified
        self.assertEqual(cls.source_fraction, 0.25)
        self.assertEqual(cls.raw_ratio_cap, 3.0)
        self.assertEqual(cls.component_center_fraction, 0.25)
        self.assertEqual(cls.expected_case_count, 17_575)
        self.assertEqual(cls.expected_source_count, 28)
        self.assertEqual(
            cls.expected_size_bin_counts, (3_946, 2_113, 1_577, 3_515, 3_242, 3_182)
        )
        self.assertEqual(
            MODULE.identifier_inventory_sha256(["a_0", "b_1"]),
            "ca0b7a463d7485e47d81e92abba2e55eea5891db66b4defd963c465b11973353",
        )
        self.assertEqual(
            cls.parent_checkpoint_sha256,
            "81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74",
        )

    def test_source_stratified_branch_uses_mildly_aggressive_recall_loss(self):
        cls = MODULE.STUNetTrainerPanCancerComponentBoxE160SourceStratified
        trainer = object.__new__(cls)
        trainer.enable_deep_supervision = False
        loss = trainer._build_loss()
        self.assertEqual(loss.alpha, 0.25)
        self.assertEqual(loss.beta, 0.75)

    def test_component_uniform_center_selects_a_real_lesion_voxel(self):
        segmentation = np.zeros((1, 5, 5, 5), dtype=np.int16)
        segmentation[0, 0, 0, 0] = 1
        segmentation[0, 3:5, 3:5, 3:5] = 1
        random_state = np.random.RandomState(7)
        selected = MODULE.component_uniform_voxel(
            segmentation, 1.0, random_state
        )
        self.assertEqual(selected.shape, (4,))
        self.assertEqual(selected[0], 0)
        self.assertEqual(segmentation[tuple(selected)], 1)

    def test_component_uniform_center_can_delegate_to_voxel_uniform(self):
        segmentation = np.ones((1, 2, 2, 2), dtype=np.int16)
        self.assertIsNone(
            MODULE.component_uniform_voxel(
                segmentation, 0.0, np.random.RandomState(7)
            )
        )

    def test_recall_loss_decision_is_bound_to_trainer_source(self):
        audit = json.loads(
            (
                PANCANCER
                / "analysis/source_stratified_recall_loss_decision_20260826.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            audit["decision"],
            "USE_MILDLY_AGGRESSIVE_RECALL_LOSS_FOR_SOURCE_STRATIFIED_BRANCH",
        )
        self.assertEqual(audit["loss"]["alpha_false_positive"], 0.25)
        self.assertEqual(audit["loss"]["beta_false_negative"], 0.75)
        self.assertEqual(
            audit["trainer_source_sha256"],
            hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
