import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PANCANCER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANCANCER))
MODULE_PATH = PANCANCER / "STUNetTrainerPanCancerCompleteHybrid.py"
SPEC = importlib.util.spec_from_file_location("complete_hybrid_trainer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeDataset:
    def __init__(self, identifiers):
        self.identifiers = identifiers


class CompleteHybridTrainerTests(unittest.TestCase):
    def test_complete_identifier_contract_is_explicit(self):
        self.assertTrue(MODULE.is_complete_identifier("Chest_LNDb_0001"))
        self.assertTrue(MODULE.is_complete_identifier("WholeBody_autoPETCT_0001"))
        self.assertFalse(MODULE.is_complete_identifier("Chest_LUNA25_0001"))
        self.assertFalse(
            MODULE.is_complete_identifier("LymphNodes-Mediastinal-SEG_0001")
        )

    def test_priority_probabilities_are_bounded_and_normalized(self):
        ids = ["Chest_LNDb_0", "Chest_LUNA25_0", "Chest_LUNA25_1"]
        probabilities = MODULE.complete_priority_probabilities(ids, 4.0)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(probabilities > 0))
        self.assertAlmostEqual(float(probabilities[0] / probabilities[1]), 4.0)
        with self.assertRaises(ValueError):
            MODULE.complete_priority_probabilities(ids, 0.5)

    def test_full_supervision_only_for_verified_complete_sample(self):
        target = torch.zeros((2, 1, 5, 5, 5), dtype=torch.int16)
        target[:, :, 2, 2, 2] = 1
        target[:, :, 0, 0, 0] = -1
        result = MODULE.complete_hybrid_target(
            target,
            ["Chest_LNDb_0", "Chest_LUNA25_0"],
            (5, 5, 5),
            (1, 1, 1),
        )
        self.assertEqual(int(result[0, 0, 4, 4, 4]), 0)
        self.assertEqual(int(result[0, 0, 0, 0, 0]), MODULE.IGNORE_LABEL)
        self.assertEqual(int(result[1, 0, 4, 4, 4]), MODULE.IGNORE_LABEL)
        self.assertEqual(int(result[1, 0, 2, 2, 2]), 1)

    def test_deep_supervision_preserves_case_policy(self):
        keys = ["Chest_LNDb_0", "Chest_LUNA25_0"]
        for shape in ((8, 8, 8), (4, 4, 4)):
            target = torch.zeros((2, 1, *shape), dtype=torch.int16)
            center = tuple(value // 2 for value in shape)
            target[(slice(None), 0, *center)] = 1
            result = MODULE.complete_hybrid_target(
                target, keys, (8, 8, 8), (2, 2, 2)
            )
            self.assertEqual(int(result[0, 0, -1, -1, -1]), 0)
            self.assertEqual(int(result[1, 0, 0, 0, 0]), MODULE.IGNORE_LABEL)

    def test_loader_factory_changes_training_only(self):
        calls = []

        def original(*args, **kwargs):
            calls.append(kwargs)
            return kwargs.get("sampling_probabilities")

        factory = MODULE.complete_priority_loader_factory(original, 4.0)
        training = factory(
            FakeDataset(["Chest_LNDb_0", "Chest_LUNA25_0"]),
            sampling_probabilities=None,
        )
        validation = factory(
            FakeDataset(["Chest_LNDb_1", "Chest_LUNA25_1"]),
            sampling_probabilities=None,
        )
        self.assertAlmostEqual(float(training.sum()), 1.0)
        self.assertIsNone(validation)
        self.assertEqual(len(calls), 2)

    def test_recipe_inherits_exact_parent_and_expected_inventory(self):
        cls = MODULE.STUNetTrainerPanCancerComponentBoxE160CompleteHybrid
        self.assertEqual(cls.expected_case_count, 17_575)
        self.assertEqual(cls.expected_complete_case_count, 3_145)
        self.assertEqual(cls.expected_partial_case_count, 14_430)
        self.assertEqual(cls.complete_case_multiplier, 4.0)
        self.assertEqual(
            cls.parent_checkpoint_sha256,
            "81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74",
        )


if __name__ == "__main__":
    unittest.main()
