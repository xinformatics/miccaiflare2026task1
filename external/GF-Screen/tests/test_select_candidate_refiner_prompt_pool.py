import unittest

from select_candidate_refiner_prompt_pool import decide


def summary(segmentation: float, composite: float, classification: float):
    return {
        "segmentation": {"segmentation_dice": segmentation},
        "composite": {"selection_score": composite},
        "classification": {"average_precision": classification},
    }


def audit(mature: float, weak: float, lung: float, *, fold: int = 0):
    return {
        "fold": fold,
        "folds": 5,
        "seed": 20260823,
        "index_sha256": "same-index",
        "source_classification_groups": {
            "mature": {"average_precision": mature},
            "weak_sources": {"average_precision": weak},
            "lung_cells": {"average_precision": lung},
        },
    }


class PromptPoolSelectorTests(unittest.TestCase):
    def test_requires_all_source_and_preservation_gates(self):
        baseline = summary(0.60, 0.70, 0.80)
        baseline_audit = audit(0.90, 0.50, 0.60)
        candidate = summary(0.596, 0.701, 0.806)
        candidate_audit = audit(0.896, 0.511, 0.606)
        decision, gains, gates = decide(
            baseline, candidate, baseline_audit, candidate_audit
        )
        self.assertEqual(decision, "EXPAND_PROMPT_MASKED_CLASSIFIER")
        self.assertTrue(all(gates.values()))
        self.assertAlmostEqual(gains["weak_sources_average_precision"], 0.011)

        failing = (
            (summary(0.594, 0.701, 0.806), candidate_audit),
            (summary(0.596, 0.699, 0.806), candidate_audit),
            (summary(0.596, 0.701, 0.804), candidate_audit),
            (candidate, audit(0.894, 0.511, 0.606)),
            (candidate, audit(0.896, 0.509, 0.606)),
            (candidate, audit(0.896, 0.511, 0.604)),
        )
        for candidate_summary, source_audit in failing:
            decision, _, gates = decide(
                baseline, candidate_summary, baseline_audit, source_audit
            )
            self.assertEqual(decision, "KEEP_GLOBAL_CLASSIFIER")
            self.assertFalse(all(gates.values()))

    def test_rejects_mixed_fold_identity(self):
        with self.assertRaisesRegex(ValueError, "fold/index identity"):
            decide(
                summary(0.6, 0.7, 0.8),
                summary(0.6, 0.7, 0.81),
                audit(0.9, 0.5, 0.6),
                audit(0.9, 0.52, 0.61, fold=1),
            )

    def test_rejects_missing_group(self):
        candidate_audit = audit(0.9, 0.52, 0.61)
        del candidate_audit["source_classification_groups"]["lung_cells"]
        with self.assertRaisesRegex(ValueError, "misses groups"):
            decide(
                summary(0.6, 0.7, 0.8),
                summary(0.6, 0.7, 0.81),
                audit(0.9, 0.5, 0.6),
                candidate_audit,
            )


if __name__ == "__main__":
    unittest.main()
