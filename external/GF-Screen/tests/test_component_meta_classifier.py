from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from component_meta_classifier import (
    FEATURE_NAMES,
    FORBIDDEN_LABEL_FEATURES,
    component_features,
    load_qualified_component_classifier,
)


class FixedModel:
    def predict_proba(self, features):
        assert features.shape[1] == len(FEATURE_NAMES)
        return np.tile(np.asarray([[0.25, 0.75]]), (len(features), 1))


def records():
    return [
        {
            "source_name": "lung_cell0",
            "seed_mm3": 125.0,
            "max_source_support": 0.75,
            "mean_source_support": 0.25,
            # These deliberately disagree across tiles. They are target-derived
            # and must have no influence on deployment features.
            "candidate_positive": index % 2,
            "seed_overlap_fraction": float(index),
            "distance_to_target_mm": float(index * 10),
        }
        for index in range(3)
    ]


class ComponentMetaClassifierTests(unittest.TestCase):
    def test_features_are_label_free_and_deterministic(self):
        first = component_features(records(), [0.2, 0.8, 0.5])
        mutated = records()
        for row in mutated:
            for name in FORBIDDEN_LABEL_FEATURES:
                row[name] = 999
        second = component_features(mutated, [0.2, 0.8, 0.5])
        np.testing.assert_allclose(first, second, rtol=0, atol=0)
        self.assertEqual(first.shape, (len(FEATURE_NAMES),))
        self.assertAlmostEqual(first[0], 0.8)

    def test_features_reject_mixed_sources_and_invalid_support(self):
        mixed = records()
        mixed[1]["source_name"] = "epoch105"
        with self.assertRaisesRegex(ValueError, "one proposal source"):
            component_features(mixed, [0.2, 0.8, 0.5])
        invalid = records()
        for row in invalid:
            row["max_source_support"] = 1.5
        with self.assertRaisesRegex(ValueError, "invalid component"):
            component_features(invalid, [0.2, 0.8, 0.5])

    def test_qualified_loader_checks_hash_and_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            bundle = tmp_path / "model.joblib"
            joblib.dump(
                {
                    "feature_names": FEATURE_NAMES,
                    "source_names": (
                        "autopet_weak", "currentgf", "epoch105", "lndb_weak",
                        "lung_cell0", "lung_cell1", "lung_cell2", "lung_cell3",
                        "lung_cell4", "lung_cell5", "lung_cell6", "lung_cell7",
                    ),
                    "fold_models": {0: FixedModel()},
                    "deployment_model": FixedModel(),
                },
                bundle,
            )
            digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
            decision = tmp_path / "DECISION.json"
            decision.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "decision": "QUALIFY_COMPONENT_META_CLASSIFIER",
                        "selection_basis": "cross-fitted exact707 grouped held-out component classification only",
                        "feature_names": FEATURE_NAMES,
                        "model_bundle": str(bundle),
                        "model_bundle_sha256": digest,
                        "public_read": False,
                        "hidden_read": False,
                        "submission_created": False,
                    }
                )
            )
            model, _ = load_qualified_component_classifier(decision)
            self.assertEqual(
                model.predict_proba(np.zeros((1, len(FEATURE_NAMES))))[0, 1], 0.75
            )
            self.assertIsNotNone(load_qualified_component_classifier(decision, fold=0)[0])
            with self.assertRaisesRegex(ValueError, "lacks requested"):
                load_qualified_component_classifier(decision, fold=4)
            bundle.write_bytes(bundle.read_bytes() + b"mutation")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_qualified_component_classifier(decision)


if __name__ == "__main__":
    unittest.main()
