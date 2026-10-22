import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from focus_classifier_fusion import (
    DEPLOYMENT_FEATURE_NAMES,
    apply_fusion,
    focus_confidence_features,
    load_qualified_fusion,
)
from train_focus_candidate_refiner import focus_confidence_features as training_features


class FocusClassifierFusionTests(unittest.TestCase):
    def deployment(self, coefficient=None):
        count = len(DEPLOYMENT_FEATURE_NAMES)
        return {
            "focus_input_mode": "soft_lung_seed",
            "scaler_mean": np.zeros(count),
            "scaler_scale": np.ones(count),
            "coefficient": np.asarray(
                coefficient if coefficient is not None else np.zeros(count),
                dtype=np.float64,
            ),
            "intercept": 0.0,
        }

    def test_deployment_features_match_training_features(self):
        probabilities = torch.linspace(0, 1, 2 * 8 * 8 * 8).reshape(2, 1, 8, 8, 8)
        images = torch.zeros(2, 3, 8, 8, 8)
        images[0, 2, 1:4, 2:5, 3:6] = 1
        images[1, 2, 4:7, 1:3, 2:6] = 1
        torch.testing.assert_close(
            focus_confidence_features(probabilities, images, "soft_lung_seed"),
            training_features(probabilities, images),
        )

    def test_frozen_logistic_model_uses_b32_logit_and_source_geometry(self):
        count = len(DEPLOYMENT_FEATURE_NAMES)
        coefficient = np.zeros(count)
        coefficient[0] = 1.0
        coefficient[-3] = 0.5  # mature one-hot
        deployment = self.deployment(coefficient)
        probabilities = torch.tensor([0.2, 0.8])
        focus = torch.zeros(2, 1, 4, 4, 4)
        images = torch.zeros(2, 3, 4, 4, 4)
        records = [
            {"source_name": "currentgf", "seed_mm3": 10},
            {"source_name": "lung_cell3", "seed_mm3": 20},
        ]
        result = apply_fusion(probabilities, focus, images, records, deployment)
        expected = torch.sigmoid(
            torch.tensor([
                torch.logit(torch.tensor(0.2)) + 0.5,
                torch.logit(torch.tensor(0.8)),
            ])
        )
        torch.testing.assert_close(result, expected)

    def test_loader_fails_closed_on_unqualified_or_malformed_model(self):
        count = len(DEPLOYMENT_FEATURE_NAMES)
        payload = {
            "schema_version": 1,
            "selection_basis": "cross-fitted exact707 grouped held-out patch classification only",
            "decision": "QUALIFY_FOCUS_CLASSIFIER_FUSION",
            "checks": {
                "average_precision_gain": True,
                "f0p5_gain": True,
                "source_nonregression": True,
            },
            "model": {"type": "standardized logistic regression", "C": 0.1},
            "index_sha256": "a" * 64,
            "provenance": [
                {
                    "fold": fold,
                    "b32_prediction_sha256": "b" * 64,
                    "focus_prediction_sha256": "c" * 64,
                }
                for fold in range(5)
            ],
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
            "feature_names": list(DEPLOYMENT_FEATURE_NAMES),
            "focus_input_mode": "soft_lung_seed",
            "deployment_model": {
                "scaler_mean": [0.0] * count,
                "scaler_scale": [1.0] * count,
                "coefficient": [0.0] * count,
                "intercept": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decision.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            model = load_qualified_fusion(path)
            self.assertEqual(model["focus_input_mode"], "soft_lung_seed")
            payload["decision"] = "KEEP_B32_CLASSIFIER"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not qualified"):
                load_qualified_fusion(path)
            payload["decision"] = "QUALIFY_FOCUS_CLASSIFIER_FUSION"
            payload["deployment_model"]["scaler_scale"][0] = 0.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid coefficients"):
                load_qualified_fusion(path)


if __name__ == "__main__":
    unittest.main()
