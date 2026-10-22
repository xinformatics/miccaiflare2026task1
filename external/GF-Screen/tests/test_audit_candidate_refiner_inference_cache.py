import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_candidate_refiner_inference_cache import (
    audit_patch,
    audit_prediction,
    validate_inference_plan,
)


class CandidateRefinerInferenceCacheAuditTests(unittest.TestCase):
    def test_valid_prediction_is_bounded_and_has_exact_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.npz"
            segmentation = np.full((1, 16, 16, 16), 0.25, dtype=np.float16)
            np.savez_compressed(
                path,
                classification_probability=np.asarray(0.75, dtype=np.float32),
                segmentation_probability=segmentation,
            )
            result = audit_prediction((str(path), (16, 16, 16)))
            self.assertEqual(result, {"minimum": 0.25, "maximum": 0.75})

    def test_prediction_cache_mode_is_bound_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.npz"
            np.savez(
                path,
                classification_probability=np.asarray(0.75, dtype=np.float32),
                segmentation_probability=np.full(
                    (1, 4, 4, 4), 0.25, dtype=np.float16
                ),
            )
            audit_prediction((str(path), (4, 4, 4), False))
            with self.assertRaisesRegex(ValueError, "cache-mode mismatch"):
                audit_prediction((str(path), (4, 4, 4), True))

    def test_inference_plan_hash_and_cache_mode_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            plan = work / "INFERENCE_PLAN.json"
            fingerprint = "a" * 64
            plan.write_text(json.dumps({
                "schema_version": 1,
                "inference_fingerprint": fingerprint,
                "identity": {
                    "prediction_cache_compressed": False,
                    "prediction_cache_fsync": False,
                },
            }))
            digest = hashlib.sha256(plan.read_bytes()).hexdigest()
            marker = {
                "inference_plan": str(plan.resolve()),
                "inference_plan_sha256": digest,
                "inference_fingerprint": fingerprint,
                "prediction_cache_compressed": False,
                "prediction_cache_fsync": False,
            }
            self.assertTrue(validate_inference_plan(work, marker, True))
            marker["prediction_cache_compressed"] = True
            with self.assertRaisesRegex(ValueError, "disagree"):
                validate_inference_plan(work, marker, True)

    def test_prediction_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.npz"
            np.savez_compressed(
                path,
                classification_probability=np.asarray(1.25, dtype=np.float32),
                segmentation_probability=np.zeros((16, 16, 16), dtype=np.float16),
            )
            with self.assertRaisesRegex(ValueError, "prediction shape mismatch"):
                audit_prediction((str(path), (16, 16, 16)))

    def test_prediction_probability_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.npz"
            np.savez_compressed(
                path,
                classification_probability=np.asarray(1.25, dtype=np.float32),
                segmentation_probability=np.zeros(
                    (1, 16, 16, 16), dtype=np.float16
                ),
            )
            with self.assertRaisesRegex(ValueError, "probability range"):
                audit_prediction((str(path), (16, 16, 16)))

    def test_valid_patch_has_nonempty_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "patch.npz"
            image = np.zeros((3, 16, 16, 16), dtype=np.float16)
            image[2, 7:9, 7:9, 7:9] = 1
            np.savez_compressed(path, image=image, patch_affine=np.eye(4))
            result = audit_patch((str(path), (16, 16, 16)))
            self.assertEqual(result["seed_voxels"], 8)

    def test_empty_prompt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "patch.npz"
            image = np.zeros((3, 16, 16, 16), dtype=np.float16)
            np.savez_compressed(path, image=image, patch_affine=np.eye(4))
            with self.assertRaisesRegex(ValueError, "empty candidate prompt"):
                audit_patch((str(path), (16, 16, 16)))


if __name__ == "__main__":
    unittest.main()
