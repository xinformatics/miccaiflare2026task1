import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from nnunet.evaluate_resenc_s6_public import (  # noqa: E402
    DEPLOYMENTS,
    cache_valid,
    evaluate,
    validate_public_labels,
)


class EvaluateResEncS6PublicTests(unittest.TestCase):
    def test_deployments_match_frozen_policy_exactly(self):
        self.assertEqual(
            DEPLOYMENTS,
            (
                "resenc_standalone_p50",
                "gf_anchor_confirm_either_p70",
                "gf_resenc_boundary3mm_p70p50_ftu",
                "gf_union_resenc_p40_ftu_s10_mm3000",
            ),
        )

    def test_only_exact_public_label_role_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / "validation" / "Validation-Public-Labels"
            hidden = root / "validation" / "Validation-Hidden-Labels"
            public.mkdir(parents=True)
            hidden.mkdir(parents=True)
            validate_public_labels(public)
            with self.assertRaises(ValueError):
                validate_public_labels(hidden)

    def test_cache_requires_exact_source_signature_and_no_hidden_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case.json"
            signature = {"policy_sha256": "abc"}
            payload = {
                "status": "COMPLETE",
                "source_signature": signature,
                "metrics": {name: [0.5, 0.4] for name in DEPLOYMENTS},
                "public_labels_read": True,
                "hidden_read": False,
                "submission_created": False,
                "uploaded": False,
            }
            import json
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(cache_valid(path, signature))
            self.assertIsNone(cache_valid(path, {"policy_sha256": "different"}))
            payload["hidden_read"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(cache_valid(path, signature))

    def test_source_has_no_threshold_grid_or_hidden_label_interface(self):
        source = (PROJECT_ROOT / "nnunet" / "evaluate_resenc_s6_public.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("PROBABILITY_THRESHOLDS", source)
        self.assertNotIn('add_argument("--hidden-labels"', source)
        self.assertNotIn("requests.post", source)

    def test_exact_fifty_case_evaluator_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "validation" / "Validation-Public-Labels"
            anchor = root / "anchor"
            resenc = root / "resenc"
            freetumor = root / "freetumor"
            cache = root / "cache"
            for directory in (labels, anchor, resenc, freetumor):
                directory.mkdir(parents=True)
            array = np.zeros((3, 3, 3), dtype=np.uint8)
            array[1, 1, 1] = 1
            affine = np.eye(4)
            for index in range(50):
                name = f"case_{index:04d}.nii.gz"
                for directory in (labels, anchor, resenc, freetumor):
                    nib.save(nib.Nifti1Image(array, affine), directory / name)
                np.savez_compressed(
                    resenc / f"case_{index:04d}.prob16.npz",
                    lesion_probability_uint16=array.astype(np.uint16) * 65535,
                )
            policy = (
                PROJECT_ROOT
                / "nnunet"
                / "RESENC_S6_PUBLIC_SELECTION_POLICY_20260827.json"
            )
            payload = evaluate(
                policy, labels, anchor, resenc, freetumor, cache, workers=1
            )
            self.assertEqual(payload["case_count"], 50)
            self.assertEqual(payload["evaluated_deployments"], list(DEPLOYMENTS))
            for values in payload["metrics"].values():
                self.assertEqual(values["Lesion_DSC"], 1.0)
                self.assertEqual(values["Lesion_NSD"], 1.0)
            # Second pass must accept and reuse all exact-signature caches.
            repeated = evaluate(
                policy, labels, anchor, resenc, freetumor, cache, workers=1
            )
            self.assertEqual(repeated["metrics"], payload["metrics"])


if __name__ == "__main__":
    unittest.main()
