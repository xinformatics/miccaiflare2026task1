import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PartialBand12PreflightTest(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(
            (ROOT / "nnunet/RESENC_S6_PARTIALBAND12_PRIMITIVE_PREFLIGHT_20260827.json")
            .read_text(encoding="utf-8")
        )

    def test_hash_bound_inputs_are_current(self):
        implementation = self.record["implementation"]
        self.assertEqual(sha256(ROOT / implementation["path"]), implementation["sha256"])
        for key in ("audit_tests", "target_tests"):
            item = self.record["tests"][key]
            self.assertEqual(sha256(ROOT / item["path"]), item["sha256"])
        audit = self.record["audit"]
        self.assertEqual(sha256(ROOT / audit["path"]), audit["sha256"])

    def test_manifest_metrics_match_audit_exactly(self):
        audit = json.loads((ROOT / self.record["audit"]["path"]).read_text())
        expected = self.record["audit"]["largest_primary_simulation"]
        mappings = {
            "shape_band_12mm": ("shape_linf_band", "12"),
            "component_box_8mm": ("component_box", "8"),
            "current_component_box_16mm": ("component_box", "16"),
        }
        for name, (policy, margin) in mappings.items():
            observed = audit["results"][policy][margin]["largest_component_as_primary"]
            self.assertEqual(
                expected[name]["pooled_omitted_voxel_collision_fraction"],
                observed["pooled_omitted_voxel_collision_fraction"],
            )
            self.assertEqual(
                expected[name]["median_clean_background_to_primary_ratio"],
                observed["trusted_background_to_primary_ratio"]["median"],
            )
            self.assertEqual(
                expected[name]["collided_omitted_components_ge_3000mm3"],
                observed["physical_component_bands"]["ge_3000mm3"][
                    "collided_component_scenarios"
                ],
            )

    def test_fail_closed_authorization(self):
        authorization = self.record["authorization"]
        self.assertEqual(self.record["status"], "PASS")
        self.assertEqual(self.record["candidate_policy"], "shape_linf_band_12mm")
        self.assertFalse(authorization["trainer_implemented"])
        self.assertFalse(authorization["queue_armed"])
        self.assertFalse(authorization["active_16mm_run_changed"])
        self.assertTrue(self.record["public_labels_read"])
        self.assertFalse(self.record["hidden_read"])
        self.assertFalse(self.record["submission_created"])
        self.assertFalse(self.record["uploaded"])


if __name__ == "__main__":
    unittest.main()
