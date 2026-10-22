import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "nnunet/RESENC_S6_POSITIVE_UNLABELED_PREFLIGHT_20260827.json"


class Dataset502PreprocessingReadinessTest(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(PREFLIGHT.read_text(encoding="utf-8"))[
            "preprocessing_readiness"
        ]

    def test_exact_reuse_and_remaining_inventory_is_frozen(self):
        result = self.record["verify_only_result"]
        self.assertEqual(result["total_cases"], 19_886)
        self.assertEqual(result["complete_reused_cases"], 17_575)
        self.assertEqual(result["remaining_pseudo_cases"], 2_311)
        self.assertEqual(result["partial_artifacts_removed"], 0)
        reuse = self.record["reused_cache"]
        self.assertEqual(reuse["symlinks"], 52_725)
        self.assertEqual(reuse["broken_symlinks"], 0)

    def test_preprocessing_inputs_are_hash_bound(self):
        for item in self.record["inputs"].values():
            observed = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, item["sha256"], item["path"])

    def test_snapshot_has_conservative_cache_headroom(self):
        disk = self.record["disk_headroom_snapshot"]
        self.assertGreater(
            disk["filesystem_available_bytes"],
            3 * disk["three_times_base_mean_projected_new_pseudo_cache_bytes"],
        )
        self.assertFalse(self.record["public_read"])
        self.assertFalse(self.record["hidden_read"])
        self.assertFalse(self.record["training_mutated"])

    def test_serialized_controller_rechecks_disk_before_preprocessing(self):
        controller_path = ROOT / self.record["inputs"][
            "serialized_preprocessing_controller"
        ]["path"]
        controller = controller_path.read_text(encoding="utf-8")
        self.assertIn(
            "minimum_disk_available_bytes=$((128 * 1024 * 1024 * 1024))",
            controller,
        )
        self.assertIn("disk_available=filesystem.f_bavail*filesystem.f_frsize", controller)
        self.assertIn("disk>=minimum", controller)
        self.assertIn("waiting for safe Dataset 502 preprocessing", controller)


if __name__ == "__main__":
    unittest.main()
