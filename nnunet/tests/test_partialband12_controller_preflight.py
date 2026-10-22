import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "nnunet/RESENC_S6_PARTIALBAND12_CONTROLLER_PREFLIGHT_20260827.json"


class PartialBand12ControllerPreflightTest(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(RECORD.read_text(encoding="utf-8"))

    def test_every_controller_input_is_hash_bound(self):
        items = [
            self.record["trainer_preflight"],
            self.record["authorization"],
            *self.record["controllers"].values(),
            self.record["parent_selector"],
            self.record["tests"],
            self.record["parent_tests"],
        ]
        for item in items:
            observed = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, item["sha256"], item["path"])

    def test_parent_and_stage_contract(self):
        self.assertEqual(self.record["controllers"]["training"]["stages"], [25, 50, 100])
        self.assertEqual(
            self.record["parent_selection"]["allowed_epochs"],
            [100, 200, 300, 400, 500],
        )
        self.assertTrue(self.record["parent_selection"]["requires_checkpoint_sha256_match"])
        self.assertTrue(self.record["controllers"]["public"]["sequential_single_case_inference"])

    def test_manifest_proves_branch_cannot_launch_now(self):
        self.assertEqual(self.record["status"], "PASS_PREPARED_NOT_ARMED")
        safety = self.record["safety"]
        self.assertFalse(safety["current_authorization_satisfies_launch"])
        self.assertFalse(safety["current_trainer_preflight_satisfies_launch"])
        self.assertTrue(safety["runtime_consumes_controller_preflight"])
        self.assertTrue(safety["runtime_verifies_controller_and_selector_hashes"])
        self.assertFalse(safety["creates_systemd_service"])
        self.assertFalse(safety["contains_hidden_path"])
        self.assertFalse(safety["contains_archive_creation"])
        self.assertFalse(safety["contains_upload_interface"])
        self.assertFalse(safety["active_16mm_run_changed"])
        self.assertFalse(safety["existing_queue_changed"])


if __name__ == "__main__":
    unittest.main()
