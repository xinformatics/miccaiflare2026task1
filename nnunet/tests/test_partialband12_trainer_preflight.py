import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "nnunet/RESENC_S6_PARTIALBAND12_TRAINER_PREFLIGHT_20260827.json"


class PartialBand12TrainerPreflightTest(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(PREFLIGHT.read_text(encoding="utf-8"))

    def test_all_code_dependencies_are_hash_bound(self):
        for item in self.record["code"].values():
            observed = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, item["sha256"], item["path"])
        geometry = self.record["geometry_preflight"]
        self.assertEqual(
            hashlib.sha256((ROOT / geometry["path"]).read_bytes()).hexdigest(),
            geometry["sha256"],
        )
        context = self.record["training_context_audit"]
        for key in ("report", "patch_metrics", "auditor", "tests"):
            item = context[key]
            observed = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, item["sha256"], item["path"])
        diagnostic = context["stratified_diagnostic"]
        for key in ("report", "analyzer", "tests"):
            item = diagnostic[key]
            observed = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, item["sha256"], item["path"])

    def test_recipe_and_public_gate_are_frozen(self):
        self.assertEqual(self.record["status"], "PASS_PREPARED_NOT_ARMED")
        self.assertEqual(self.record["model"]["parameter_count"], 107_467_050)
        self.assertEqual(self.record["recipe"]["initial_lr"], 1e-4)
        self.assertEqual(self.record["recipe"]["checkpoint_epochs"], [25, 50, 100])
        self.assertEqual(
            self.record["recipe"]["partial_label_policy"], "shape_linf_band_12mm"
        )
        self.assertEqual(self.record["recipe"]["margin_voxels_zyx"], [5, 8, 8])
        context = self.record["training_context_audit"]
        self.assertEqual(context["status"], "PASS_CONTEXT_ADEQUATE")
        self.assertGreaterEqual(
            context["median_background_to_foreground_ratio"],
            context["minimum_required_median_ratio"],
        )
        self.assertTrue(context["all_annotated_foreground_retained"])
        diagnostic = context["stratified_diagnostic"]
        self.assertEqual(
            diagnostic["status"], "COMPLETE_DESCRIPTIVE_NO_POLICY_CHANGE"
        )
        self.assertEqual(
            diagnostic["decision"],
            "KEEP_PREDECLARED_FIXED_12MM_SHAPE_BAND_CANDIDATE",
        )
        self.assertEqual(
            self.record["public_selection"]["minimum_fixed_public_DSC"],
            0.6366639257651584,
        )
        self.assertEqual(
            self.record["public_selection"]["minimum_fixed_public_NSD"],
            0.5347520263176905,
        )
        self.assertFalse(
            self.record["public_selection"]["threshold_search_after_public_read"]
        )

    def test_no_training_or_hidden_action_is_authorized(self):
        authorization = self.record["authorization"]
        self.assertTrue(authorization["trainer_implemented"])
        self.assertFalse(authorization["training_controller_implemented"])
        self.assertFalse(authorization["public_controller_implemented"])
        self.assertFalse(authorization["queue_armed"])
        self.assertFalse(self.record["active_16mm_training_changed"])
        self.assertFalse(self.record["public_used_for_training"])
        self.assertFalse(self.record["hidden_read"])
        self.assertFalse(self.record["submission_created"])
        self.assertFalse(self.record["uploaded"])


if __name__ == "__main__":
    unittest.main()
