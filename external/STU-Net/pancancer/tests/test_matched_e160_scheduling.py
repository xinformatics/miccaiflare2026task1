import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "run_matched_e160_public_20260826.sh"
).read_text(encoding="utf-8")


class MatchedEpoch160SchedulingTests(unittest.TestCase):
    def test_uses_immutable_matched_epoch160_snapshots(self):
        self.assertIn("checkpoint_inference_epoch_0160.pth", SCRIPT)
        self.assertIn("STUNetTrainerPanCancerComponentBox__", SCRIPT)
        self.assertIn('payload.get("current_epoch") != 160', SCRIPT)

    def test_serializes_full_volume_branches(self):
        base = '"${base_predictions}" BASE 0'
        component = '"${component_predictions}" COMPONENT 1'
        self.assertIn(base, SCRIPT)
        self.assertIn(component, SCRIPT)
        self.assertLess(SCRIPT.index(base), SCRIPT.index(component))
        self.assertNotIn('"${base_predictions}" BASE 0 &', SCRIPT)
        self.assertNotIn('"${component_predictions}" COMPONENT 1 &', SCRIPT)
        self.assertIn("--save-probabilities --sequential", SCRIPT)

    def test_resumes_training_on_normal_and_error_exit(self):
        self.assertIn("trap finalize EXIT", SCRIPT)
        self.assertIn('if [[ "${training_paused}" == true ]]; then resume_branches || true; fi', SCRIPT)
        self.assertIn("resume_branches\ntraining_paused=false", SCRIPT)

    def test_has_no_hidden_inventory_or_submission_path(self):
        self.assertNotIn("Validation-Hidden", SCRIPT)
        self.assertNotIn("zip -", SCRIPT)
        self.assertNotIn("codabench", SCRIPT.lower())

    def test_complete_checkpoint_bound_sets_skip_full_volume_reopen(self):
        self.assertIn("prediction_set_complete()", SCRIPT)
        self.assertIn(
            'payload.get("checkpoint", {}).get("sha256") != checkpoint_sha',
            SCRIPT,
        )
        self.assertIn('payload.get("dataset_role") != "public"', SCRIPT)
        self.assertIn('cases != mask_names or len(cases) != 50', SCRIPT)


if __name__ == "__main__":
    unittest.main()
