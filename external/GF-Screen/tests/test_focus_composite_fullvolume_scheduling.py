import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_focus_composite_fullvolume_oof_20260824.sh").read_text(
    encoding="utf-8"
)


class CompositeFocusFullVolumeSchedulingTests(unittest.TestCase):
    def test_requires_composite_and_primary_exact_gates(self):
        self.assertIn("QUALIFY_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME", SCRIPT)
        self.assertIn("FULL_VOLUME_OOF_COMPLETE.json", SCRIPT)
        self.assertIn("focus_fullvolume_storage_reservation_20260824.lock", SCRIPT)

    def test_uses_route_selective_tta_and_validates_both_markers(self):
        self.assertIn("apply_candidate_refiner_focus_tta_xy.py", SCRIPT)
        self.assertIn("validate_candidate_refiner_inference_marker.py", SCRIPT)
        self.assertIn("validate_focus_tta_xy_marker.py", SCRIPT)
        self.assertIn("*_tta_xy", SCRIPT)
        self.assertIn("ln \"${value}\" \"${alias}\"", SCRIPT)
        self.assertIn("VALIDATOR_MODE_ARGS", SCRIPT)
        infer_call = SCRIPT.split("apply_candidate_refiner_focus_tta_xy.py", 1)[1].split(
            "> \"${output}/fold${fold}.infer.log\"", 1
        )[0]
        self.assertNotIn("focus-routed-input-mode", infer_call)

    def test_runs_one_frozen_candidate_and_local_paired_gate(self):
        self.assertIn('"tag":"composite"', SCRIPT)
        self.assertIn("source_classification_thresholds", SCRIPT)
        self.assertIn("source_segmentation_thresholds", SCRIPT)
        self.assertIn("select_checkpoint_robust.py", SCRIPT)
        self.assertNotIn("segmentation_threshold_sweep", SCRIPT)

    def test_cannot_read_challenge_validation_or_create_submission(self):
        lowered = SCRIPT.lower()
        self.assertNotIn("codabench", lowered)
        self.assertNotIn("validation-public", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"public_read":false', lowered)
        self.assertIn('"hidden_read":false', lowered)


if __name__ == "__main__":
    unittest.main()
