import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_source_stunet_verified707_fullvolume_oof_20260827.sh").read_text()


class SourceStunetVerified707FullVolumeOOFTests(unittest.TestCase):
    def test_requires_patch_gate_and_exact_excluded_folds(self):
        self.assertIn("QUALIFY_STUNET_SEED_FULL_VOLUME_OOF", SCRIPT)
        self.assertIn("stage_candidate_refiner_oof_cases.py", SCRIPT)
        self.assertIn("--folds 5 --seed 20260827", SCRIPT)
        self.assertIn("checkpoint_best_classification.pt", SCRIPT)
        self.assertIn("validate_candidate_refiner_inference_marker.py", SCRIPT)

    def test_preserves_stu_contour_and_uses_frozen_gf_anchor(self):
        self.assertIn('--source-mode source_stunet=seed', SCRIPT)
        self.assertIn("currentgf_g050_l050", SCRIPT)
        self.assertIn('--base "${stage}/proposals/currentgf"', SCRIPT)
        self.assertNotIn("segmentation_threshold_sweep", SCRIPT)

    def test_material_gain_and_nsd_gates_are_predeclared(self):
        self.assertIn("--min-dsc-gain 0.015", SCRIPT)
        self.assertIn("--min-nsd-gain 0.0", SCRIPT)
        self.assertIn("--min-win-rate 0.52", SCRIPT)
        self.assertIn("--max-catastrophic-rate 0.10", SCRIPT)
        self.assertIn("QUALIFY_STUNET_SEED_FIXED_PUBLIC", SCRIPT)

    def test_public_hidden_and_upload_are_unreachable(self):
        self.assertIn('"public_read":False', SCRIPT)
        self.assertIn('"hidden_read":False', SCRIPT)
        self.assertIn('"submission_created":False', SCRIPT)
        self.assertIn('"uploaded":False', SCRIPT)
        for forbidden in ("Validation-Public", "Validation-Hidden", "codabench.org", "curl "):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
