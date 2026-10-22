import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "run_component_meta_encoder_composite_public_20260825.sh"
).read_text(encoding="utf-8")


class ComponentMetaEncoderPublicSchedulingTests(unittest.TestCase):
    def test_requires_qualified_exact707_result(self):
        self.assertIn("component_meta_encoder_composite_fullvolume_oof_20260825", SCRIPT)
        self.assertIn("QUALIFY_FOR_FIXED_PUBLIC", SCRIPT)
        self.assertIn("OOF provenance contaminated", SCRIPT)

    def test_separates_classifier_and_encoder_caches(self):
        self.assertIn("component_meta_encoder_classifier_cache_20260825", SCRIPT)
        self.assertIn("work_bbox24_encoder_routes", SCRIPT)
        self.assertIn("prepare_component_probability_override.py", SCRIPT)
        self.assertIn("apply_candidate_refiner_focus_tta_xy.py", SCRIPT)
        self.assertIn("materialize_candidate_refiner_component_grid.py", SCRIPT)

    def test_ensembles_five_folds_and_validates_tta(self):
        self.assertIn("for fold in 0 1 2 3 4", SCRIPT)
        self.assertIn("--candidate-contour-route", SCRIPT)
        self.assertIn("--focus-contour-route", SCRIPT)
        self.assertIn("validate_focus_tta_xy_marker.py", SCRIPT)

    def test_uses_touchcomp_as_fixed_public_anchor(self):
        self.assertIn("touching_component.metrics.csv", SCRIPT)
        self.assertIn("--anchor-name gf_touchcomp_0825", SCRIPT)
        self.assertIn("--min-nsd-gain 0.0", SCRIPT)

    def test_cannot_read_hidden_or_create_submission(self):
        lowered = SCRIPT.lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("codabench", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"hidden_read":false', lowered)
        self.assertIn('"submission_created":false', lowered)


if __name__ == "__main__":
    unittest.main()
