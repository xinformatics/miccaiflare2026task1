import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "run_component_meta_encoder_composite_fullvolume_oof_20260825.sh"
).read_text(encoding="utf-8")


class ComponentMetaEncoderCompositeSchedulingTests(unittest.TestCase):
    def test_waits_for_both_exact_full_volume_branches(self):
        self.assertIn("component_meta_focus_fusion_fullvolume_oof_20260824", SCRIPT)
        self.assertIn("focus_encoder_upgrade_composite_fullvolume_oof_20260824", SCRIPT)
        self.assertIn('encoder_status}" != COMPLETE', SCRIPT)
        self.assertIn('component_status}" != COMPLETE', SCRIPT)

    def test_reuses_fused_component_confidence_on_encoder_contours(self):
        self.assertIn("component_probabilities.npz", SCRIPT)
        self.assertIn("patch identity mismatch", SCRIPT)
        self.assertIn("override does not cover encoder patches", SCRIPT)
        self.assertIn("materialize_candidate_refiner_component_grid.py", SCRIPT)
        self.assertIn("source_segmentation_thresholds", SCRIPT)

    def test_uses_exact_707_labels_and_prequalified_anchor_gate(self):
        self.assertIn("expected_cases=707", SCRIPT)
        self.assertIn("exact-707 OOF coverage mismatch", SCRIPT)
        self.assertIn("highest DSC among branches that passed their own exact-707 gate", SCRIPT)
        self.assertIn("select_dsc_first_nonregression.py", SCRIPT)
        self.assertIn("--min-nsd-gain 0.0", SCRIPT)

    def test_is_local_only_and_cannot_package_or_upload(self):
        lowered = SCRIPT.lower()
        self.assertNotIn("codabench", lowered)
        self.assertNotIn("validation-public", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"public_read":false', lowered)
        self.assertIn('"hidden_read":false', lowered)
        self.assertIn('"uploaded":false', lowered)


if __name__ == "__main__":
    unittest.main()
