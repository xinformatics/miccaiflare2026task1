import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "run_focus_encoder_upgrade_composite_fullvolume_oof_20260824.sh"
).read_text(encoding="utf-8")


class FocusEncoderUpgradeCompositeFullVolumeSchedulingTests(unittest.TestCase):
    def test_requires_encoder_route_primary_and_frozen_composite(self):
        self.assertIn("focus_encoder_upgrade_composite_routes_20260824", SCRIPT)
        self.assertIn("FULL_VOLUME_OOF_COMPLETE.json", SCRIPT)
        self.assertIn("frozen_composite_terminal", SCRIPT)
        self.assertIn("frozen_composite_metrics", SCRIPT)
        self.assertIn("focus_fullvolume_storage_reservation_20260824.lock", SCRIPT)
        self.assertIn(
            "QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME", SCRIPT
        )
        self.assertIn(
            "QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_NOTTA_FULL_VOLUME", SCRIPT
        )

    def test_supports_route_selective_tta_and_notta(self):
        self.assertIn("apply_candidate_refiner_focus_tta_xy.py", SCRIPT)
        self.assertIn('deployment="${root}/apply_candidate_refiner.py"', SCRIPT)
        self.assertIn("validate_candidate_refiner_inference_marker.py", SCRIPT)
        self.assertIn("validate_focus_tta_xy_marker.py", SCRIPT)
        self.assertIn('"${#TTA_MARKER_ARGS[@]}" -gt 0', SCRIPT)

    def test_pairs_against_frozen_composite_with_material_gain_gate(self):
        self.assertIn('--anchor "${frozen_composite_metrics}"', SCRIPT)
        self.assertIn("paired_vs_frozen_composite.robust.json", SCRIPT)
        self.assertIn("--min-dsc-gain 0.002", SCRIPT)
        self.assertIn("--min-nsd-gain 0.0", SCRIPT)
        self.assertIn("--min-win-rate 0.51", SCRIPT)

    def test_primary_gate_is_mandatory_before_promotion(self):
        self.assertIn('--anchor "${primary_metrics}"', SCRIPT)
        self.assertIn("paired_vs_primary.robust.json", SCRIPT)
        self.assertIn("--min-dsc-gain 0.005", SCRIPT)
        self.assertIn("--min-win-rate 0.52", SCRIPT)
        self.assertIn('if [[ "${primary_gate}" == PROMOTE ]]', SCRIPT)
        self.assertIn("decision=KEEP_PRIMARY", SCRIPT)

    def test_is_local_only_and_cannot_package(self):
        lowered = SCRIPT.lower()
        self.assertNotIn("codabench", lowered)
        self.assertNotIn("validation-public", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"public_read":false', lowered)
        self.assertIn('"hidden_read":false', lowered)


if __name__ == "__main__":
    unittest.main()
