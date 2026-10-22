import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_source_stunet_refiner_public_save_20260827.sh").read_text()


class SourceStunetRefinerPublicSaveTests(unittest.TestCase):
    def test_public_replay_starts_from_actual_epoch160_p70_champion(self):
        self.assertIn("gf_anchor_confirm_either_p70", SCRIPT)
        self.assertIn("0.6216639257651584", SCRIPT)
        self.assertIn("0.5347520263176905", SCRIPT)
        self.assertIn("epoch160_p70_champion", SCRIPT)
        self.assertIn("materialize_stunet_gf_ensemble.py", SCRIPT)

    def test_threshold_and_seed_mode_are_frozen_from_oof(self):
        self.assertIn("QUALIFY_STUNET_SEED_FIXED_PUBLIC", SCRIPT)
        self.assertIn('g.get("deployment_mode")!="seed"', SCRIPT)
        self.assertIn('--source-mode source_stunet=seed', SCRIPT)
        self.assertNotIn("threshold_sweep", SCRIPT)

    def test_public_gate_requires_material_dsc_and_preserved_nsd(self):
        self.assertIn("--min-dsc-gain 0.015", SCRIPT)
        self.assertIn("--min-nsd-gain 0.0", SCRIPT)
        self.assertIn("c[\"Lesion_DSC\"]>=0.6366639257651584", SCRIPT)
        self.assertIn("c[\"Lesion_NSD\"]>=0.5347520263176905", SCRIPT)

    def test_hidden_is_conditional_label_blind_and_short_named(self):
        self.assertIn('if [[ "${public_decision}" != QUALIFY_HIDDEN_SAVE ]]', SCRIPT)
        self.assertIn("Validation-Hidden-Images", SCRIPT)
        self.assertNotIn("Validation-Hidden-Labels", SCRIPT)
        self.assertIn("gf_ssr_0827.zip", SCRIPT)
        self.assertIn('"hidden_labels_read":False', SCRIPT)
        self.assertIn('"uploaded":False', SCRIPT)
        for forbidden in ("codabench.org", "requests.post", "curl ", "selenium"):
            self.assertNotIn(forbidden, SCRIPT)

    def test_every_resumable_inference_cache_is_revalidated(self):
        self.assertEqual(
            SCRIPT.count("validate_candidate_refiner_inference_marker.py"), 4
        )
        self.assertIn("validate_mask_inventory", SCRIPT)
        self.assertIn("hidden source-STU shard case union mismatch", SCRIPT)
        self.assertIn('p.get("checkpoint",{}).get("sha256")!=expected_sha', SCRIPT)

    def test_hidden_reconstruction_is_exactly_one_hundred_cases(self):
        self.assertIn(
            'validate_mask_inventory "${hidden_images}" "${hidden_base}" 100', SCRIPT
        )
        self.assertIn(
            'validate_mask_inventory "${hidden_images}" "${hidden_candidate}" 100',
            SCRIPT,
        )


if __name__ == "__main__":
    unittest.main()
