import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_recall_shortlist_public_after_grid_20260825.sh"


class RecallShortlistPublicSchedulingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_waits_for_completed_internal_grid_and_hash_binds_shortlist(self):
        self.assertIn("waiting for recall-aware internal grid", self.text)
        self.assertIn('identity.get("sha256")!=sha(shortlist_path)', self.text)
        self.assertIn('promotion_authority")!="fixed_public_50_only"', self.text)

    def test_screens_at_most_three_high_value_candidates_in_two_gpu_waves(self):
        self.assertIn('not 1<=len(candidates)<=3', self.text)
        self.assertIn(
            "case_balanced_dice_plus_ge3000mm3_and_voxel_recall", self.text
        )
        self.assertIn(
            "best_large_component_recall_ge3000_within_dice_guard", self.text
        )
        self.assertIn('evaluate_epoch "${epochs[${index}]}" 0', self.text)
        self.assertIn('evaluate_epoch "${epochs[$((index + 1))]}" 1', self.text)
        self.assertIn('index=$((index + 2))', self.text)

    def test_fixed_public_evaluator_and_selector_are_explicit(self):
        self.assertIn("evaluate_long_focus_pu_champion_public_20260825.sh", self.text)
        self.assertIn("select_recall_shortlist_public.py", self.text)
        self.assertIn('[[ -s "${child}/TERMINAL.json"', self.text)

    def test_hidden_packaging_and_upload_are_unreachable(self):
        self.assertNotIn("Validation-Hidden", self.text)
        self.assertNotIn("zip ", self.text)
        self.assertNotIn("codabench", self.text.lower())
        self.assertNotIn("curl ", self.text)

    def test_failure_provenance_detects_partial_public_work(self):
        self.assertIn('work.rglob("*.nii.gz")', self.text)
        self.assertIn('work.glob("epoch_*/*.metrics.csv")', self.text)
        self.assertIn('get("public_read")', self.text)


if __name__ == "__main__":
    unittest.main()
