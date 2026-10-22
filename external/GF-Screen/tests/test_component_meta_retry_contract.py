import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OOF = (ROOT / "run_component_meta_focus_fusion_fullvolume_oof_20260824.sh").read_text()
PUBLIC = (ROOT / "run_component_meta_focus_fusion_public_20260825.sh").read_text()
HIDDEN = (ROOT / "save_component_meta_focus_fusion_candidate_20260825.sh").read_text()


class ComponentMetaRetryContractTests(unittest.TestCase):
    def test_only_nonfinal_fail_closed_states_are_archived(self):
        self.assertIn("COMPLETE) exit 0", OOF)
        self.assertIn("FAILED|SKIPPED)", OOF)
        self.assertIn("COMPLETE) exit 0", PUBLIC)
        self.assertIn("FAILED|SKIPPED_OOF)", PUBLIC)
        self.assertIn("COMPLETE) exit 0", HIDDEN)
        self.assertIn("FAILED|SKIPPED_PUBLIC)", HIDDEN)
        for script in (OOF, PUBLIC, HIDDEN):
            self.assertIn("mv --no-clobber", script)
            self.assertIn("_retry_$(date +%Y%m%dT%H%M%S%N)", script)

    def test_public_and_hidden_separation_remains_intact(self):
        self.assertNotIn("Validation-Hidden", OOF)
        self.assertNotIn("Validation-Hidden", PUBLIC)
        self.assertNotIn("package_codabench", PUBLIC)
        self.assertIn("Validation-Hidden", HIDDEN)
        self.assertNotIn("codabench.org", HIDDEN.lower())


if __name__ == "__main__":
    unittest.main()
