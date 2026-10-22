import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAVER = (ROOT / "save_long_glance_grl_after_public_20260825.sh").read_text(encoding="utf-8")
VALIDATOR = (ROOT / "validate_long_glance_grl_public_terminal.py").read_text(encoding="utf-8")


class LongGlanceGRLSaveSchedulingTest(unittest.TestCase):
    def test_failed_controller_terminal_is_not_consumed_as_final(self):
        self.assertIn('COMPLETE|SKIPPED) break', SAVER)
        self.assertIn('waiting for replacement of failed long Glance controller terminal', SAVER)
        self.assertIn("trap 'failure_line=${LINENO}' ERR", SAVER)

    def test_own_failed_terminal_is_preserved_for_resume(self):
        self.assertIn('COMPLETE|SKIPPED) trap - EXIT; exit 0', SAVER)
        self.assertIn('TERMINAL.failed_retry_', SAVER)
        self.assertIn('preserved failed long Glance saver terminal for resume', SAVER)

    def test_hash_validates_before_hidden_transition(self):
        self.assertIn("validate_long_glance_grl_public_terminal.py", SAVER)
        self.assertLess(
            SAVER.index("validate_long_glance_grl_public_terminal.py"),
            SAVER.index('"${python_bin}" - "${validation}" "${hidden_started}"'),
        )
        self.assertIn('"controller recipe"', VALIDATOR)
        self.assertIn("selected child checkpoint", VALIDATOR)
        self.assertIn("selected trajectory row is not the best eligible state", VALIDATOR)

    def test_failed_public_gate_creates_no_archive(self):
        self.assertIn('if [[ "${decision}" != QUALIFY_SUBMISSION_PREPARATION ]]', SAVER)
        self.assertIn("write_terminal SKIPPED KEEP_TOUCHCOMP", SAVER)
        self.assertLess(SAVER.index("write_terminal SKIPPED KEEP_TOUCHCOMP"), SAVER.index("package_codabench.py"))

    def test_reproduces_selected_threshold_and_deployment(self):
        self.assertIn('--glance_threshold "${glance_threshold}"', SAVER)
        self.assertIn('--lesion_threshold 0.50', SAVER)
        self.assertIn("--min-mm3 \"${minimum}\"", SAVER)
        self.assertIn("biomedparse_fusion.py", SAVER)
        self.assertIn("screen_disconnected_guided_additions.py", SAVER)
        self.assertIn("touching_component", SAVER)

    def test_exact_one_archive_manual_only(self):
        self.assertEqual(SAVER.count("package_codabench.py"), 1)
        self.assertIn("--expected-count 100", SAVER)
        self.assertIn("len(archive.name)>=64", SAVER)
        self.assertIn('"uploaded":False', SAVER)
        for forbidden in ("curl ", "wget ", "requests.post", "codabench.org"):
            self.assertNotIn(forbidden, SAVER)


if __name__ == "__main__":
    unittest.main()
