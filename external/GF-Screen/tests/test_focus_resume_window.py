import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "run_focus_fusion_resume_window_20260826.sh"
).read_text(encoding="utf-8")


class FocusResumeWindowTests(unittest.TestCase):
    def test_window_is_bounded_and_yields_to_balanced_pipeline(self):
        self.assertIn("FOCUS_RESUME_WINDOW_SECONDS:-3300", SCRIPT)
        self.assertIn("--signal=TERM --kill-after=60s", SCRIPT)
        self.assertIn("balanced_terminal=", SCRIPT)
        self.assertIn('[[ ! -s "${balanced_terminal}" ]]', SCRIPT)

    def test_requires_idle_gpu1_and_safe_host_memory(self):
        self.assertIn("nvidia-smi --id=1", SCRIPT)
        self.assertIn("available>=30*1024*1024", SCRIPT)
        self.assertIn("pressure<=2.0", SCRIPT)
        self.assertIn("gpu<=1024", SCRIPT)

    def test_window_records_progress_without_submission_capability(self):
        self.assertIn("fold0_prediction_count_before", SCRIPT)
        self.assertIn("fold0_prediction_count_after", SCRIPT)
        self.assertIn('"public_read":False,"hidden_read":False', SCRIPT)
        for forbidden in (
            "Validation-Public",
            "Validation-Hidden",
            "codabench.org",
            "requests.post",
            "curl ",
            "package_codabench",
        ):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
