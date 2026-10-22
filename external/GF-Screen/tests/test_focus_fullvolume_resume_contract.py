import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "run_focus_candidate_refiner_fullvolume_oof_20260823.sh"
).read_text(encoding="utf-8")


class FocusFullVolumeResumeContractTests(unittest.TestCase):
    def test_failed_terminal_is_archived_but_complete_or_skipped_is_terminal(self):
        self.assertIn("COMPLETE|SKIPPED) exit 0", SCRIPT)
        self.assertIn("FAILED)", SCRIPT)
        self.assertIn("failed_retry_$(date +%Y%m%dT%H%M%S%N)", SCRIPT)
        self.assertIn("mv --no-clobber", SCRIPT)

    def test_partial_prediction_cache_is_validated_and_reused(self):
        self.assertIn("inference_matches", SCRIPT)
        self.assertIn("apply_candidate_refiner.py\" infer", SCRIPT)
        self.assertIn("--batch-size 4 --workers 2 --resume", SCRIPT)

    def test_controller_remains_local_only(self):
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
