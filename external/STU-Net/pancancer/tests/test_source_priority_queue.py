import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "queue_source_stratified_after_focus_20260826.sh"
).read_text(encoding="utf-8")


class SourcePriorityQueueTests(unittest.TestCase):
    def test_runs_after_balanced_but_no_longer_waits_for_focus(self):
        self.assertIn('while [[ ! -s "${balanced}" ]]', SCRIPT)
        self.assertIn('"REVIEW_TRAINED_CANDIDATE","NO_HIDDEN_ARCHIVE"', SCRIPT)
        self.assertIn('while [[ ! -s "${balanced_saved}" ]]', SCRIPT)
        self.assertNotIn("gf_focus_pu_long30_trajectory_saved", SCRIPT)
        self.assertNotIn("waiting for long Focus", SCRIPT)

    def test_priority_reason_and_two_gpu_service_are_explicit(self):
        self.assertIn("PRIORITIZE_RECALL_CORRECTION_AFTER_BALANCED_EVIDENCE", SCRIPT)
        self.assertIn("MemoryHigh=48G", SCRIPT)
        self.assertIn("MemoryMax=64G", SCRIPT)
        self.assertIn("Restart=on-failure", SCRIPT)

    def test_queue_cannot_read_hidden_labels_or_upload(self):
        for forbidden in (
            "Validation-Public",
            "Validation-Hidden",
            "codabench.org",
            "requests.post",
            "curl ",
            "package_codabench",
        ):
            self.assertNotIn(forbidden, SCRIPT)
        self.assertIn('"public_read":False,"hidden_read":False', SCRIPT)


if __name__ == "__main__":
    unittest.main()
