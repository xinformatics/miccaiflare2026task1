import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_interim_e303_predict_and_resume_20260825.sh").read_text(
    encoding="utf-8"
)


class InterimEpoch303SchedulingTests(unittest.TestCase):
    def test_waits_for_complete_resumable_checkpoint_before_pause(self):
        self.assertLess(SCRIPT.index('int(x["epoch"])!=302'), SCRIPT.index('systemctl --user stop'))
        self.assertIn('{"state_dict","optimizer_seg","scheduler_seg","epoch"}', SCRIPT)
        self.assertIn('checkpoint_sha_before', SCRIPT)
        self.assertIn('checkpoint_sha_after', SCRIPT)

    def test_public_gate_precedes_hidden_transition(self):
        self.assertLess(
            SCRIPT.index('if [[ "${public_decision}" != QUALIFY_SUBMISSION_PREPARATION ]]'),
            SCRIPT.index('"${hidden_started}" <<\'PY\''),
        )
        self.assertIn("0.6154316379500749", SCRIPT)
        self.assertIn("0.5104225107769019", SCRIPT)
        self.assertIn('row["gate"]!="PROMOTE"', SCRIPT)

    def test_hidden_inference_is_two_gpu_and_archive_exact(self):
        self.assertIn("infer_shard 0 0 &", SCRIPT)
        self.assertIn("infer_shard 1 1 &", SCRIPT)
        self.assertIn("--expected-count 100", SCRIPT)
        self.assertIn("len(members)!=100", SCRIPT)
        self.assertIn('"uploaded":False', SCRIPT)

    def test_training_resumes_on_failure_but_defers_after_success_for_ensemble(self):
        self.assertIn("trap finalize EXIT", SCRIPT)
        self.assertIn("resume_training || true", SCRIPT)
        self.assertIn("run_long_focus_pu_after_direct_public_20260825.sh", SCRIPT)
        self.assertIn("miccai-long-focus-pu-resume-e303-20260825", SCRIPT)
        self.assertIn("TRAINING_DEFERRED_FOR_ENSEMBLE.json", SCRIPT)
        self.assertIn("stunetb_ensemble_saved_20260825/TERMINAL.json", SCRIPT)
        self.assertLess(SCRIPT.index("terminal_status"), SCRIPT.index("resume_training || true"))

    def test_no_upload_or_hidden_label_access(self):
        for forbidden in ("codabench.org", "requests.post", "curl ", "selenium", "Validation-Hidden-Labels"):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
