import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_interim_e293_predict_and_resume_20260825.sh").read_text(
    encoding="utf-8"
)


class InterimEpoch293SchedulingTests(unittest.TestCase):
    def test_public_gate_precedes_hidden_transition(self):
        self.assertLess(
            SCRIPT.index('if [[ "${public_decision}" != QUALIFY_SUBMISSION_PREPARATION ]]'),
            SCRIPT.index('"${hidden_started}" <<\'PY\''),
        )
        self.assertIn("0.6154316379500749", SCRIPT)
        self.assertIn("0.5104225107769019", SCRIPT)
        self.assertIn('row["gate"] != "PROMOTE"', SCRIPT)

    def test_archive_is_exact_and_upload_is_absent(self):
        self.assertIn("--expected-count 100", SCRIPT)
        self.assertIn('len(members) != 100', SCRIPT)
        self.assertIn('"uploaded": False', SCRIPT)
        for forbidden in ("codabench.org", "requests.post", "curl ", "selenium"):
            self.assertNotIn(forbidden, SCRIPT)

    def test_trainer_resume_is_an_exit_invariant(self):
        self.assertIn("trap finalize EXIT", SCRIPT)
        self.assertIn("resume_training || true", SCRIPT)
        self.assertIn("run_long_focus_pu_after_direct_public_20260825.sh", SCRIPT)
        self.assertIn("checkpoint_epoch_0297.pt", SCRIPT)

    def test_prediction_uses_only_unoccupied_gpu(self):
        self.assertIn("CUDA_VISIBLE_DEVICES=1", SCRIPT)


if __name__ == "__main__":
    unittest.main()
