import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
SCRIPT = (PANCANCER / "run_interim_e20_public_ensemble_20260825.sh").read_text(encoding="utf-8")


class InterimE20PublicSchedulingTest(unittest.TestCase):
    def test_waits_for_exact_resumable_epoch20_pair_before_pause(self):
        self.assertIn('interim_epoch="${STUNET_INTERIM_EPOCH:-20}"', SCRIPT)
        self.assertIn('checkpoint_epoch_${epoch4}.pth', SCRIPT)
        self.assertIn("checkpoint_latest.pth", SCRIPT)
        self.assertIn('int(x.get("current_epoch",-1))!=epoch', SCRIPT)
        self.assertIn("torch.serialization.safe_globals", SCRIPT)
        self.assertLess(SCRIPT.index("stable=0"), SCRIPT.index('systemctl --user stop'))

    def test_uses_both_gpu_partitions_sequentially_and_memory_safe_analysis(self):
        self.assertIn("for part_id in $(seq 0 49)", SCRIPT)
        self.assertIn("--num-parts 50", SCRIPT)
        self.assertIn("analyze_stunet_gf_ensemble.py", SCRIPT)
        self.assertIn('analysis_workers="${STUNET_ANALYSIS_WORKERS:-4}"', SCRIPT)
        self.assertIn('--workers "${analysis_workers}"', SCRIPT)
        self.assertNotIn("case_pids", SCRIPT)
        self.assertIn('gpu=$((part_id % 2))', SCRIPT)
        self.assertIn("--save-probabilities", SCRIPT)
        self.assertIn("--stunet-probabilities", SCRIPT)

    def test_success_hands_off_while_failure_resumes(self):
        self.assertIn("if [[ \"${status}\" -ne 0 ]]; then resume_training", SCRIPT)
        self.assertIn("Successful public completion deliberately leaves DDP paused", SCRIPT)
        self.assertIn("run_training_after_independent_audit_20260825.sh", SCRIPT)

    def test_public_only_and_immutable_gate(self):
        self.assertIn("0.6154316379500749", SCRIPT)
        self.assertIn("0.5104225107769019", SCRIPT)
        lowered = SCRIPT.lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("codabench.org", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"hidden_read":False', SCRIPT)
        self.assertIn('"submission_created":False', SCRIPT)
        self.assertIn("eligible=[row for row in rows", SCRIPT)
        self.assertIn("best=eligible[0] if eligible else rows[0]", SCRIPT)


if __name__ == "__main__":
    unittest.main()
