import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
SCRIPT = (PANCANCER / "run_public_ensemble_after_training_20260825.sh").read_text(encoding="utf-8")


class PublicEnsembleSchedulingTest(unittest.TestCase):
    def test_waits_for_complete_training_and_hash_binds_trajectory(self):
        self.assertIn("TRAINING_TERMINAL.json", SCRIPT)
        self.assertIn("trajectory_checkpoints", SCRIPT)
        self.assertIn('epochs=(100 200 300)', SCRIPT)
        self.assertIn('checkpoint_epoch_{epoch:04d}.pth', SCRIPT)

    def test_uses_both_gpus_serially_and_predeclared_ensemble(self):
        self.assertIn('epochs=(100 200 300)', SCRIPT)
        self.assertIn("infer_epoch_casewise", SCRIPT)
        self.assertIn("for part_id in $(seq 0 49)", SCRIPT)
        self.assertIn('gpu=$((part_id % 2))', SCRIPT)
        self.assertIn("analyze_stunet_gf_ensemble.py", SCRIPT)
        self.assertIn('--workers "${analysis_workers}"', SCRIPT)
        self.assertNotIn("--workers 16", SCRIPT)
        self.assertIn("--freetumor", SCRIPT)

    def test_stunet_exports_use_bounded_host_memory(self):
        self.assertIn('--workers 1 --save-probabilities "${compact_args[@]}" --dataset-role public', SCRIPT.replace("\\\n      ", ""))
        self.assertNotIn("--workers 4 --dataset-role public", SCRIPT)
        self.assertNotIn("pid100", SCRIPT)
        self.assertNotIn("pid200", SCRIPT)
        self.assertNotIn("pid0", SCRIPT)
        self.assertIn("--stunet-probabilities", SCRIPT)
        self.assertIn("*.prob16.npz", SCRIPT)

    def test_direct_compact_requires_exact_real_case_equivalence(self):
        self.assertIn("direct_compact_equivalence_e100_20260826/EQUIVALENCE.json", SCRIPT)
        self.assertIn('x.get("maximum_probability_code_difference")!=0', SCRIPT)
        self.assertIn('x.get("checkpoint",{}).get("sha256")!=sha(source)', SCRIPT)
        self.assertIn("--direct-compact-probabilities", SCRIPT)

    def test_public_gate_and_no_hidden_or_submission(self):
        self.assertIn("0.6154316379500749", SCRIPT)
        self.assertIn("0.5104225107769019", SCRIPT)
        lowered=SCRIPT.lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("codabench", lowered)
        self.assertNotIn("zip -", lowered)
        self.assertIn('"hidden_read":False', SCRIPT)
        self.assertIn('"submission_created":False', SCRIPT)
        self.assertIn('"uploaded":False', SCRIPT)
        self.assertIn("eligible=[row for row in rows", SCRIPT)
        self.assertIn("best=eligible[0] if eligible else rows[0]", SCRIPT)


if __name__ == "__main__":
    unittest.main()
