import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "run_balanced_e40_e80_public_20260826.sh"
).read_text(encoding="utf-8")


class BalancedE40E80SchedulingTests(unittest.TestCase):
    def test_only_predeclared_midpoint_and_endpoint_are_evaluated(self):
        self.assertIn("epochs=(40 80)", SCRIPT)
        self.assertIn('checkpoint_inference_epoch_${epoch4}.pth', SCRIPT)
        self.assertIn('p.get("current_epoch")!=epoch', SCRIPT)
        self.assertIn('"predeclared_epochs":[40,80]', SCRIPT)

    def test_serializes_after_original_epoch300_or_recorded_deferral_and_uses_gpu_one(self):
        self.assertIn('while [[ ! -s "${e300_terminal}" && ! -s "${e300_deferred}" ]]', SCRIPT)
        self.assertIn('p.get("evaluation_epoch") != 300', SCRIPT)
        self.assertIn('p.get("completed_case_count")!=24', SCRIPT)
        self.assertIn('"trajectory_dependency"', SCRIPT)
        self.assertIn('--checkpoint "$(basename "${checkpoint}")" --gpu 1', SCRIPT)
        self.assertNotIn("systemctl --user stop", SCRIPT)

    def test_waits_for_training_completion_to_avoid_host_memory_concurrency(self):
        self.assertIn('while [[ ! -s "${training_terminal}" ]]', SCRIPT)
        self.assertIn('p.get("status")!="COMPLETE"', SCRIPT)
        self.assertIn('"training_terminal"', SCRIPT)
        self.assertLess(
            SCRIPT.index('while [[ ! -s "${training_terminal}" ]]'),
            SCRIPT.index('for epoch in "${epochs[@]}"'),
        )

    def test_uses_new_best_slot_gate_and_preserves_nsd(self):
        self.assertIn("0.6366639257651584", SCRIPT)
        self.assertIn("0.5347520263176905", SCRIPT)
        self.assertIn('"PUBLIC_SLOT_GATE_MET" if gate', SCRIPT)
        self.assertIn('qualified=[row for row in rows if row["public_slot_gate"]]', SCRIPT)
        self.assertIn("selected=max(qualified or rows", SCRIPT)
        self.assertIn('"REVIEW_TRAINED_CANDIDATE" if qualified', SCRIPT)
        self.assertIn("maximum fixed-p70 public DSC among endpoints satisfying", SCRIPT)

    def test_nsd_failing_higher_dsc_cannot_displace_qualified_endpoint(self):
        rows = [
            {
                "epoch": 40,
                "Lesion_DSC": 0.637,
                "Lesion_NSD": 0.535,
                "public_slot_gate": True,
            },
            {
                "epoch": 80,
                "Lesion_DSC": 0.650,
                "Lesion_NSD": 0.530,
                "public_slot_gate": False,
            },
        ]
        qualified = [row for row in rows if row["public_slot_gate"]]
        selected = max(
            qualified or rows,
            key=lambda row: (row["Lesion_DSC"], row["Lesion_NSD"]),
        )
        self.assertEqual(selected["epoch"], 40)

    def test_public_only_and_cannot_package_or_upload(self):
        self.assertIn("--workers 2 --closed-only", SCRIPT)
        self.assertNotIn("Validation-Hidden", SCRIPT)
        self.assertNotIn("codabench", SCRIPT.lower())
        self.assertNotIn("zip -", SCRIPT)
        self.assertIn(
            '"public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False',
            SCRIPT,
        )

    def test_oom_restart_skips_only_validated_complete_shards(self):
        self.assertIn('case_terminal="${output}/BALANCED_E${epoch4}_CASE_${part4}.json"', SCRIPT)
        self.assertIn('p.get("checkpoint",{}).get("sha256")!=expected_sha', SCRIPT)
        self.assertIn('p.get("case_count")!=1', SCRIPT)
        self.assertIn("probability.stat().st_size<1", SCRIPT)
        self.assertIn("reusing validated balanced epoch-%s public shard", SCRIPT)


if __name__ == "__main__":
    unittest.main()
