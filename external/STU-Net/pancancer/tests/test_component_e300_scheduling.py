import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "run_component_e300_public_20260826.sh"
).read_text(encoding="utf-8")


class ComponentEpoch300SchedulingTests(unittest.TestCase):
    def test_uses_immutable_epoch300_endpoint_on_free_gpu_one(self):
        self.assertIn("checkpoint_inference_epoch_0300.pth", SCRIPT)
        self.assertIn('p.get("current_epoch")!=300', SCRIPT)
        self.assertIn('--checkpoint "$(basename "${checkpoint}")" --gpu 1', SCRIPT)

    def test_waits_for_completed_epoch200_evaluation(self):
        self.assertIn('while [[ ! -s "${e200_terminal}" ]]', SCRIPT)
        self.assertIn('p.get("status") != "COMPLETE"', SCRIPT)
        self.assertIn('p.get("evaluation_epoch") != 200', SCRIPT)
        self.assertLess(
            SCRIPT.index('p.get("evaluation_epoch") != 200'),
            SCRIPT.index('for part_id in $(seq 0 49)'),
        )

    def test_does_not_interrupt_training_or_touch_hidden_data(self):
        self.assertNotIn("systemctl --user stop", SCRIPT)
        self.assertNotIn("Validation-Hidden", SCRIPT)
        self.assertNotIn("zip -", SCRIPT)
        self.assertNotIn("codabench", SCRIPT.lower())
        self.assertIn(
            '"public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False',
            SCRIPT,
        )

    def test_closed_comparison_is_bound_to_epochs160_200_and300(self):
        self.assertIn("--workers 2 --closed-only", SCRIPT)
        self.assertIn('"epoch300_minus_epoch160":d160', SCRIPT)
        self.assertIn('"epoch300_minus_epoch200":d200', SCRIPT)
        self.assertIn('"epoch300_minus_best_previous":gain_over_best', SCRIPT)
        self.assertIn("over the better of component epochs 160 and 200", SCRIPT)

    def test_oom_restart_skips_only_fully_validated_completed_shards(self):
        self.assertIn('case_terminal="${output}/E300_COMPONENT_CASE_${part4}.json"', SCRIPT)
        self.assertIn('p.get("checkpoint",{}).get("sha256")!=expected_sha', SCRIPT)
        self.assertIn('p.get("case_count")!=1', SCRIPT)
        self.assertIn("probability.stat().st_size<1", SCRIPT)
        self.assertIn("reusing validated component epoch-300 public shard", SCRIPT)
        self.assertLess(
            SCRIPT.index("reusing validated component epoch-300 public shard"),
            SCRIPT.index('"${python_bin}" -u "${predictor}"'),
        )


if __name__ == "__main__":
    unittest.main()
