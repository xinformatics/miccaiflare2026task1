import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "run_component_e200_public_20260826.sh"
).read_text(encoding="utf-8")


class ComponentEpoch200SchedulingTests(unittest.TestCase):
    def test_uses_component_epoch200_snapshot_on_gpu_zero(self):
        self.assertIn("checkpoint_inference_epoch_0200.pth", SCRIPT)
        self.assertIn('p.get("current_epoch")!=200', SCRIPT)
        self.assertIn('--checkpoint "$(basename "${checkpoint}")" --gpu 0', SCRIPT)

    def test_only_pauses_global_control_and_resumes_before_metrics(self):
        self.assertIn('systemctl --user stop "${base_unit}"', SCRIPT)
        self.assertNotIn('systemctl --user stop "${component_unit}"', SCRIPT)
        self.assertLess(
            SCRIPT.index("resume_base\nbase_paused=false"),
            SCRIPT.index('"${python_bin}" -u "${analyzer}"'),
        )
        self.assertIn("trap finalize EXIT", SCRIPT)

    def test_closed_public_only_no_submission(self):
        self.assertIn("--workers 2 --closed-only", SCRIPT)
        self.assertNotIn("Validation-Hidden", SCRIPT)
        self.assertNotIn("zip -", SCRIPT)
        self.assertNotIn("codabench", SCRIPT.lower())
        self.assertIn('"hidden_read":False,"submission_created":False,"uploaded":False', SCRIPT)


if __name__ == "__main__":
    unittest.main()
