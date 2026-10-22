import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
SCRIPT = (
    PANCANCER / "queue_anchor_sparse_e160_after_sourcestrat_20260827.sh"
).read_text(encoding="utf-8")


class AnchorSparseQueueSchedulingTests(unittest.TestCase):
    def test_training_terminal_is_validated_before_probe(self):
        wait = 'while [[ ! -s "${training_terminal}" ]]'
        validation = 'p.get("status")!="COMPLETE"'
        probe = 'SPARSE_PROBE_GPU=0'
        self.assertIn(wait, SCRIPT)
        self.assertIn(validation, SCRIPT)
        self.assertLess(SCRIPT.index(wait), SCRIPT.index(validation))
        self.assertLess(SCRIPT.index(validation), SCRIPT.index(probe))

    def test_only_idle_gpu_zero_is_used(self):
        self.assertIn("nvidia-smi --id=0", SCRIPT)
        self.assertIn('"${used}" -le 1024', SCRIPT)
        self.assertIn("SPARSE_PROBE_GPU=0", SCRIPT)
        self.assertIn("SPARSE_EQUIVALENCE_GPU=0", SCRIPT)

    def test_all50_requires_successful_probe(self):
        self.assertIn('if [[ "${probe_ok}" != true ]]', SCRIPT)
        self.assertIn('/usr/bin/bash "${probe}"', SCRIPT)
        self.assertLess(
            SCRIPT.index('/usr/bin/bash "${probe}"'),
            SCRIPT.index('/usr/bin/bash "${all50}"'),
        )

    def test_no_hidden_packaging_or_upload_path(self):
        for forbidden in (
            "Validation-Hidden",
            "package_codabench",
            "codabench.org",
            "requests.post",
            "curl ",
        ):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
