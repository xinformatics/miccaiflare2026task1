import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
SCRIPT = (PANCANCER / "run_anchor_sparse_e160_probe_20260827.sh").read_text(
    encoding="utf-8"
)


class AnchorSparseProbeSchedulingTests(unittest.TestCase):
    def test_probe_is_frozen_to_worst_runtime_case_and_epoch160(self):
        self.assertIn('part_id=14', SCRIPT)
        self.assertIn('case_name="FLARE23Ts_0029.nii.gz"', SCRIPT)
        self.assertIn('checkpoint_inference_epoch_0160.pth', SCRIPT)
        self.assertIn('probability>0.70', SCRIPT)

    def test_probe_requires_idle_gpu_and_55_second_reserve(self):
        self.assertIn('--query-gpu=memory.used', SCRIPT)
        self.assertIn('"${used}" -le 1024', SCRIPT)
        self.assertIn('wall<=55', SCRIPT)
        self.assertIn('selected_tiles",0)<row.get("full_tiles",0)', SCRIPT)

    def test_one_case_does_not_authorize_deployment(self):
        self.assertIn('"deployment_authorized":False', SCRIPT)
        self.assertIn('all 50 fixed-public final masks must match', SCRIPT)

    def test_public_only_without_submission_side_effect(self):
        self.assertIn('--dataset-role public', SCRIPT)
        self.assertIn('"public_labels_read":False', SCRIPT)
        self.assertIn('"hidden_read":False', SCRIPT)
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
