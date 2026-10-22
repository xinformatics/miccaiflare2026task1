import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "nnunet" / "run_resenc_s6_public_endpoints_20260827.sh"
POSITIVE_SCRIPT = (
    ROOT / "nnunet" / "run_resenc_s6_positive_public_endpoints_20260827.sh"
)


class ResEncS6PublicControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.positive_source = POSITIVE_SCRIPT.read_text(encoding="utf-8")

    def test_all_and_only_predeclared_endpoints_are_replayed(self):
        self.assertIn("epochs=(100 200 300 400 500)", self.source)
        self.assertIn("RESENC_S6_PUBLIC_SELECTION_POLICY_20260827.json", self.source)
        self.assertIn("select_resenc_s6_public.py", self.source)

    def test_inference_is_single_case_sequential_and_direct_compact(self):
        self.assertIn('--num-parts 50 --workers 1 --save-probabilities', self.source)
        self.assertIn('--direct-compact-probabilities --sequential', self.source)
        self.assertIn('nvidia-smi --id=0', self.source)

    def test_public_only_controller_has_no_hidden_or_upload_capability(self):
        self.assertIn("Validation-Public-Images", self.source)
        self.assertIn("Validation-Public-Labels", self.source)
        for forbidden in (
            "Validation-Hidden-Images",
            "Validation-Hidden-Labels",
            "codabench.org",
            "requests.post",
            "curl ",
            "zip ",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_checkpoint_and_case_terminals_are_hash_bound(self):
        self.assertIn("checkpoint_sha=$(sha256sum", self.source)
        self.assertIn('p.get("checkpoint",{}).get("sha256")!=sha', self.source)
        self.assertIn('p.get("trainer")!="nnUNetTrainerResEncS6PanCancerPartialBox"', self.source)

    def test_each_endpoint_runs_as_soon_as_its_checkpoint_releases_the_gpu(self):
        self.assertNotIn(
            "waiting for corrected ResEnc S6 training terminal", self.source
        )
        self.assertIn(
            "waiting for corrected ResEnc endpoint checkpoint", self.source
        )
        self.assertIn("ENDPOINT_TERMINAL.json", self.source)
        self.assertIn('p.get("dataset_role")!="public"', self.source)
        self.assertIn("hashlib.sha256(analysis.read_bytes())", self.source)

    def test_endpoint_terminals_require_all_four_frozen_deployments(self):
        expected = 'len(payload.get("metrics",{}))!=4'
        self.assertIn(expected, self.source)
        self.assertIn(expected, self.positive_source)


if __name__ == "__main__":
    unittest.main()
