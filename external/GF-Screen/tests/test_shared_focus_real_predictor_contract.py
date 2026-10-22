import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "docker_submission/predict_dual_glance_shared_focus.py").read_text(
    encoding="utf-8"
)


class SharedFocusRealPredictorContractTests(unittest.TestCase):
    def test_hash_binds_both_deployment_checkpoints(self):
        self.assertIn("04a3cf7cb5209e65", SCRIPT)
        self.assertIn("3008615b57960e36", SCRIPT)
        self.assertIn("ba5a4ef9462bb191", SCRIPT)
        self.assertIn("db8e5769afd43c9e", SCRIPT)
        self.assertIn("Focus tensors are not bit-identical", SCRIPT)

    def test_compact_assets_are_paired_and_provenance_bound(self):
        self.assertIn("compact_pair", SCRIPT)
        self.assertIn('bbox_payload.get("source_sha256")', SCRIPT)
        self.assertIn('cbg_payload.get("source_sha256")', SCRIPT)
        self.assertIn('asset_mode = "compact_shared_focus"', SCRIPT)

    def test_native_deployment_contract_is_frozen(self):
        for contract in (
            'grid_mode="legacy"',
            "selected_a = actions_a.softmax(1)[:, 1] > 0.50",
            "selected_b = actions_b.softmax(1)[:, 1] > 0.20",
            "probability_a[:, 0] >= 0.50",
            "probability_b[:, 0] >= 0.50",
            'mode="uniform"',
        ):
            self.assertIn(contract, SCRIPT)

    def test_focus_runs_on_union_and_accumulates_routes_separately(self):
        self.assertIn("selected_union = selected_a | selected_b", SCRIPT)
        self.assertIn("bbox.segmentation_model, crops[ids]", SCRIPT)
        self.assertIn("sum_a, count_a", SCRIPT)
        self.assertIn("sum_b, count_b", SCRIPT)

    def test_guided_route_reuses_the_same_focus_cache_exactly(self):
        self.assertIn('parser.add_argument("--guided-proposal"', SCRIPT)
        self.assertIn("min_voxels=500", SCRIPT)
        self.assertIn("selected_guided = selected_a | proposal_selected", SCRIPT)
        self.assertIn("selected_union = selected_union | selected_guided", SCRIPT)
        self.assertIn("sum_guided", SCRIPT)
        self.assertIn("minimum_mm3=500.0", SCRIPT)

    def test_public_only_without_packaging_or_upload(self):
        self.assertIn('choices=("public",)', SCRIPT)
        self.assertIn('"public_labels_read": False', SCRIPT)
        self.assertIn('"hidden_read": False', SCRIPT)
        for forbidden in ("package_codabench", "requests.post", "codabench.org", "curl "):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
