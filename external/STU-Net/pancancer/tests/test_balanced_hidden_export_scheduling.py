import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
ARMER = (
    PANCANCER / "arm_balanced_public_qualified_hidden_export_20260826.sh"
).read_text(encoding="utf-8")
SAVER = (PANCANCER / "save_public_qualified_ensemble_20260825.sh").read_text(
    encoding="utf-8"
)


class BalancedHiddenExportSchedulingTests(unittest.TestCase):
    def test_only_public_qualified_predeclared_endpoint_can_transition(self):
        self.assertIn('"${decision}" != REVIEW_TRAINED_CANDIDATE', ARMER)
        self.assertIn('"${decision}" == NO_HIDDEN_ARCHIVE', ARMER)
        self.assertIn('selection.get("predeclared_epochs")!=[40,80]', ARMER)
        self.assertIn("epoch not in (40,80)", ARMER)
        self.assertIn("not selected.get(\"public_slot_gate\")", ARMER)
        self.assertLess(
            ARMER.index('"${decision}" != REVIEW_TRAINED_CANDIDATE'),
            ARMER.index('exec env'),
        )

    def test_exact_checkpoint_analysis_and_metric_gate_are_bound(self):
        self.assertIn("checkpoint_inference_epoch_{epoch:04d}.pth", ARMER)
        self.assertIn('checkpoint_item.get("sha256")!=sha(checkpoint)', ARMER)
        self.assertIn('analysis_item.get("sha256")!=sha(analysis)', ARMER)
        self.assertIn("0.6366639257651584", ARMER)
        self.assertIn("0.5347520263176905", ARMER)
        self.assertIn('best.get("configuration")!="gf_anchor_confirm_either_p70"', ARMER)

    def test_balanced_model_and_short_archive_tag_are_replayed(self):
        self.assertIn("STUNET_COMPONENT_MODEL_DIR", ARMER)
        self.assertIn("STUNET_ALLOWED_EPOCHS=\"40,80\"", ARMER)
        self.assertIn('archive_tag="gf_bal_e${epoch}_p70_0826"', ARMER)
        self.assertIn("STUNET_ARCHIVE_TAG", ARMER)
        self.assertIn("STUNET_COMPONENT_MODEL_DIR", SAVER)
        self.assertIn("STUNET_ARCHIVE_TAG", SAVER)

    def test_no_upload_capability_is_added(self):
        for forbidden in (
            "codabench.org",
            "requests.post",
            "curl ",
            "selenium",
            "Validation-Hidden-Labels",
        ):
            self.assertNotIn(forbidden, ARMER)


if __name__ == "__main__":
    unittest.main()
