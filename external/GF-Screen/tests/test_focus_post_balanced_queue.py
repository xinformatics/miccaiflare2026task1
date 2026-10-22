import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "queue_focus_e303_to313_after_balanced_20260826.sh"
).read_text(encoding="utf-8")


class FocusPostBalancedQueueTests(unittest.TestCase):
    def test_waits_for_valid_partial_box_aggressive_campaign(self):
        self.assertIn('while [[ ! -s "${aggressive_terminal}" ]]', SCRIPT)
        self.assertIn("source_aggressive_campaign_20260827", SCRIPT)
        self.assertIn('p.get("status") not in {"COMPLETE","SKIPPED"}', SCRIPT)
        self.assertIn('p.get("loss")!={"alpha_fp":0.20,"beta_fn":0.80}', SCRIPT)
        self.assertIn('p.get("hidden_labels_read")', SCRIPT)

    def test_excludes_invalid_complete_negative_dependencies(self):
        self.assertNotIn("component_meta_recovery_after_sourcestrat_20260826", SCRIPT)
        self.assertNotIn("stunetb_completehybrid_e40_e80_20260826", SCRIPT)
        self.assertIn("annotations as negatives", SCRIPT)

    def test_exact_checkpoint303_optimizer_resume_is_required(self):
        self.assertIn("76022fa403700a38ce2aa9042aad3446687020e259c0bb5d1a65d8f698dd0bbc", SCRIPT)
        self.assertIn('int(state.get("epoch",-1))!=302', SCRIPT)
        self.assertIn('not state.get("optimizer_seg")', SCRIPT)
        self.assertIn('not state.get("scheduler_seg")', SCRIPT)
        self.assertIn('[[ ! -e "${final_checkpoint}" ]]', SCRIPT)

    def test_launch_is_serialized_and_resource_gated(self):
        self.assertIn("available>=30*1024*1024", SCRIPT)
        self.assertIn("pressure<=2.0", SCRIPT)
        self.assertIn("gpu0<=4096 and gpu1<=1024", SCRIPT)
        self.assertIn("--property=Restart=on-failure", SCRIPT)
        self.assertIn("--property=StartLimitBurst=3", SCRIPT)

    def test_queue_has_no_prediction_or_upload_capability(self):
        for forbidden in (
            "Validation-Public",
            "Validation-Hidden",
            "codabench.org",
            "requests.post",
            "curl ",
            "package_codabench",
        ):
            self.assertNotIn(forbidden, SCRIPT)
        self.assertIn('"public_read":False,"hidden_read":False', SCRIPT)


if __name__ == "__main__":
    unittest.main()
