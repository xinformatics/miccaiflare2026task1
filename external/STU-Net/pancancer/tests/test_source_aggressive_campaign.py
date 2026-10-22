import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = (ROOT / "run_component_e160_source_aggressive_continuation_20260827.sh").read_text()
PUBLIC = (ROOT / "run_source_aggressive_e40_e80_public_20260827.sh").read_text()
EVALUATOR = (ROOT / "run_complete_hybrid_e40_e80_public_20260826.sh").read_text()
SAVER = (ROOT / "arm_source_aggressive_qualified_hidden_20260827.sh").read_text()
CAMPAIGN = (ROOT / "run_source_aggressive_campaign_20260827.sh").read_text()
COMPLETE_QUEUE = (ROOT / "queue_complete_hybrid_after_focus_20260826.sh").read_text()


class SourceAggressiveCampaignTests(unittest.TestCase):
    def test_training_is_exact_parent_two_gpu_and_loss_only(self):
        self.assertIn(
            "81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74",
            TRAIN,
        )
        self.assertIn("STUNetTrainerPanCancerComponentBoxE160SourceAggressive", TRAIN)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", TRAIN)
        self.assertIn("-num_gpus 2", TRAIN)
        self.assertIn('"per_rank_batch_size":1,"global_batch_size":2', TRAIN)
        self.assertIn("SOURCE_AGGRESSIVE_N_PROC_DA:-0", TRAIN)
        self.assertIn("Tversky a020 b080", TRAIN)
        self.assertIn('"Tversky alpha/beta only"', TRAIN)
        self.assertIn("checkpoint_latest.pth", TRAIN)
        self.assertIn("optimizer_state", TRAIN)

    def test_launch_is_fail_closed_on_predeclared_diagnostic(self):
        self.assertIn("QUALIFY_TVERSKY_A020_B080_FOLLOWUP", TRAIN)
        self.assertIn('not selected.get("qualifies")', TRAIN)
        self.assertIn('not all(selected.get("checks",{}).values())', TRAIN)
        self.assertIn("PartialBoxTverskyCELoss(alpha=0.20, beta=0.80)", TRAIN)
        self.assertIn('"public_read":False', TRAIN)
        self.assertIn('"hidden_read":False', TRAIN)
        self.assertIn('"uploaded":False', TRAIN)

    def test_public_evaluation_keeps_fixed_endpoints_and_existing_gate(self):
        self.assertIn('STUNET_EVALUATION_BRANCH="source_aggressive_e160"', PUBLIC)
        self.assertIn("STUNetTrainerPanCancerComponentBoxE160SourceAggressive", PUBLIC)
        self.assertIn('branch=="source_aggressive_e160"', EVALUATOR)
        self.assertIn("epochs=(40 80)", EVALUATOR)
        self.assertIn("dsc>=0.6366639257651584", EVALUATOR)
        self.assertIn("nsd>=0.5347520263176905", EVALUATOR)

    def test_campaign_skips_without_qualification_and_never_uploads(self):
        self.assertIn("KEEP_A025_B075_NO_STRONGER_LOSS", CAMPAIGN)
        self.assertIn("SKIPPED KEEP_A025_B075", CAMPAIGN)
        self.assertIn("component_meta_recovery_after_sourcestrat_20260826", CAMPAIGN)
        self.assertIn('"uploaded":False', CAMPAIGN)
        for forbidden in ("codabench.org", "requests.post", "curl ", "selenium"):
            self.assertNotIn(forbidden, TRAIN)
            self.assertNotIn(forbidden, CAMPAIGN)
            self.assertNotIn(forbidden, SAVER)

    def test_complete_hybrid_waits_for_conditional_campaign(self):
        self.assertIn("source_aggressive_campaign_20260827/TERMINAL.json", COMPLETE_QUEUE)
        self.assertIn('while [[ ! -s "${aggressive}" ]]', COMPLETE_QUEUE)
        self.assertIn('p.get("status") not in {"COMPLETE","SKIPPED"}', COMPLETE_QUEUE)
        self.assertIn('p.get("loss")!={"alpha_fp":0.20,"beta_fn":0.80}', COMPLETE_QUEUE)

    def test_hidden_save_requires_the_same_public_qualified_contract(self):
        self.assertIn('STUNET_EXPECTED_EVALUATION_BRANCH="source_aggressive_e160"', SAVER)
        self.assertIn('STUNET_BRANCH_ARCHIVE_PREFIX="gf_sa"', SAVER)
        self.assertNotIn("Validation-Hidden-Labels", SAVER)


if __name__ == "__main__":
    unittest.main()
