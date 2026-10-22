import hashlib
import json
import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
TRAINER = PANCANCER / "STUNetTrainerPanCancerSourceBalanced.py"
CONTROLLER = (
    PANCANCER / "run_component_e160_source_stratified_continuation_20260826.sh"
)
LOSS_AUDIT = (
    PANCANCER / "analysis/source_stratified_recall_loss_decision_20260826.json"
)
LOADER_SMOKE = (
    PANCANCER
    / "analysis/component_center_source_loader_smoke_20260826/TERMINAL.json"
)
BATCH_AUDIT = (
    PANCANCER / "analysis/source_batch_contract_correction_20260827.json"
)
AGGRESSIVE_CONTROLLER = (
    PANCANCER / "run_component_e160_source_aggressive_continuation_20260827.sh"
)


class SourceComponentLaunchContractTests(unittest.TestCase):
    def test_batch_correction_is_bound_to_current_controllers(self):
        correction = json.loads(BATCH_AUDIT.read_text(encoding="utf-8"))
        runtime = correction["actual_runtime_contract"]
        sources = correction["corrected_sources"]
        self.assertEqual(runtime["ddp_world_size"], 2)
        self.assertEqual(runtime["per_rank_batch_size"], 1)
        self.assertEqual(runtime["global_batch_size"], 2)
        self.assertFalse(correction["training_semantics_changed"])
        self.assertEqual(
            sources["source_stratified_controller_sha256"],
            hashlib.sha256(CONTROLLER.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            sources["source_aggressive_controller_sha256"],
            hashlib.sha256(AGGRESSIVE_CONTROLLER.read_bytes()).hexdigest(),
        )

    def test_audits_are_bound_to_current_trainer_source(self):
        trainer_sha = hashlib.sha256(TRAINER.read_bytes()).hexdigest()
        loss = json.loads(LOSS_AUDIT.read_text(encoding="utf-8"))
        smoke = json.loads(LOADER_SMOKE.read_text(encoding="utf-8"))
        self.assertEqual(loss["trainer_source_sha256"], trainer_sha)
        self.assertEqual(smoke["trainer_source"]["sha256"], trainer_sha)
        self.assertEqual(loss["loss"]["alpha_false_positive"], 0.25)
        self.assertEqual(loss["loss"]["beta_false_negative"], 0.75)
        self.assertEqual(smoke["component_center_fraction"], 0.25)

    def test_controller_fails_closed_around_recall_recipe(self):
        controller = CONTROLLER.read_text(encoding="utf-8")
        required_fragments = (
            'loss.get("loss",{}).get("alpha_false_positive")!=0.25',
            'loss.get("loss",{}).get("beta_false_negative")!=0.75',
            'smoke.get("component_center_fraction")!=0.25',
            'center.get("foreground_voxel_uniform_fraction")!=0.75',
            'center.get("annotated_component_uniform_fraction")!=0.25',
            'E40_INTERIM_COMPLETE.json',
            'interim_payload.get("epoch_terminal",{}).get("sha256")!=loss.get("balanced_epoch40_terminal_sha256")',
            'balanced_metric.get("Lesion_DSC")!=metric.get("Lesion_DSC")',
            'interim_metric.get("Lesion_NSD")!=metric.get("Lesion_NSD")',
            '"case_sampling":"75% uniform plus 25% size-bin-preserving bounded source target"',
            '"patch_center_sampling":"75% foreground-voxel uniform plus 25% annotated-component uniform"',
            '"per_rank_batch_size":1,"global_batch_size":2',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, controller)

    def test_training_controller_has_no_hidden_or_upload_target(self):
        controller = CONTROLLER.read_text(encoding="utf-8").lower()
        self.assertNotIn("validation-hidden", controller)
        self.assertNotIn("codabench", controller)
        self.assertNotIn("requests.post", controller)
        self.assertNotIn("curl ", controller)


if __name__ == "__main__":
    unittest.main()
