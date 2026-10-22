import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = (ROOT / "nnunet/run_resenc_s6_partialband12_2gpu_20260827.sh").read_text()
PUBLIC = (ROOT / "nnunet/run_resenc_s6_partialband12_public_20260827.sh").read_text()
PARENT = (ROOT / "nnunet/select_resenc_s6_partialband12_parent.py").read_text()


class PartialBand12ControllerTest(unittest.TestCase):
    def test_training_controller_is_fail_closed_until_explicit_arm(self):
        self.assertIn('auth.get("status")!="AUTHORIZED_AFTER_BASE_SELECTION"', TRAIN)
        self.assertIn('preflight.get("status")!="PASS_ARMED"', TRAIN)
        self.assertIn('controller_preflight.get("status")!="PASS_ARMED"', TRAIN)
        self.assertIn('get("queue_armed") is not True', TRAIN)
        self.assertIn("base training is not safely complete", TRAIN)
        self.assertIn("base fixed-public selection is not safely complete", PARENT)

    def test_runtime_consumes_hash_bound_controller_preflight(self):
        manifest = "RESENC_S6_PARTIALBAND12_CONTROLLER_PREFLIGHT_20260827.json"
        for source in (TRAIN, PUBLIC):
            self.assertIn(manifest, source)
            self.assertIn('safety.get("current_authorization_satisfies_launch") is not True', source)
            self.assertIn('safety.get("current_trainer_preflight_satisfies_launch") is not True', source)
            self.assertIn('sha(path)!=item.get("sha256")', source)

    def test_parent_is_public_selected_hash_bound_base_checkpoint(self):
        self.assertIn("select_resenc_s6_partialband12_parent import select_parent", TRAIN)
        self.assertIn("parent_record=select_parent(selection_path,expected)", TRAIN)
        self.assertIn("selected or diagnostic", PARENT)
        self.assertIn("ALLOWED_PARENT_EPOCHS", PARENT)
        self.assertIn('checkpoint_record.get("sha256") != observed_sha', PARENT)
        self.assertIn('state.get("trainer_name") != BASE_TRAINER', PARENT)
        self.assertIn('export RESENC_BAND12_PARENT_EPOCH="${parent_epoch}"', TRAIN)
        self.assertIn('export RESENC_BAND12_PARENT_SHA256="${parent_sha}"', TRAIN)

    def test_training_is_staged_and_releases_for_public_endpoints(self):
        self.assertIn("stage_end_epochs=(25 50 100)", TRAIN)
        self.assertIn('export RESENC_BAND12_STAGE_END_EPOCH="${stage_end}"', TRAIN)
        self.assertIn('-pretrained_weights "${parent}"', TRAIN)
        self.assertIn("ENDPOINT_TERMINAL.json", TRAIN)
        self.assertIn("checkpoint_epoch_${stage4}.pth", TRAIN)
        self.assertIn('[[ "${stage_end}" -eq 100 ]] && landmark="${final}"', TRAIN)

    def test_public_controller_is_public_only_and_sequential(self):
        self.assertIn("validation/Validation-Public-Images", PUBLIC)
        self.assertIn("validation/Validation-Public-Labels", PUBLIC)
        self.assertIn("epochs=(25 50 100)", PUBLIC)
        self.assertIn("--sequential --dataset-role public", PUBLIC)
        self.assertIn("--part-id \"${part_id}\" --num-parts 50", PUBLIC)
        self.assertIn("--save-probabilities", PUBLIC)
        for forbidden in (
            "Validation-Hidden",
            "hidden_predictions",
            "curl ",
            "codabench.org",
            "zip -",
        ):
            self.assertNotIn(forbidden, TRAIN)
            self.assertNotIn(forbidden, PUBLIC)

    def test_controllers_cannot_start_or_upload_services(self):
        self.assertNotIn("systemd-run", TRAIN)
        self.assertNotIn("systemd-run", PUBLIC)
        self.assertNotIn("docker push", TRAIN)
        self.assertNotIn("docker push", PUBLIC)


if __name__ == "__main__":
    unittest.main()
