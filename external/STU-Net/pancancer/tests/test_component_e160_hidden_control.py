import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAVER = (ROOT / "save_component_e160_p70_now_20260826.sh").read_text(encoding="utf-8")
CONTROL = (ROOT / "run_component_e160_hidden_controlled_20260826.sh").read_text(encoding="utf-8")


class ComponentEpoch160HiddenControlTests(unittest.TestCase):
    def test_exact_public_candidate_is_bound(self):
        self.assertIn("checkpoint_inference_epoch_0160.pth", SAVER)
        self.assertIn('"selected_branch":"component_box"', SAVER)
        self.assertIn('"gf_anchor_confirm_either_p70"', SAVER)
        self.assertIn("STUNET_ALLOWED_EPOCHS=160", SAVER)
        self.assertIn("STUNET_HIDDEN_GPU=0", SAVER)

    def test_component_training_continues_and_base_is_restored(self):
        self.assertIn('systemctl --user stop "${base_unit}"', CONTROL)
        self.assertNotIn("component-gpu1", CONTROL)
        self.assertIn("trap resume_services EXIT", CONTROL)
        self.assertIn('systemctl --user stop "${e200_watcher}"', CONTROL)
        self.assertIn("run_component_e200_public_20260826.sh", CONTROL)

    def test_no_upload_or_hidden_labels(self):
        joined=SAVER+CONTROL
        for forbidden in ("requests.post", "curl ", "selenium", "Validation-Hidden-Labels"):
            self.assertNotIn(forbidden, joined)


if __name__ == "__main__":
    unittest.main()
