from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

from nnunet.predict_resenc_s6_dualwindow_safe import (  # noqa: E402
    TRAINER_NAME,
    nnUNetTrainerResEncS6PanCancerDualWindow,
    rewrite_terminal,
    safe,
)
from nnunet.run_resenc_s6_dualwindow import resolve_trainer  # noqa: E402
from nnunet.select_resenc_s6_dualwindow_public import load_policy  # noqa: E402


class DualWindowDeploymentTests(unittest.TestCase):
    def test_runner_and_predictor_resolve_exact_trainer(self):
        self.assertIs(
            resolve_trainer(None, TRAINER_NAME, None),
            nnUNetTrainerResEncS6PanCancerDualWindow,
        )
        self.assertIs(
            safe.resolve_trainer(None, TRAINER_NAME, None),
            nnUNetTrainerResEncS6PanCancerDualWindow,
        )

    def test_public_policy_is_frozen_before_training(self):
        policy = load_policy(
            PROJECT_ROOT
            / "nnunet/RESENC_S6_DUALWINDOW_PUBLIC_SELECTION_POLICY_20260827.json"
        )
        self.assertEqual(policy["checkpoint_epochs"], [25, 50, 100])
        self.assertEqual(policy["architecture_parameters"], 107_468_778)
        self.assertEqual(
            policy["selection"]["qualification"]["minimum_fixed_public_DSC"],
            0.6366639257651584,
        )
        self.assertFalse(policy["threshold_search_after_public_read"])

    def test_inference_terminal_records_dynamic_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal.json"
            path.write_text(
                json.dumps(
                    {"status": "COMPLETE", "trainer": TRAINER_NAME, "uploaded": False}
                ),
                encoding="utf-8",
            )
            rewrite_terminal(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["architecture_parameters"], 107_468_778)
        self.assertEqual(len(payload["input_views"]), 3)
        self.assertFalse(payload["uploaded"])

    def test_controllers_are_staged_hash_bound_and_do_not_upload(self):
        training = (
            PROJECT_ROOT / "nnunet/run_resenc_s6_dualwindow_2gpu_20260827.sh"
        ).read_text(encoding="utf-8")
        public = (
            PROJECT_ROOT
            / "nnunet/run_resenc_s6_dualwindow_public_endpoints_20260827.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("stage_end_epochs=(25 50 100)", training)
        self.assertIn("RESENC_DUALWINDOW_PARENT_CHECKPOINT", training)
        self.assertNotIn('launch_args=(-pretrained_weights', training)
        self.assertIn("resenc_s6_organ_target_endpoints_20260827/TERMINAL.json", training)
        self.assertIn('p.get("checkpoint",{}).get("sha256")!=sha', training)
        self.assertIn("ENDPOINT_TERMINAL.json", public)
        self.assertIn("direct-compact-probabilities", public)
        for source in (training, public):
            for forbidden in ("codabench.org", "requests.post", "selenium"):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
