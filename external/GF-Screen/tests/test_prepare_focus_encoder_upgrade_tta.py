import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prepare_focus_encoder_upgrade_tta import prepare


class PrepareFocusEncoderUpgradeTTATests(unittest.TestCase):
    def fixture(self, root: Path):
        route = root / "routes.json"
        route.write_text(
            json.dumps(
                {
                    "folds": 5,
                    "seed": 20260823,
                    "expected_epochs": 11,
                    "training_recipe": {
                        "freeze_encoder": False,
                        "encoder_lr_scale": 0.1,
                    },
                    "public_read": False,
                    "hidden_read": False,
                    "submission_created": False,
                }
            ),
            encoding="utf-8",
        )
        decision = root / "decision.json"
        decision.write_text(
            json.dumps(
                {
                    "decision": "PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF",
                    "passed": True,
                    "qualified_routes": ["lung_cells"],
                    "mature_route": "B32_UNCHANGED",
                    "candidate_summary": str(route.resolve()),
                    "candidate_summary_sha256": hashlib.sha256(
                        route.read_bytes()
                    ).hexdigest(),
                    "public_read": False,
                    "hidden_read": False,
                    "submission_created": False,
                }
            ),
            encoding="utf-8",
        )
        return route, decision

    def test_binds_only_qualified_deployed_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            route, decision = self.fixture(Path(directory))
            result = prepare(route, decision)
            self.assertEqual(result["decision"], "PROMOTE_TTA_TO_FIVEFOLD")
            self.assertEqual(result["eligible_groups"], ["lung_cells"])

    def test_rejects_changed_route_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            route, decision = self.fixture(Path(directory))
            route.write_text(route.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not bound"):
                prepare(route, decision)


if __name__ == "__main__":
    unittest.main()
