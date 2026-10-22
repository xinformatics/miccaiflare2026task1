import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restrict_focus_encoder_upgrade_routes import restrict


def write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RestrictFocusEncoderUpgradeRoutesTests(unittest.TestCase):
    def terminals(self, root: Path, pooled: list[str], pilot: list[str]):
        fivefold = {
            "decision": "PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF",
            "qualified_routes": pooled,
            "routes": {
                route: {"qualified_for_full_volume": route in pooled}
                for route in ("weak_sources", "lung_cells")
            },
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        pilot_decision = {
            "decision": "PROMOTE_ROUTES_TO_FIVEFOLD",
            "qualified_routes": pilot,
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        fivefold_path = root / "fivefold.json"
        pilot_path = root / "pilot.json"
        fivefold_hash = write(fivefold_path, fivefold)
        pilot_hash = write(pilot_path, pilot_decision)
        return (
            {
                "status": "COMPLETE",
                "decision": fivefold["decision"],
                "fivefold_decision": str(fivefold_path),
                "fivefold_decision_sha256": fivefold_hash,
                "public_read": False,
                "hidden_read": False,
                "submission_created": False,
            },
            {
                "status": "COMPLETE",
                "decision": pilot_decision["decision"],
                "route_decision": str(pilot_path),
                "route_decision_sha256": pilot_hash,
                "public_read": False,
                "hidden_read": False,
                "submission_created": False,
            },
        )

    def test_weak_cannot_bypass_lung_only_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            fivefold, pilot = self.terminals(
                Path(temporary), ["weak_sources", "lung_cells"], ["lung_cells"]
            )
            result = restrict(fivefold, pilot)
        self.assertEqual(result["qualified_routes"], ["lung_cells"])
        self.assertFalse(result["routes"]["weak_sources"]["qualified_for_full_volume"])
        self.assertEqual(
            result["routes"]["weak_sources"]["restriction_reason"],
            "route did not pass the predeclared fold-0 pilot gate",
        )

    def test_rejects_when_pilot_route_fails_pooled_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            fivefold, pilot = self.terminals(
                Path(temporary), ["weak_sources"], ["lung_cells"]
            )
            result = restrict(fivefold, pilot)
        self.assertEqual(result["decision"], "KEEP_FROZEN_FOCUS")
        self.assertEqual(result["qualified_routes"], [])

    def test_mutated_bound_decision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fivefold, pilot = self.terminals(root, ["lung_cells"], ["lung_cells"])
            Path(fivefold["fivefold_decision"]).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not bind"):
                restrict(fivefold, pilot)


if __name__ == "__main__":
    unittest.main()
