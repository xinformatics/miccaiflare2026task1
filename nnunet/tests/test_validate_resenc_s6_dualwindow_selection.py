from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nnunet.validate_resenc_s6_dualwindow_selection import (
    DEPLOYMENTS,
    validate_public_selection,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY_SOURCE = (
    ROOT / "nnunet/RESENC_S6_DUALWINDOW_PUBLIC_SELECTION_POLICY_20260827.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidateDualWindowSelectionTests(unittest.TestCase):
    def fixture(self, root: Path, *, decision="REVIEW_TRAINED_CANDIDATE"):
        policy = root / "policy.json"
        policy.write_bytes(POLICY_SOURCE.read_bytes())
        model = root / "model"
        checkpoint = model / "fold_all/checkpoint_epoch_0025.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"dual-window-checkpoint")
        selected = {
            "epoch": 25,
            "deployment": "gf_anchor_confirm_either_p70",
            "analysis_configuration": "gf_anchor_confirm_either_p70",
            "Lesion_DSC": 0.65,
            "Lesion_NSD": 0.56,
            "DSC_gain": 0.0283360742348416,
            "NSD_gain": 0.0252479736823095,
            "qualified": True,
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha(checkpoint)},
        }
        terminal = root / "terminal.json"
        terminal.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "decision": decision,
                    "policy": {"path": str(policy.resolve()), "sha256": sha(policy)},
                    "predeclared_epochs": [25, 50, 100],
                    "predeclared_deployments": list(DEPLOYMENTS),
                    "selected": None if decision == "NO_HIDDEN_ARCHIVE" else selected,
                    "public_read": True,
                    "hidden_read": False,
                    "submission_created": False,
                    "uploaded": False,
                }
            ),
            encoding="utf-8",
        )
        return terminal, policy, model, checkpoint

    def test_accepts_exact_qualified_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            terminal, policy, model, checkpoint = self.fixture(Path(directory))
            selected = validate_public_selection(terminal, policy, model)
        self.assertEqual(selected.epoch, 25)
        self.assertEqual(selected.deployment, "gf_anchor_confirm_either_p70")
        self.assertEqual(selected.checkpoint, str(checkpoint.resolve()))

    def test_no_hidden_archive_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            terminal, policy, model, _ = self.fixture(
                Path(directory), decision="NO_HIDDEN_ARCHIVE"
            )
            self.assertIsNone(validate_public_selection(terminal, policy, model))

    def test_rejects_checkpoint_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            terminal, policy, model, checkpoint = self.fixture(Path(directory))
            checkpoint.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                validate_public_selection(terminal, policy, model)

    def test_rejects_candidate_below_frozen_dsc_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            terminal, policy, model, _ = self.fixture(Path(directory))
            payload = json.loads(terminal.read_text())
            payload["selected"]["Lesion_DSC"] = 0.63
            terminal.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "frozen gates"):
                validate_public_selection(terminal, policy, model)

    def test_rejects_any_hidden_or_upload_side_effect(self):
        for field in ("hidden_read", "submission_created", "uploaded"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                terminal, policy, model, _ = self.fixture(Path(directory))
                payload = json.loads(terminal.read_text())
                payload[field] = True
                terminal.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "invalid.*terminal"):
                    validate_public_selection(terminal, policy, model)


if __name__ == "__main__":
    unittest.main()
