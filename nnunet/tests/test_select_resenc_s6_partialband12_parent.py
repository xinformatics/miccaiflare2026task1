import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from nnunet.select_resenc_s6_partialband12_parent import (
    BASE_TRAINER,
    select_parent,
)


class SelectPartialBand12ParentTest(unittest.TestCase):
    def make_fixture(self, root: Path, *, selected: bool = True, epoch: int = 200):
        fold = root / "base" / "fold_all"
        fold.mkdir(parents=True)
        checkpoint = fold / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(
            {
                "trainer_name": BASE_TRAINER,
                "current_epoch": epoch,
                "network_weights": {"weight": torch.ones(1)},
            },
            checkpoint,
        )
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        row = {
            "epoch": epoch,
            "deployment": "gf_anchor_confirm_either_p70",
            "Lesion_DSC": 0.65,
            "Lesion_NSD": 0.56,
            "qualified": selected,
            "checkpoint": {"path": str(checkpoint), "sha256": digest},
        }
        selection = root / "selection.json"
        selection.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "selected": row if selected else None,
                    "diagnostic_best": row,
                    "public_read": True,
                    "hidden_read": False,
                    "submission_created": False,
                    "uploaded": False,
                }
            )
        )
        return fold, checkpoint, selection

    def test_qualified_parent_is_preferred_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            fold, checkpoint, selection = self.make_fixture(Path(temporary))
            result = select_parent(selection, fold)
            self.assertEqual(result["source"], "qualified_selected")
            self.assertEqual(result["epoch"], 200)
            self.assertEqual(result["checkpoint"]["path"], str(checkpoint.resolve()))
            self.assertTrue(result["fixed_public_metrics"]["qualified"])
            self.assertFalse(result["public_labels_read"])

    def test_diagnostic_best_is_used_when_none_qualifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            fold, _, selection = self.make_fixture(Path(temporary), selected=False)
            result = select_parent(selection, fold)
            self.assertEqual(result["source"], "diagnostic_best")
            self.assertFalse(result["fixed_public_metrics"]["qualified"])

    def test_checkpoint_hash_and_fold_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fold, checkpoint, selection = self.make_fixture(root)
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                select_parent(selection, fold)
            other = root / "other"
            other.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                select_parent(selection, other)

    def test_checkpoint_trainer_and_epoch_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fold, checkpoint, selection = self.make_fixture(root)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state["trainer_name"] = "wrong"
            torch.save(state, checkpoint)
            payload = json.loads(selection.read_text())
            payload["selected"]["checkpoint"]["sha256"] = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            selection.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "trainer"):
                select_parent(selection, fold)


if __name__ == "__main__":
    unittest.main()
