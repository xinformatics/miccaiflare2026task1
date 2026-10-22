import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_recall_shortlist_save_contract import build_save_contract


def identity(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class ValidateRecallShortlistSaveContractTest(unittest.TestCase):
    def fixture(self, root, qualified=True):
        run = root / "run"
        children = root / "children"
        run.mkdir()
        children.mkdir()
        epoch, variant = 301, "touching"
        checkpoint = run / f"checkpoint_epoch_{epoch:04d}.pt"
        checkpoint.write_bytes(b"checkpoint")
        candidate = {
            "epoch": epoch,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "selection_roles": [
                "best_large_component_recall_ge3000_within_dice_guard"
            ],
        }
        shortlist = root / "shortlist.json"
        shortlist.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "promotion_authority": "fixed_public_50_only",
                    "selection_objective": "case_balanced_dice_plus_ge3000mm3_and_voxel_recall",
                    "large_component_minimum_mm3": 3000,
                    "candidate_count": 1,
                    "candidates": [candidate],
                    "public_read": False,
                    "hidden_read": False,
                    "submission_created": False,
                }
            ),
            encoding="utf-8",
        )
        child_dir = children / f"epoch_{epoch:04d}"
        child_dir.mkdir()
        child_summary = child_dir / "summary.tsv"
        child_summary.write_text("unused\n", encoding="utf-8")
        child_terminal = child_dir / "TERMINAL.json"
        child_terminal.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "decision": "QUALIFY_SUBMISSION_PREPARATION",
                    "selected_variant": variant,
                    "checkpoint": identity(checkpoint),
                    "hidden_read": False,
                    "submission_created": False,
                    "uploaded": False,
                }
            ),
            encoding="utf-8",
        )
        summary = root / "summary.tsv"
        with summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("checkpoint_epoch", "variant", "DSC", "NSD", "gate", "predictions"),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "checkpoint_epoch": epoch,
                    "variant": variant,
                    "DSC": 0.620,
                    "NSD": 0.520,
                    "gate": "PROMOTE",
                    "predictions": str((child_dir / variant).resolve()),
                }
            )
        public_terminal = root / "TERMINAL.json"
        public = {
            "status": "COMPLETE",
            "decision": "QUALIFY_SUBMISSION_PREPARATION" if qualified else "KEEP_TOUCHCOMP",
            "internal_shortlist": identity(shortlist),
            "summary": identity(summary),
            "screened_epochs": [epoch],
            "selected_epoch": epoch if qualified else None,
            "selected_variant": variant if qualified else None,
            "checkpoint": identity(checkpoint) if qualified else None,
            "selected_public_terminal": identity(child_terminal) if qualified else None,
            "selected_predictions": str((child_dir / variant).resolve()) if qualified else None,
            "selected_fixed_public_DSC": 0.620 if qualified else None,
            "selected_fixed_public_NSD": 0.520 if qualified else None,
            "fixed_public_DSC_gate": 0.6154316379500749,
            "fixed_public_NSD_floor": 0.5104225107769019,
            "public_read": True,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
        }
        public_terminal.write_text(json.dumps(public), encoding="utf-8")
        return run, children, shortlist, summary, public_terminal

    def test_qualified_contract_is_hash_bound_and_submission_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, children, shortlist, summary, terminal = self.fixture(root)
            payload = build_save_contract(
                terminal, shortlist, summary, run, children, root / "contract.json"
            )
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["selected_epoch"], 301)
            self.assertEqual(payload["selected_variant"], "touching")
            self.assertEqual(payload["minimum_component_mm3"], 1000)
            self.assertFalse(payload["hidden_read"])

    def test_nonqualified_contract_skips_without_hidden_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, children, shortlist, summary, terminal = self.fixture(
                root, qualified=False
            )
            payload = build_save_contract(
                terminal, shortlist, summary, run, children, root / "contract.json"
            )
            self.assertEqual(payload["status"], "SKIP")
            self.assertEqual(payload["decision"], "KEEP_TOUCHCOMP")
            self.assertFalse(payload["hidden_read"])

    def test_rejects_tampered_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, children, shortlist, summary, terminal = self.fixture(root)
            (run / "checkpoint_epoch_0301.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "shortlist/checkpoint"):
                build_save_contract(
                    terminal,
                    shortlist,
                    summary,
                    run,
                    children,
                    root / "contract.json",
                )

    def test_rejects_deprecated_lesion_count_selection_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, children, shortlist, summary, terminal = self.fixture(root)
            shortlist_payload = json.loads(shortlist.read_text(encoding="utf-8"))
            shortlist_payload["candidates"][0]["selection_roles"] = [
                "best_small_component_recall_within_dice_guard"
            ]
            shortlist.write_text(json.dumps(shortlist_payload), encoding="utf-8")
            public_payload = json.loads(terminal.read_text(encoding="utf-8"))
            public_payload["internal_shortlist"] = identity(shortlist)
            terminal.write_text(json.dumps(public_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection role mismatch"):
                build_save_contract(
                    terminal,
                    shortlist,
                    summary,
                    run,
                    children,
                    root / "contract.json",
                )


if __name__ == "__main__":
    unittest.main()
