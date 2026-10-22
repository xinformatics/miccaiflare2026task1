import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from select_recall_shortlist_public import select_public_shortlist


def identity(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class SelectRecallShortlistPublicTest(unittest.TestCase):
    def build_fixture(self, root):
        run = root / "run"
        children = root / "children"
        run.mkdir()
        children.mkdir()
        candidates = []
        promoted = {
            286: None,
            287: ("touching", 0.618, 0.520),
            288: ("deployed500", 0.621, 0.515),
        }
        roles = {
            286: "best_internal_dice",
            287: "best_large_component_recall_ge3000_within_dice_guard",
            288: "best_voxel_recall_within_dice_guard",
        }
        for epoch, chosen in promoted.items():
            checkpoint = run / f"checkpoint_epoch_{epoch:04d}.pt"
            checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
            candidates.append(
                {
                    "epoch": epoch,
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "selection_roles": [roles[epoch]],
                }
            )
            child = children / f"epoch_{epoch:04d}"
            child.mkdir()
            summary = child / "summary.tsv"
            rows = []
            for index, variant in enumerate(
                ("deployed500", "deployed1000", "touching")
            ):
                if chosen and variant == chosen[0]:
                    dsc, nsd, gate = chosen[1], chosen[2], "PROMOTE"
                else:
                    dsc, nsd, gate = 0.59 - index * 0.001, 0.50, "KEEP_ANCHOR"
                rows.append(
                    {
                        "variant": variant,
                        "DSC": str(dsc),
                        "NSD": str(nsd),
                        "gate": gate,
                        "predictions": str((child / variant).resolve()),
                    }
                )
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("variant", "DSC", "NSD", "gate", "predictions"),
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(rows)
            terminal = {
                "status": "COMPLETE",
                "decision": (
                    "QUALIFY_SUBMISSION_PREPARATION" if chosen else "KEEP_TOUCHCOMP"
                ),
                "selected_variant": chosen[0] if chosen else None,
                "checkpoint": identity(checkpoint),
                "summary": identity(summary),
                "fixed_public_DSC_gate": 0.6154316379500749,
                "fixed_public_NSD_floor": 0.5104225107769019,
                "public_read": True,
                "hidden_read": False,
                "submission_created": False,
                "uploaded": False,
            }
            (child / "TERMINAL.json").write_text(
                json.dumps(terminal), encoding="utf-8"
            )
        shortlist = root / "shortlist.json"
        shortlist.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "promotion_authority": "fixed_public_50_only",
                    "selection_objective": "case_balanced_dice_plus_ge3000mm3_and_voxel_recall",
                    "large_component_minimum_mm3": 3000,
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "fixed_public_DSC_gate": 0.6154316379500749,
                    "fixed_public_NSD_floor": 0.5104225107769019,
                    "public_read": False,
                    "hidden_read": False,
                    "submission_created": False,
                }
            ),
            encoding="utf-8",
        )
        return run, children, shortlist

    def test_selects_one_highest_dsc_public_qualified_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, children, shortlist = self.build_fixture(root)
            payload = select_public_shortlist(
                run,
                shortlist,
                children,
                children / "summary.tsv",
                children / "TERMINAL.json",
            )
            self.assertEqual(payload["decision"], "QUALIFY_SUBMISSION_PREPARATION")
            self.assertEqual(payload["selected_epoch"], 288)
            self.assertEqual(payload["selected_variant"], "deployed500")
            self.assertEqual(payload["screened_epochs"], [286, 287, 288])
            self.assertFalse(payload["hidden_read"])
            self.assertFalse(payload["submission_created"])
            with (children / "summary.tsv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 9)
            self.assertEqual(
                rows[0]["internal_selection_roles"], "best_internal_dice"
            )

    def test_rejects_deprecated_lesion_count_selection_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, children, shortlist = self.build_fixture(root)
            payload = json.loads(shortlist.read_text(encoding="utf-8"))
            payload["candidates"][0]["selection_roles"] = [
                "best_small_component_recall_within_dice_guard"
            ]
            shortlist.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deprecated selection role"):
                select_public_shortlist(
                    run,
                    shortlist,
                    children,
                    children / "summary.tsv",
                    children / "TERMINAL.json",
                )

    def test_rejects_checkpoint_mutated_after_internal_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, children, shortlist = self.build_fixture(root)
            (run / "checkpoint_epoch_0287.pt").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "shortlist/checkpoint hash"):
                select_public_shortlist(
                    run,
                    shortlist,
                    children,
                    children / "summary.tsv",
                    children / "TERMINAL.json",
                )

    def test_rejects_public_child_with_hidden_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, children, shortlist = self.build_fixture(root)
            terminal = children / "epoch_0286" / "TERMINAL.json"
            payload = json.loads(terminal.read_text(encoding="utf-8"))
            payload["hidden_read"] = True
            terminal.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "public-only provenance"):
                select_public_shortlist(
                    run,
                    shortlist,
                    children,
                    children / "summary.tsv",
                    children / "TERMINAL.json",
                )


if __name__ == "__main__":
    unittest.main()
