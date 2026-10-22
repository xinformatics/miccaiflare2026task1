import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from select_long_focus_pu_trajectory import select_trajectory


def identity(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class SelectLongFocusTrajectoryTest(unittest.TestCase):
    def build_fixture(self, root):
        run_dir = root / "run"
        trajectory = root / "trajectory"
        run_dir.mkdir()
        trajectory.mkdir()
        promoted = {
            293: None,
            303: ("deployed500", 0.620, 0.520),
            313: ("touching", 0.618, 0.530),
        }
        for epoch, chosen in promoted.items():
            checkpoint = run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
            child = trajectory / f"epoch_{epoch:04d}"
            child.mkdir()
            summary = child / "summary.tsv"
            rows = []
            for index, variant in enumerate(
                ("deployed500", "deployed1000", "touching")
            ):
                if chosen and variant == chosen[0]:
                    dsc, nsd, gate = chosen[1], chosen[2], "PROMOTE"
                else:
                    dsc, nsd, gate = 0.59 - index * 0.001, 0.51, "KEEP"
                rows.append({
                    "variant": variant,
                    "DSC": str(dsc),
                    "NSD": str(nsd),
                    "gate": gate,
                    "predictions": str((child / variant).resolve()),
                })
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
                "hidden_read": False,
                "submission_created": False,
                "uploaded": False,
            }
            (child / "TERMINAL.json").write_text(
                json.dumps(terminal), encoding="utf-8"
            )
        return run_dir, trajectory

    def test_selects_exactly_one_best_qualified_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, trajectory = self.build_fixture(root)
            summary = trajectory / "summary.tsv"
            terminal = trajectory / "TERMINAL.json"
            diagnostic = trajectory / "best_dsc_lesion_error.json"
            diagnostic.write_text(
                json.dumps(
                    {
                        "cases": 50,
                        "configuration": {
                            "predictions": str(
                                (trajectory / "epoch_0303" / "deployed500").resolve()
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            payload = select_trajectory(
                run_dir, trajectory, summary, terminal, diagnostic
            )
            self.assertEqual(payload["decision"], "QUALIFY_SUBMISSION_PREPARATION")
            self.assertEqual(payload["selected_epoch"], 303)
            self.assertEqual(payload["selected_variant"], "deployed500")
            self.assertEqual(payload["predeclared_epochs"], [293, 303, 313])
            self.assertFalse(payload["hidden_read"])
            self.assertFalse(payload["submission_created"])
            self.assertFalse(payload["uploaded"])
            self.assertEqual(
                payload["best_dsc_diagnostic"]["selection_role"],
                "diagnostic_only_not_used_for_promotion",
            )
            self.assertEqual(
                payload["best_dsc_diagnostic"]["lesion_error_summary"],
                identity(diagnostic),
            )
            with summary.open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle, delimiter="\t"))), 9)

    def test_rejects_tampered_child_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, trajectory = self.build_fixture(root)
            child_summary = trajectory / "epoch_0303" / "summary.tsv"
            child_summary.write_text(
                child_summary.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "summary identity mismatch"):
                select_trajectory(
                    run_dir,
                    trajectory,
                    trajectory / "summary.tsv",
                    trajectory / "TERMINAL.json",
                )

    def test_rejects_lesion_error_for_different_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, trajectory = self.build_fixture(root)
            diagnostic = trajectory / "wrong.json"
            diagnostic.write_text(
                json.dumps(
                    {
                        "cases": 50,
                        "configuration": {"predictions": str(root / "wrong")},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "diagnostic identity mismatch"):
                select_trajectory(
                    run_dir,
                    trajectory,
                    trajectory / "summary.tsv",
                    trajectory / "TERMINAL.json",
                    diagnostic,
                )


if __name__ == "__main__":
    unittest.main()
