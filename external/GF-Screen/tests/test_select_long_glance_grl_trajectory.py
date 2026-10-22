import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from select_long_glance_grl_trajectory import select_trajectory, threshold_tag


def identity(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class SelectLongGlanceTrajectoryTest(unittest.TestCase):
    def build_fixture(self, root):
        run_dir = root / "run"
        trajectory = root / "trajectory"
        run_dir.mkdir()
        trajectory.mkdir()
        source_checkpoint = root / "source.pt"
        source_checkpoint.write_bytes(b"source")
        source = root / "SOURCE.json"
        source.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "source_epoch": 283,
                    "source_checkpoint": identity(source_checkpoint),
                }
            ),
            encoding="utf-8",
        )
        for epoch in (293, 303, 313):
            checkpoint = run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
            for threshold in (0.10, 0.20, 0.35):
                child = trajectory / f"epoch_{epoch:04d}" / threshold_tag(threshold)
                child.mkdir(parents=True)
                summary = child / "summary.tsv"
                rows = []
                for variant in ("deployed500", "deployed1000", "touching"):
                    promote = epoch == 303 and threshold == 0.20 and variant == "touching"
                    rows.append(
                        {
                            "variant": variant,
                            "DSC": "0.625" if promote else "0.600",
                            "NSD": "0.535" if promote else "0.510",
                            "gate": "PROMOTE" if promote else "KEEP",
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
                        "QUALIFY_SUBMISSION_PREPARATION" if any(
                            row["gate"] == "PROMOTE" for row in rows
                        ) else "KEEP_TOUCHCOMP"
                    ),
                    "glance_threshold": threshold,
                    "checkpoint": identity(checkpoint),
                    "summary": identity(summary),
                    "hidden_read": False,
                    "submission_created": False,
                    "uploaded": False,
                }
                (child / "TERMINAL.json").write_text(
                    json.dumps(terminal), encoding="utf-8"
                )
        return run_dir, trajectory, source

    def test_selects_best_predeclared_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, trajectory, source = self.build_fixture(root)
            diagnostic = trajectory / "best_dsc_lesion_error.json"
            diagnostic.write_text(
                json.dumps(
                    {
                        "cases": 50,
                        "configuration": {
                            "predictions": str(
                                (
                                    trajectory
                                    / "epoch_0303"
                                    / "g0p20"
                                    / "touching"
                                ).resolve()
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            payload = select_trajectory(
                run_dir,
                source,
                trajectory,
                trajectory / "summary.tsv",
                trajectory / "TERMINAL.json",
                diagnostic,
            )
            self.assertEqual(payload["decision"], "QUALIFY_SUBMISSION_PREPARATION")
            self.assertEqual(payload["selected_epoch"], 303)
            self.assertEqual(payload["selected_glance_threshold"], 0.20)
            self.assertEqual(payload["selected_variant"], "touching")
            self.assertEqual(payload["predeclared_offsets"], [10, 20, 30])
            self.assertEqual(len(payload["children"]), 9)
            self.assertEqual(
                payload["best_dsc_diagnostic"]["lesion_error_summary"],
                identity(diagnostic),
            )
            with (trajectory / "summary.tsv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle, delimiter="\t"))), 27)

    def test_rejects_threshold_identity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, trajectory, source = self.build_fixture(root)
            terminal = trajectory / "epoch_0303" / "g0p20" / "TERMINAL.json"
            payload = json.loads(terminal.read_text(encoding="utf-8"))
            payload["glance_threshold"] = 0.35
            terminal.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Glance threshold mismatch"):
                select_trajectory(
                    run_dir,
                    source,
                    trajectory,
                    trajectory / "summary.tsv",
                    trajectory / "TERMINAL.json",
                )

    def test_rejects_lesion_error_for_different_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, trajectory, source = self.build_fixture(root)
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
                    source,
                    trajectory,
                    trajectory / "summary.tsv",
                    trajectory / "TERMINAL.json",
                    diagnostic,
                )

    def test_rejects_tampered_child_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, trajectory, source = self.build_fixture(root)
            summary = trajectory / "epoch_0293" / "g0p10" / "summary.tsv"
            summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary identity mismatch"):
                select_trajectory(
                    run_dir,
                    source,
                    trajectory,
                    trajectory / "summary.tsv",
                    trajectory / "TERMINAL.json",
                )


if __name__ == "__main__":
    unittest.main()
