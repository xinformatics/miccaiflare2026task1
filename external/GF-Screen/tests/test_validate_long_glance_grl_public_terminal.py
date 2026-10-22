import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from select_long_glance_grl_trajectory import select_trajectory, threshold_tag
from validate_long_glance_grl_public_terminal import validate_handoff


def identity(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class ValidateLongGlancePublicTerminalTest(unittest.TestCase):
    def build_fixture(self, root, promote=True):
        run_dir = root / "run"
        trajectory = root / "trajectory"
        run_dir.mkdir()
        trajectory.mkdir()
        source_checkpoint = root / "checkpoint_epoch_0283.pt"
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
        recipe = root / "TRAINING_RECIPE.json"
        recipe.write_text(
            json.dumps(
                {
                    "start_epoch": 283,
                    "max_epochs": 313,
                    "predeclared_checkpoint_offsets": [10, 20, 30],
                    "predeclared_glance_thresholds": [0.10, 0.20, 0.35],
                    "public_labels_used_for_gradient": False,
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
                    selected = (
                        promote
                        and epoch == 303
                        and threshold == 0.20
                        and variant == "touching"
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "DSC": "0.625" if selected else "0.600",
                            "NSD": "0.535" if selected else "0.510",
                            "gate": "PROMOTE" if selected else "KEEP",
                        }
                    )
                with summary.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=("variant", "DSC", "NSD", "gate"),
                        delimiter="\t",
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                chosen = next((row for row in rows if row["gate"] == "PROMOTE"), None)
                terminal = {
                    "status": "COMPLETE",
                    "decision": "QUALIFY_SUBMISSION_PREPARATION" if chosen else "KEEP_TOUCHCOMP",
                    "selected_variant": chosen["variant"] if chosen else None,
                    "glance_threshold": threshold,
                    "checkpoint": identity(checkpoint),
                    "summary": identity(summary),
                    "hidden_read": False,
                    "submission_created": False,
                    "uploaded": False,
                }
                (child / "TERMINAL.json").write_text(json.dumps(terminal), encoding="utf-8")
        summary = trajectory / "summary.tsv"
        public_terminal = trajectory / "TERMINAL.json"
        public_payload = select_trajectory(
            run_dir, source, trajectory, summary, public_terminal
        )
        controller = root / "CONTROLLER.json"
        controller.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "decision": public_payload["decision"],
                    "source": identity(source),
                    "training_recipe": identity(recipe),
                    "public_terminal": identity(public_terminal),
                    "hidden_read": False,
                    "submission_created": False,
                    "uploaded": False,
                }
            ),
            encoding="utf-8",
        )
        return controller, public_terminal, source, recipe, summary, run_dir, trajectory

    def validate(self, fixture):
        return validate_handoff(*fixture)

    def test_accepts_hash_bound_qualified_state(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.validate(self.build_fixture(Path(directory)))
            self.assertEqual(payload["decision"], "QUALIFY_SUBMISSION_PREPARATION")
            self.assertEqual(payload["selected_epoch"], 303)
            self.assertEqual(payload["selected_glance_threshold"], 0.20)
            self.assertEqual(payload["selected_variant"], "touching")
            self.assertFalse(payload["hidden_read"])

    def test_accepts_keep_only_when_no_state_is_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.validate(self.build_fixture(Path(directory), promote=False))
            self.assertEqual(payload["decision"], "KEEP_TOUCHCOMP")
            self.assertFalse(payload["submission_created"])

    def test_rejects_changed_recipe_after_controller_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_fixture(Path(directory))
            recipe = fixture[3]
            recipe.write_text(recipe.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "controller recipe identity mismatch"):
                self.validate(fixture)

    def test_rejects_controller_public_decision_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_fixture(Path(directory))
            controller = fixture[0]
            payload = json.loads(controller.read_text(encoding="utf-8"))
            payload["decision"] = "KEEP_TOUCHCOMP"
            controller.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "controller/public decision mismatch"):
                self.validate(fixture)


if __name__ == "__main__":
    unittest.main()
