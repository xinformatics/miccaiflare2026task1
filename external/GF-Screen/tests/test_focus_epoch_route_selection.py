import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from focus_epoch_route_selection import ROUTE_GROUPS, route_records


class FocusEpochRouteSelectionTests(unittest.TestCase):
    def fixture(self, root: Path):
        sources = {
            "epoch105": "mature",
            "currentgf": "mature",
            "autopet_weak": "weak_sources",
            "lndb_weak": "weak_sources",
            "lung_cell0": "lung_cells",
        }
        folds = []
        for fold in range(5):
            selected = {}
            for group in ROUTE_GROUPS:
                checkpoint = root / f"fold{fold}_{group}.pt"
                checkpoint.write_bytes(f"{fold}:{group}".encode())
                selected[group] = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                }
            folds.append({"fold": fold, "selected": selected})
        summary = root / "routes.json"
        summary.write_text(
            json.dumps(
                {
                    "folds": 5,
                    "route_groups": sorted(ROUTE_GROUPS),
                    "training_recipe": {"input_mode": "soft_lung_seed"},
                    "source_to_route": sources,
                    "fold_records": folds,
                    "public_read": False,
                    "hidden_read": False,
                }
            )
        )
        decision = root / "decision.json"
        decision.write_text(
            json.dumps(
                {
                    "decision": "QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME",
                    "candidate": str(summary.resolve()),
                    "candidate_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                    "public_read": False,
                    "hidden_read": False,
                }
            )
        )
        return summary, decision, sources

    def test_emits_complete_hashed_fivefold_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary, decision, sources = self.fixture(Path(temporary))
            records = route_records(summary, decision, set(sources))
            self.assertEqual(records[0], ("INPUT", "-", "-", "soft_lung_seed"))
            self.assertEqual(sum(row[0] == "ROUTE" for row in records), 15)
            self.assertEqual(sum(row[0] == "SOURCE" for row in records), 5)

    def test_rejects_stale_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary, decision, sources = self.fixture(Path(temporary))
            payload = json.loads(summary.read_text())
            Path(payload["fold_records"][2]["selected"]["mature"]["checkpoint"]).write_bytes(
                b"changed"
            )
            with self.assertRaisesRegex(ValueError, "checkpoint checksum mismatch"):
                route_records(summary, decision, set(sources))

    def test_rejects_unqualified_or_incomplete_source_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary, decision, sources = self.fixture(Path(temporary))
            payload = json.loads(decision.read_text())
            payload["decision"] = "KEEP_GLOBAL_FOCUS_EPOCHS"
            decision.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "did not pass"):
                route_records(summary, decision, set(sources))

            summary, decision, sources = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "source route coverage mismatch"):
                route_records(summary, decision, set(sources) | {"lung_cell1"})


if __name__ == "__main__":
    unittest.main()
