import json
import tempfile
import unittest
from pathlib import Path

from select_focus_fold_tta_pilot import select_tta


def metrics(dice, cohort, score, negative=0.2):
    return {
        "case_balanced_dice": dice,
        "cohort_balanced_dice": cohort,
        "selection_score": score,
        "negative_nonempty_rate": negative,
    }


class FocusFoldTTAPilotTests(unittest.TestCase):
    def fixture(self, root: Path, weak_gain: float, lung_gain: float):
        route = root / "routes.json"
        selected = {
            "mature": {
                "epoch": 3,
                "checkpoint_sha256": "mature",
                "metrics": metrics(0.5, 0.5, 0.49),
            },
            "weak_sources": {
                "epoch": 5,
                "checkpoint_sha256": "weak",
                "metrics": metrics(0.3, 0.3, 0.29),
            },
            "lung_cells": {
                "epoch": 7,
                "checkpoint_sha256": "lung",
                "metrics": metrics(0.4, 0.4, 0.39),
            },
        }
        route.write_text(
            json.dumps(
                {
                    "fold": 4,
                    "seed": 9,
                    "index_sha256": "index",
                    "selected": selected,
                    "public_read": False,
                    "hidden_read": False,
                    "submission_created": False,
                }
            ),
            encoding="utf-8",
        )
        audits = []
        for epoch, group, digest, gain, base in (
            (5, "weak_sources", "weak", weak_gain, 0.3),
            (7, "lung_cells", "lung", lung_gain, 0.4),
        ):
            path = root / f"tta{epoch}.json"
            path.write_text(
                json.dumps(
                    {
                        "fold": 4,
                        "seed": 9,
                        "epoch": epoch,
                        "index_sha256": "index",
                        "checkpoint_sha256": digest,
                        "groups": {
                            group: metrics(
                                base + gain,
                                base + gain,
                                base - 0.01 + gain,
                            )
                        },
                        "public_read": False,
                        "hidden_read": False,
                        "submission_created": False,
                    }
                ),
                encoding="utf-8",
            )
            audits.append(path)
        return route, audits

    def test_promotes_only_groups_clearing_paired_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            route, audits = self.fixture(Path(temporary), 0.01, -0.01)
            result = select_tta(route, audits)
            self.assertEqual(result["decision"], "PROMOTE_TTA_TO_FIVEFOLD")
            self.assertEqual(result["eligible_groups"], ["weak_sources"])

    def test_keeps_notta_when_neither_group_qualifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            route, audits = self.fixture(Path(temporary), 0.0, -0.01)
            result = select_tta(route, audits)
            self.assertEqual(result["decision"], "KEEP_NOTTA")
            self.assertEqual(result["eligible_groups"], [])


if __name__ == "__main__":
    unittest.main()
