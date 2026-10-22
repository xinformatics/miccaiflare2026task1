import copy
import unittest

from select_focus_xy4_pilot import select


def metrics(value, negative=0.20):
    return {
        "case_balanced_dice": value,
        "cohort_balanced_dice": value,
        "selection_score": value,
        "negative_nonempty_rate": negative,
    }


class FocusXY4PilotSelectionTests(unittest.TestCase):
    def fixture(self, weak_gain=0.002, lung_gain=0.002):
        route = {
            "fold": 4,
            "seed": 20260823,
            "index_sha256": "index",
            "selected": {
                "weak_sources": {
                    "epoch": 4,
                    "checkpoint_sha256": "weak-checkpoint",
                },
                "lung_cells": {
                    "epoch": 9,
                    "checkpoint_sha256": "lung-checkpoint",
                },
            },
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        xy = {
            "decision": "PROMOTE_TTA_TO_FIVEFOLD",
            "groups": {
                "weak_sources": {"epoch": 4, "tta_xy": metrics(0.30)},
                "lung_cells": {"epoch": 9, "tta_xy": metrics(0.40)},
            },
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        audits = [
            {
                "fold": 4,
                "seed": 20260823,
                "epoch": 4,
                "index_sha256": "index",
                "checkpoint_sha256": "weak-checkpoint",
                "groups": {"weak_sources": metrics(0.30 + weak_gain)},
                "public_read": False,
                "hidden_read": False,
                "submission_created": False,
            },
            {
                "fold": 4,
                "seed": 20260823,
                "epoch": 9,
                "index_sha256": "index",
                "checkpoint_sha256": "lung-checkpoint",
                "groups": {"lung_cells": metrics(0.40 + lung_gain)},
                "public_read": False,
                "hidden_read": False,
                "submission_created": False,
            },
        ]
        return xy, route, audits

    def test_promotes_groups_that_clear_frozen_gates(self):
        xy, route, audits = self.fixture()
        result = select(xy, route, audits)
        self.assertEqual(result["decision"], "PROMOTE_XY4_TO_FIVEFOLD")
        self.assertEqual(result["eligible_groups"], ["weak_sources", "lung_cells"])

    def test_keeps_three_view_when_gains_are_insufficient(self):
        xy, route, audits = self.fixture(0.0005, -0.001)
        result = select(xy, route, audits)
        self.assertEqual(result["decision"], "KEEP_XY3")
        self.assertEqual(result["eligible_groups"], [])

    def test_rejects_nonlocal_or_mismatched_evidence(self):
        xy, route, audits = self.fixture()
        nonlocal_audits = copy.deepcopy(audits)
        nonlocal_audits[0]["public_read"] = True
        with self.assertRaisesRegex(ValueError, "not local-only"):
            select(xy, route, nonlocal_audits)

        mismatched = copy.deepcopy(audits)
        mismatched[0]["checkpoint_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "checkpoint identity mismatch"):
            select(xy, route, mismatched)


if __name__ == "__main__":
    unittest.main()
