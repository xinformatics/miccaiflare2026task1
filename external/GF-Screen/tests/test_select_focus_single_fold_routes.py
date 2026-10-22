import json
import tempfile
import unittest
from pathlib import Path

from select_focus_single_fold_routes import GROUPS, select_routes


class FocusSingleFoldRouteTests(unittest.TestCase):
    def test_selects_each_group_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audits = []
            for epoch in range(3):
                path = root / f"epoch{epoch}.json"
                groups = {}
                for offset, group in enumerate(GROUPS):
                    score = 0.1 + epoch * 0.01
                    if group == "weak_sources" and epoch == 1:
                        score = 0.9
                    if group == "lung_cells" and epoch == 2:
                        score = 0.8
                    groups[group] = {
                        "selection_score": score,
                        "case_balanced_dice": score + 0.01,
                        "cohort_balanced_dice": score + 0.02,
                        "negative_nonempty_rate": 0.1,
                    }
                payload = {
                    "fold": 4,
                    "seed": 7,
                    "epoch": epoch,
                    "index_sha256": "index",
                    "checkpoint": str(root / f"checkpoint{epoch}.pt"),
                    "checkpoint_sha256": f"checkpoint{epoch}",
                    "prediction": str(root / f"prediction{epoch}.npz"),
                    "prediction_sha256": f"prediction{epoch}",
                    "groups": groups,
                    "public_read": False,
                    "hidden_read": False,
                    "submission_created": False,
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                audits.append(path)
            result = select_routes(audits, 3)
            self.assertEqual(result["selected"]["weak_sources"]["epoch"], 1)
            self.assertEqual(result["selected"]["lung_cells"]["epoch"], 2)
            self.assertEqual(result["selected"]["mature"]["epoch"], 2)

    def test_rejects_incomplete_epoch_coverage(self):
        with self.assertRaisesRegex(ValueError, "every expected epoch"):
            select_routes([], 3)


if __name__ == "__main__":
    unittest.main()
