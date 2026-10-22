import tempfile
import unittest
from pathlib import Path

from audit_focus_fold_epoch_sources import audit
import test_summarize_focus_epoch_routes as fixtures


class FocusFoldEpochSourceAuditTests(unittest.TestCase):
    def test_audits_one_identity_bound_fold_without_selecting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, runs = fixtures.FocusEpochRouteSummaryTests().build_fixture(root)
            result = audit(
                index,
                runs[1] / "checkpoint_epoch_0001.pt",
                runs[1] / "validation_predictions_epoch_0001.npz",
                folds=2,
                fold=1,
                seed=17,
                epoch=1,
            )
            self.assertEqual(result["fold"], 1)
            self.assertEqual(result["epoch"], 1)
            self.assertEqual(
                set(result["groups"]), {"mature", "weak_sources", "lung_cells"}
            )
            self.assertGreater(
                result["groups"]["lung_cells"]["case_balanced_dice"], 0.9
            )
            self.assertFalse(result["public_read"])
            self.assertFalse(result["hidden_read"])
            self.assertFalse(result["submission_created"])

    def test_wrong_epoch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, runs = fixtures.FocusEpochRouteSummaryTests().build_fixture(root)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                audit(
                    index,
                    runs[0] / "checkpoint_epoch_0001.pt",
                    runs[0] / "validation_predictions_epoch_0001.npz",
                    folds=2,
                    fold=0,
                    seed=17,
                    epoch=0,
                )


if __name__ == "__main__":
    unittest.main()
