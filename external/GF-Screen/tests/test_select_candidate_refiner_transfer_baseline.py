import copy
import unittest

from select_candidate_refiner_transfer_baseline import select


def record(name, selection, dice, ap):
    return {
        "checkpoint": f"/{name}.pt",
        "evaluation_index_sha256": "a" * 64,
        "cross_index_transfer": True,
        "fold": 0,
        "seed": 17,
        "metrics": {
            "selection_score": selection,
            "case_balanced_segmentation_dice": dice,
            "average_precision": ap,
        },
        "public_read": False,
        "hidden_read": False,
    }


class TransferBaselineSelectorTests(unittest.TestCase):
    def setUp(self):
        self.records = {
            "composite": record("composite", 0.61, 0.50, 0.72),
            "segmentation": record("segmentation", 0.62, 0.53, 0.71),
            "classification": record("classification", 0.60, 0.47, 0.73),
        }

    def test_selects_by_case_balanced_composite_score(self):
        result = select(self.records)
        self.assertEqual(result["selected_family"], "segmentation")
        self.assertFalse(result["public_read"])

    def test_fails_closed_on_domain_or_coverage_drift(self):
        bad = copy.deepcopy(self.records)
        bad["classification"]["hidden_read"] = True
        with self.assertRaisesRegex(ValueError, "forbidden"):
            select(bad)
        with self.assertRaisesRegex(ValueError, "all three"):
            select({"composite": self.records["composite"]})

    def test_supports_an_explicit_untouched_fold(self):
        for value in self.records.values():
            value["fold"] = 1
        result = select(self.records, expected_fold=1)
        self.assertEqual(result["fold"], 1)
        with self.assertRaisesRegex(ValueError, "requested exact-fold"):
            select(self.records, expected_fold=0)


if __name__ == "__main__":
    unittest.main()
