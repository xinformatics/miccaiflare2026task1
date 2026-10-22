import unittest

from select_candidate_refiner_source_checkpoint import choose


def record(epoch, dice, threshold=0.5, fold=0):
    return {
        "fold": fold,
        "folds": 5,
        "seed": 20260823,
        "index_sha256": "a" * 64,
        "epoch": epoch,
        "source_segmentation_groups": {
            "mature": {"dice": dice, "threshold": threshold}
        },
    }


class SourceCheckpointSelectionTests(unittest.TestCase):
    def test_chooses_highest_group_dice_not_latest_epoch(self):
        selected = choose([record(13, 0.56), record(16, 0.54)], "mature")
        self.assertEqual(selected["epoch"], 13)

    def test_tie_prefers_threshold_nearest_half_then_earlier_epoch(self):
        selected = choose(
            [record(10, 0.5, 0.8), record(13, 0.5, 0.5), record(16, 0.5, 0.5)],
            "mature",
        )
        self.assertEqual(selected["epoch"], 13)

    def test_rejects_mixed_fold_identity(self):
        with self.assertRaisesRegex(ValueError, "mix fold/index identities"):
            choose([record(13, 0.5, fold=0), record(13, 0.6, fold=1)], "mature")

    def test_selects_a_single_source_without_fabricating_a_pool(self):
        first, second = record(13, 0.1), record(16, 0.1)
        first["source_segmentation"] = {"currentgf": {"dice": 0.59, "threshold": 0.8}}
        second["source_segmentation"] = {"currentgf": {"dice": 0.57, "threshold": 0.8}}
        self.assertEqual(choose([first, second], "currentgf")["epoch"], 13)


if __name__ == "__main__":
    unittest.main()
