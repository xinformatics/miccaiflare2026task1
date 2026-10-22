import unittest

import numpy as np

from audit_fold_size_checkpoint_routing import select, size_bin


class FoldSizeCheckpointRoutingTests(unittest.TestCase):
    def test_size_bins_are_fixed_in_physical_volume(self):
        self.assertEqual(size_bin({"seed_mm3": "999", "patch_id": "a"}), "small_lt1000")
        self.assertEqual(
            size_bin({"seed_mm3": "1000", "patch_id": "b"}),
            "medium_1000to10000",
        )
        self.assertEqual(
            size_bin({"seed_mm3": "10000", "patch_id": "c"}),
            "large_ge10000",
        )
        with self.assertRaisesRegex(ValueError, "invalid seed_mm3"):
            size_bin({"seed_mm3": "0", "patch_id": "bad"})

    def test_conservative_gate_can_select_a_supported_size_checkpoint(self):
        rows = []
        ids = []
        epoch1 = []
        epoch2 = []
        # Source-only epoch 1 wins overall because small cases are twice as
        # numerous, while the independently supported large stratum prefers 2.
        for index in range(9):
            large = index < 3
            patch_id = f"p{index}"
            rows.append(
                {
                    "patch_id": patch_id,
                    "case": f"case{index}",
                    "cohort": "cohort",
                    "label": "1",
                    "source_name": "currentgf",
                    "seed_mm3": "20000" if large else "500",
                }
            )
            ids.append(patch_id)
            epoch1.append([0.4 if large else 0.8])
            epoch2.append([0.9 if large else 0.5])
        predictions = {
            1: {
                "positive_patch_id": np.asarray(ids),
                "segmentation_thresholds": np.asarray([0.5]),
                "segmentation_dice_grid": np.asarray(epoch1),
            },
            2: {
                "positive_patch_id": np.asarray(ids),
                "segmentation_thresholds": np.asarray([0.5]),
                "segmentation_dice_grid": np.asarray(epoch2),
            },
        }
        report = select(rows, predictions, minimum_cases=3, minimum_gain=0.01)
        self.assertEqual(report["source_only"]["selected"]["currentgf"]["epoch"], 1)
        self.assertGreater(
            report["conservative_source_size"]["case_dice_gain"], 0.15
        )
        large = next(
            row
            for row in report["conservative_source_size"]["strata"]
            if row["size_bin"] == "large_ge10000"
        )
        self.assertTrue(large["accepted"])
        self.assertEqual(large["selected_epoch"], 2)

    def test_prediction_coverage_mismatch_fails_closed(self):
        rows = [
            {
                "patch_id": "p0",
                "case": "case0",
                "cohort": "cohort",
                "label": "1",
                "source_name": "currentgf",
                "seed_mm3": "100",
            }
        ]
        predictions = {
            1: {
                "positive_patch_id": np.asarray(["wrong"]),
                "segmentation_thresholds": np.asarray([0.5]),
                "segmentation_dice_grid": np.asarray([[0.5]]),
            }
        }
        with self.assertRaisesRegex(ValueError, "positive coverage mismatch"):
            select(rows, predictions, minimum_cases=1, minimum_gain=0)


if __name__ == "__main__":
    unittest.main()
