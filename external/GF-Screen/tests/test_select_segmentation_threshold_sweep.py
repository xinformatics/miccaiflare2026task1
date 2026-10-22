import unittest

from select_segmentation_threshold_sweep import select


def row(threshold, dsc, nsd, classification=0.9, mode="refined"):
    return {
        "variant": f"s{threshold}",
        "classification_threshold": str(classification),
        "segmentation_threshold": str(threshold),
        "deployment_mode": mode,
        "DSC": str(dsc),
        "NSD": str(nsd),
        "metrics": f"s{threshold}.csv",
    }


class SelectSegmentationThresholdSweepTests(unittest.TestCase):
    def test_selects_dsc_then_nsd_from_predeclared_grid(self):
        result = select(
            [row(0.8, 0.60, 0.55), row(0.85, 0.62, 0.54), row(0.9, 0.62, 0.57)],
            0.8,
        )
        self.assertEqual(result["predeclared_thresholds"], [0.8, 0.85, 0.9])
        self.assertEqual(result["selected_segmentation_threshold"], 0.9)
        self.assertFalse(result["public_read"])

    def test_rejects_unplanned_or_mixed_grid(self):
        with self.assertRaises(ValueError):
            select(
                [row(0.75, 0.6, 0.5), row(0.8, 0.6, 0.5), row(0.85, 0.6, 0.5)],
                0.8,
            )
        with self.assertRaises(ValueError):
            select(
                [row(0.8, 0.6, 0.5), row(0.85, 0.6, 0.5, 0.85), row(0.9, 0.6, 0.5)],
                0.8,
            )


if __name__ == "__main__":
    unittest.main()
