import unittest

from select_candidate_refiner_discovery_specialist import select_specialist


def metrics(discovery: float, weak: float, cells: float):
    return {
        "discovery": {"routed_segmentation_dice": discovery},
        "weak": {"routed_segmentation_dice": weak},
        "lung_cells": {"routed_segmentation_dice": cells},
    }


class DiscoverySpecialistSelectionTests(unittest.TestCase):
    def test_promotes_material_gain_with_both_group_nonregressions(self):
        decision, result = select_specialist(
            metrics(0.20, 0.18, 0.22),
            metrics(0.22, 0.179, 0.24),
            minimum_combined_gain=0.01,
            maximum_group_drop=0.005,
        )
        self.assertEqual(decision, "EXPAND_DISCOVERY_SPECIALIST")
        self.assertTrue(all(result["checks"].values()))

    def test_rejects_gain_caused_by_collapsing_one_discovery_family(self):
        decision, result = select_specialist(
            metrics(0.20, 0.18, 0.22),
            metrics(0.22, 0.16, 0.25),
            minimum_combined_gain=0.01,
            maximum_group_drop=0.005,
        )
        self.assertEqual(decision, "KEEP_PRIMARY")
        self.assertFalse(result["checks"]["weak_source_nonregression"])


if __name__ == "__main__":
    unittest.main()
