import unittest

from select_focus_b32_source_hybrid import decide


SOURCES = {
    "mature": ["epoch105", "currentgf"],
    "weak_sources": ["autopet_weak", "lndb_weak"],
    "lung_cells": [f"lung_cell{index}" for index in range(8)],
}


def summaries(gains):
    focus_groups, b32_groups = {}, {}
    for group, sources in SOURCES.items():
        case_gain, cohort_gain = gains[group]
        b32_groups[group] = {
            "sources": sources,
            "threshold": 0.8,
            "dice": 0.4,
            "cohort_balanced_dice": 0.4,
        }
        focus_groups[group] = {
            "sources": sources,
            "threshold": 0.6,
            "case_balanced_dice": 0.4 + case_gain,
            "cohort_balanced_dice": 0.4 + cohort_gain,
            "negative_nonempty_rate": 0.9,
        }
    focus = {
        "folds": 5,
        "index_sha256": "same",
        "input_mode": "soft_lung_seed",
        "negative_nonempty_rate": 0.9,
        "source_segmentation_groups": focus_groups,
    }
    b32 = {
        "index_sha256": "same",
        "coverage": {"cases": 707, "every_patch_exactly_once": True},
        "source_segmentation_groups": b32_groups,
    }
    return focus, b32


class FocusB32SourceHybridTests(unittest.TestCase):
    def test_qualifies_a_true_mixed_capacity_route(self):
        focus, b32 = summaries(
            {
                "mature": (-0.01, -0.01),
                "weak_sources": (0.02, 0.01),
                "lung_cells": (0.03, 0.02),
            }
        )
        result = decide(focus, b32)
        self.assertEqual(result["decision"], "QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME")
        self.assertEqual(result["b32_groups"], ["mature"])
        self.assertEqual(result["focus_groups"], ["lung_cells", "weak_sources"])
        self.assertEqual(result["source_to_route"]["currentgf"], "b32")
        self.assertEqual(result["source_to_route"]["lung_cell0"], "focus")

    def test_single_metric_gain_is_not_enough(self):
        focus, b32 = summaries(
            {
                "mature": (0.02, 0.0),
                "weak_sources": (0.0, 0.02),
                "lung_cells": (0.0, 0.0),
            }
        )
        result = decide(focus, b32)
        self.assertEqual(result["decision"], "KEEP_B32")

    def test_all_focus_uses_existing_global_path(self):
        focus, b32 = summaries({group: (0.02, 0.02) for group in SOURCES})
        result = decide(focus, b32)
        self.assertEqual(result["decision"], "USE_GLOBAL_FOCUS_PATH")
        self.assertFalse(result["passed"])

    def test_exact_707_coverage_is_required(self):
        focus, b32 = summaries({group: (0.02, 0.02) for group in SOURCES})
        b32["coverage"]["cases"] = 706
        with self.assertRaisesRegex(ValueError, "707-case"):
            decide(focus, b32)

    def test_source_specific_negative_activation_fails_closed(self):
        focus, b32 = summaries(
            {
                "mature": (-0.01, -0.01),
                "weak_sources": (0.02, 0.02),
                "lung_cells": (0.02, 0.02),
            }
        )
        focus["source_segmentation_groups"]["weak_sources"][
            "negative_nonempty_rate"
        ] = 0.96
        result = decide(focus, b32)
        self.assertEqual(result["groups"]["weak_sources"]["selected_family"], "b32")
        self.assertEqual(result["groups"]["lung_cells"]["selected_family"], "focus")


if __name__ == "__main__":
    unittest.main()
