import unittest

from select_candidate_refiner_partial_source_routes import (
    EXPECTED_SOURCES,
    select_routes,
)


def report(dice, cohort, cases=20):
    return {
        "dice": dice,
        "cohort_balanced_dice": cohort,
        "positive_cases": cases,
    }


def summary(*, ap=0.8, index="a" * 64, coverage_cases=707, gains=None):
    gains = gains or {}
    source = {
        "epoch105": report(0.50, 0.49),
        "currentgf": report(0.51, 0.50),
        "autopet_weak": report(0.30, 0.29),
        "lndb_weak": report(0.35, 0.34),
    }
    lung = report(0.40, 0.39, 80)
    for name, (case_gain, cohort_gain) in gains.items():
        target = lung if name == "lung" else source[name]
        target["dice"] += case_gain
        target["cohort_balanced_dice"] += cohort_gain
    return {
        "index_sha256": index,
        "coverage": {
            "cases": coverage_cases,
            "patches": 1000,
            "positive": 400,
            "negative": 600,
            "every_patch_exactly_once": True,
        },
        "classification": {"average_precision": ap},
        "source_segmentation": source,
        "source_segmentation_groups": {
            "lung_cells": {**lung, "sources": [f"lung_cell{i}" for i in range(8)]}
        },
    }


class PartialSourceRouteSelectionTests(unittest.TestCase):
    def test_selects_only_independently_improving_sources(self):
        baseline = summary()
        candidate = summary(
            gains={
                "epoch105": (0.003, 0.002),
                "currentgf": (-0.01, -0.01),
                "autopet_weak": (0.03, 0.02),
                "lndb_weak": (0.001, 0.01),
                "lung": (0.004, 0.003),
            }
        )
        decision, selection, checks = select_routes(baseline, candidate)
        self.assertEqual(decision, "QUALIFY_GROUP_SELECTED_SOURCE_ROUTE")
        self.assertTrue(all(checks.values()))
        self.assertEqual(selection["selected_units"], ["epoch105", "autopet", "lung"])
        routes = selection["source_to_route"]
        self.assertEqual(routes["epoch105"], "epoch105")
        self.assertEqual(routes["currentgf"], "global")
        self.assertEqual(routes["autopet_weak"], "autopet")
        self.assertEqual(routes["lndb_weak"], "global")
        self.assertTrue(all(routes[f"lung_cell{i}"] == "lung" for i in range(8)))
        self.assertEqual(set(routes), EXPECTED_SOURCES)
        self.assertEqual(
            set(selection["contour_checkpoint_filenames"]),
            {"global", "epoch105", "autopet", "lung"},
        )

    def test_no_improving_source_keeps_global(self):
        decision, selection, checks = select_routes(summary(), summary())
        self.assertEqual(decision, "KEEP_GLOBAL_CONTOUR")
        self.assertFalse(checks["at_least_one_source_route_selected"])
        self.assertEqual(set(selection["source_to_route"].values()), {"global"})

    def test_requires_both_case_and_cohort_gain(self):
        decision, selection, _ = select_routes(
            summary(),
            summary(gains={"autopet_weak": (0.03, 0.0005)}),
        )
        self.assertEqual(decision, "KEEP_GLOBAL_CONTOUR")
        self.assertFalse(selection["unit_reports"]["autopet"]["selected"])

    def test_index_coverage_and_classifier_must_match(self):
        for candidate in (
            summary(index="b" * 64, gains={"autopet_weak": (0.03, 0.02)}),
            summary(coverage_cases=706, gains={"autopet_weak": (0.03, 0.02)}),
            summary(ap=0.81, gains={"autopet_weak": (0.03, 0.02)}),
        ):
            decision, _, checks = select_routes(summary(), candidate)
            self.assertEqual(decision, "KEEP_GLOBAL_CONTOUR")
            self.assertFalse(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
