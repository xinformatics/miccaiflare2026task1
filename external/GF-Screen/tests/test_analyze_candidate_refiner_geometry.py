import unittest

from analyze_candidate_refiner_geometry import summarize


class CandidateRefinerGeometryTests(unittest.TestCase):
    def test_positive_geometry_is_grouped_by_cohort_and_component(self):
        rows = [
            {
                "label": "1", "cohort": "A", "case": "a", "component_id": "1",
                "seed_mm3": "10", "seed_overlap_fraction": "0.5",
                "intersection_voxels": "4", "target_resample_fallback": "0",
            },
            {
                "label": "1", "cohort": "A", "case": "a", "component_id": "1",
                "seed_mm3": "10", "seed_overlap_fraction": "0.75",
                "intersection_voxels": "6", "target_resample_fallback": "1",
            },
            {
                "label": "0", "cohort": "A", "case": "a", "component_id": "2",
                "seed_mm3": "20", "seed_overlap_fraction": "0",
                "intersection_voxels": "0", "target_resample_fallback": "0",
            },
            {
                "label": "1", "cohort": "B", "case": "b", "component_id": "1",
                "seed_mm3": "30", "seed_overlap_fraction": "1",
                "intersection_voxels": "8", "target_resample_fallback": "0",
            },
        ]
        report = summarize(rows)
        self.assertEqual([row["cohort"] for row in report], ["A", "B"])
        self.assertEqual(report[0]["positive_components"], 1)
        self.assertEqual(report[0]["positive_tiles"], 2)
        self.assertEqual(report[0]["tiles_per_positive_component"], 2.0)
        self.assertEqual(report[0]["target_resample_fallbacks"], 1)
        self.assertEqual(report[1]["seed_overlap_fraction"]["median"], 1.0)

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            summarize(
                [{
                    "label": "1", "cohort": "A", "case": "a", "component_id": "1",
                    "seed_mm3": "10", "seed_overlap_fraction": "1.1",
                    "intersection_voxels": "4",
                }]
            )


if __name__ == "__main__":
    unittest.main()
