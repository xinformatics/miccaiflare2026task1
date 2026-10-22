import unittest

from audit_candidate_refiner_source_domains import summarize_rows


class CandidateRefinerSourceDomainAuditTests(unittest.TestCase):
    def test_semantic_lung_sources_are_pooled_and_mass_is_normalized(self):
        rows = [
            {"patch_id": "a", "case": "c1", "cohort": "x", "label": "1", "source_name": "currentgf"},
            {"patch_id": "b", "case": "c1", "cohort": "x", "label": "0", "source_name": "currentgf"},
            {"patch_id": "c", "case": "c2", "cohort": "x", "label": "1", "source_name": "lung_cell0"},
            {"patch_id": "d", "case": "c2", "cohort": "x", "label": "0", "source_name": "lung_cell7"},
        ]
        report, masses = summarize_rows(rows, 0.5)
        self.assertEqual(set(report), {"currentgf", "lung_cells"})
        self.assertEqual(report["lung_cells"]["rows"], 2)
        self.assertAlmostEqual(sum(masses.values()), 1.0)
        self.assertAlmostEqual(
            sum(float(value["source_balanced_sampling_mass"]) for value in report.values()),
            1.0,
        )

    def test_invalid_patch_or_label_identity_fails_closed(self):
        duplicate = [
            {"patch_id": "a", "case": "c1", "cohort": "x", "label": "1", "source_name": "currentgf"},
            {"patch_id": "a", "case": "c2", "cohort": "x", "label": "0", "source_name": "epoch105"},
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_rows(duplicate, 0.5)
        invalid = [dict(duplicate[0], patch_id="b", label="2")]
        with self.assertRaisesRegex(ValueError, "binary"):
            summarize_rows(invalid, 0.5)


if __name__ == "__main__":
    unittest.main()
