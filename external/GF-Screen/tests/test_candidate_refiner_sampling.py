import unittest

import numpy as np

from candidate_refiner_sampling import (
    hierarchical_sample_weights,
    sampling_audit,
    semantic_source_group,
)


class CandidateRefinerSamplingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"label": "0", "cohort": "A", "case": "a0"},
            {"label": "0", "cohort": "A", "case": "a0"},
            {"label": "0", "cohort": "B", "case": "b0"},
            {"label": "1", "cohort": "A", "case": "a1"},
            {"label": "1", "cohort": "B", "case": "b1"},
            {"label": "1", "cohort": "B", "case": "b2"},
            {"label": "1", "cohort": "B", "case": "b2"},
        ]

    def test_zero_reproduces_label_balance(self):
        weights = hierarchical_sample_weights(self.rows, 0.0)
        labels = np.asarray([int(row["label"]) for row in self.rows])
        legacy = np.where(
            labels == 1,
            0.5 / int(labels.sum()),
            0.5 / int((labels == 0).sum()),
        ).astype(np.float64)
        self.assertTrue(np.array_equal(weights, legacy))
        for label in (0, 1):
            mass = sum(w for row, w in zip(self.rows, weights) if int(row["label"]) == label)
            self.assertAlmostEqual(mass, 0.5)
        self.assertAlmostEqual(weights[0], weights[1])

    def test_one_balances_cohorts_cases_and_patches(self):
        weights = hierarchical_sample_weights(self.rows, 1.0)
        for label in (0, 1):
            for cohort in ("A", "B"):
                mass = sum(
                    w
                    for row, w in zip(self.rows, weights)
                    if int(row["label"]) == label and row["cohort"] == cohort
                )
                self.assertAlmostEqual(mass, 0.25)
        # The two patches from the same case divide that case's probability.
        self.assertAlmostEqual(weights[-1], weights[-2])
        self.assertAlmostEqual(weights[-1] + weights[-2], 0.125)

    def test_intermediate_is_finite_and_auditable(self):
        weights = hierarchical_sample_weights(self.rows, 0.5)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        audit = sampling_audit(self.rows, 0.5)
        self.assertEqual(audit["patches"], len(self.rows))
        self.assertGreater(audit["effective_sample_size"], 0)

    def test_rejects_missing_label(self):
        with self.assertRaises(ValueError):
            hierarchical_sample_weights(self.rows[:3], 0.5)

    def test_audit_reports_source_mass_without_changing_weights(self):
        rows = [
            {**row, "source_name": "epoch105" if index % 2 == 0 else "weak"}
            for index, row in enumerate(self.rows)
        ]
        audit = sampling_audit(rows, 0.5)
        self.assertTrue(audit["source_cell_mass"])
        self.assertAlmostEqual(
            sum(row["probability"] for row in audit["source_cell_mass"]),
            1.0,
        )
        np.testing.assert_array_equal(
            hierarchical_sample_weights(rows, 0.5),
            hierarchical_sample_weights(self.rows, 0.5),
        )

    def test_audit_rejects_partial_source_identity(self):
        rows = [{**row} for row in self.rows]
        rows[0]["source_name"] = "epoch105"
        with self.assertRaisesRegex(ValueError, "source_name must cover every row"):
            sampling_audit(rows, 0.5)

    def test_source_group_balance_collapses_lung_parity_streams(self):
        rows = []
        for source, count in (
            ("epoch105", 1),
            ("currentgf", 1),
            ("lung_cell0", 4),
            ("lung_cell1", 4),
        ):
            rows.extend(
                {
                    "label": str(label),
                    "cohort": "A",
                    "case": f"case{label}",
                    "source_name": source,
                }
                for label in (0, 1)
                for _ in range(count)
            )
        weights = hierarchical_sample_weights(
            rows, 1.0, balance_source_groups=True
        )
        for label in (0, 1):
            mass = {}
            for row, weight in zip(rows, weights):
                if int(row["label"]) == label:
                    group = semantic_source_group(row["source_name"])
                    mass[group] = mass.get(group, 0.0) + float(weight)
            self.assertEqual(set(mass), {"epoch105", "currentgf", "lung_cells"})
            for value in mass.values():
                self.assertAlmostEqual(value, 1.0 / 6.0)

    def test_source_group_balance_requires_source_identity(self):
        with self.assertRaisesRegex(ValueError, "source_name is required"):
            hierarchical_sample_weights(
                self.rows, 0.5, balance_source_groups=True
            )


if __name__ == "__main__":
    unittest.main()
