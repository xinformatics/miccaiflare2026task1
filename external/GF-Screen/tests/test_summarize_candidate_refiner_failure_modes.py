import unittest

from summarize_candidate_refiner_failure_modes import choose_priority, summarize


def macro(dice=0.5, fp=0.7, fn=0.6):
    return {
        "dice": dice,
        "precision": 0.6,
        "recall": 0.7,
        "lesion_detection_recall": 0.65,
        "component_precision": 0.55,
        "fp_free_oracle_dice": fp,
        "fn_free_oracle_dice": fn,
    }


def row(name, values):
    return {"case": name, **{key: str(value) for key, value in values.items()}}


class FailureModeSummaryTests(unittest.TestCase):
    def test_priority_uses_predeclared_oracle_gap(self):
        self.assertEqual(
            choose_priority(macro(fp=0.75, fn=0.60))[0],
            "PRIORITIZE_FALSE_POSITIVE_CONTROL",
        )
        self.assertEqual(
            choose_priority(macro(fp=0.60, fn=0.75))[0],
            "PRIORITIZE_RECALL_AND_CONTOUR",
        )
        self.assertEqual(
            choose_priority(macro(fp=0.65, fn=0.64))[0],
            "PRIORITIZE_BALANCED_CLASSIFICATION_AND_CONTOUR",
        )

    def test_summary_requires_exact_manifest_coverage_and_matches_macro(self):
        values = macro()
        rows = [row("a.nii.gz", values), row("b.nii.gz", values)]
        manifest = {
            "items": [
                {"prediction_name": "a.nii.gz", "cohort": "A"},
                {"prediction_name": "b.nii.gz", "cohort": "B"},
            ]
        }
        lesion_summary = {
            "macro": {key: values[key] for key in (
                "dice", "precision", "recall", "fp_free_oracle_dice", "fn_free_oracle_dice"
            )},
            "gt_by_size_mm3": {"lt100": {"components": 2}},
            "prediction_by_size_mm3": {},
        }
        result = summarize(manifest, rows, lesion_summary)
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["cohorts"], 2)
        self.assertEqual(result["priority"], "PRIORITIZE_FALSE_POSITIVE_CONTROL")
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            summarize(manifest, rows[:1], lesion_summary)

    def test_undefined_ratio_counts_as_zero_in_case_macro(self):
        first = macro()
        second = macro()
        second.update(
            precision=None,
            recall=None,
            lesion_detection_recall=None,
            component_precision=None,
        )
        rows = [row("a.nii.gz", first), row("b.nii.gz", second)]
        manifest = {
            "items": [
                {"prediction_name": "a.nii.gz", "cohort": "A"},
                {"prediction_name": "b.nii.gz", "cohort": "A"},
            ]
        }
        lesion_summary = {
            "macro": {
                "dice": 0.5,
                "precision": 0.3,
                "recall": 0.35,
                "fp_free_oracle_dice": 0.7,
                "fn_free_oracle_dice": 0.6,
            }
        }
        result = summarize(manifest, rows, lesion_summary)
        self.assertEqual(result["macro"]["precision"], 0.3)
        self.assertEqual(result["macro"]["recall"], 0.35)
        self.assertEqual(result["macro"]["lesion_detection_recall"], 0.325)
        self.assertEqual(result["macro"]["component_precision"], 0.275)


if __name__ == "__main__":
    unittest.main()
