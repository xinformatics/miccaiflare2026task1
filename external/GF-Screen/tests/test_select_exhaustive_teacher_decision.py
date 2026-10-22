from __future__ import annotations

import unittest

from select_exhaustive_teacher_decision import decide


def rows(*, overall_gain: float, small_gain: float, precision_gain: float):
    result = []
    for threshold in (0.85, 0.90, 0.95, 0.97):
        result.append(
            {
                "mode": "gated",
                "lesion_threshold": str(threshold),
                "detection_recall": "0.30",
                "small_recall_100to1000": "0.10",
                "component_precision": "0.60",
                "DSC": "0.50",
                "NSD": "0.40",
            }
        )
        result.append(
            {
                "mode": "exhaustive",
                "lesion_threshold": str(threshold),
                "detection_recall": str(0.30 + overall_gain),
                "small_recall_100to1000": str(0.10 + small_gain),
                "component_precision": str(0.60 + precision_gain),
                "DSC": "0.51",
                "NSD": "0.41",
            }
        )
    return result


class ExhaustiveTeacherDecisionTests(unittest.TestCase):
    def test_retains_only_when_all_three_guards_pass(self):
        report = decide(rows(overall_gain=0.03, small_gain=0.02, precision_gain=-0.01))
        self.assertEqual(report["decision"], "RETAIN_EXHAUSTIVE_FOR_UNLABELED")
        self.assertEqual(report["eligible_thresholds"], [0.85, 0.9, 0.95, 0.97])

    def test_rejects_gain_that_does_not_recover_small_lesions(self):
        report = decide(rows(overall_gain=0.03, small_gain=0.0, precision_gain=0.0))
        self.assertEqual(report["decision"], "KEEP_GATED_TEACHER")
        self.assertEqual(report["eligible_thresholds"], [])

    def test_rejects_component_precision_collapse(self):
        report = decide(rows(overall_gain=0.04, small_gain=0.03, precision_gain=-0.03))
        self.assertEqual(report["decision"], "KEEP_GATED_TEACHER")

    def test_rejects_incomplete_grid(self):
        incomplete = rows(overall_gain=0.03, small_gain=0.02, precision_gain=0.0)[:-1]
        with self.assertRaisesRegex(ValueError, "teacher state mismatch"):
            decide(incomplete)


if __name__ == "__main__":
    unittest.main()
