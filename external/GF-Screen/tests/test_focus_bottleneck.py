import unittest

from analyze_focus_bottleneck import classify_route


class FocusBottleneckTests(unittest.TestCase):
    def test_focus_limited_when_oracle_has_little_headroom(self):
        self.assertEqual(classify_route(0.02, 0.01), "focus_segmentation_limited")

    def test_waits_for_full_coverage_after_large_oracle_gain(self):
        self.assertEqual(classify_route(0.10, None), "await_full_coverage")

    def test_precise_proposals_needed_when_all_crops_do_not_help(self):
        self.assertEqual(classify_route(0.10, 0.01), "proposal_quality_limited")

    def test_glance_recall_when_all_crops_and_oracle_both_help(self):
        self.assertEqual(classify_route(0.10, 0.04), "glance_recall_limited")

    def test_intermediate_headroom_is_mixed(self):
        self.assertEqual(classify_route(0.05, 0.02), "mixed_glance_focus")


if __name__ == "__main__":
    unittest.main()
