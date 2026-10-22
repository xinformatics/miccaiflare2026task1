import unittest

import numpy as np

from GF_RL_trainer import annotated_recall_counts


class AnnotatedRecallDiagnosticsTests(unittest.TestCase):
    def test_prediction_only_regions_are_not_counted_as_false_positives(self):
        reference = np.zeros((20, 20, 20), dtype=bool)
        reference[1:3, 1:3, 1:3] = True
        reference[10:12, 10:12, 10:12] = True
        prediction = np.zeros_like(reference)
        prediction[1:3, 1:3, 1:3] = True
        prediction[15:19, 1:5, 12:16] = True
        counts = annotated_recall_counts(prediction, reference, voxel_mm3=3.0)
        self.assertEqual(counts["annotated_voxel_tp"], 8)
        self.assertEqual(counts["annotated_voxel_total"], 16)
        self.assertEqual(counts["annotated_components_detected"], 1)
        self.assertEqual(counts["annotated_components_total"], 2)

    def test_ten_percent_overlap_is_the_detection_boundary(self):
        reference = np.zeros((20, 20, 20), dtype=bool)
        reference[2:12, 2, 2] = True
        prediction = np.zeros_like(reference)
        prediction[2, 2, 2] = True
        detected = annotated_recall_counts(prediction, reference, voxel_mm3=1.0)
        rejected = annotated_recall_counts(
            prediction,
            reference,
            voxel_mm3=1.0,
            minimum_component_overlap_fraction=0.11,
        )
        self.assertEqual(detected["annotated_components_detected"], 1)
        self.assertEqual(rejected["annotated_components_detected"], 0)

    def test_physical_size_bins_are_exclusive_and_exhaustive(self):
        reference = np.zeros((60, 60, 60), dtype=bool)
        reference[1:3, 1:3, 1:3] = True       # 8 mm3
        reference[5:10, 5:10, 5:10] = True   # 125 mm3
        reference[15:23, 15:23, 15:23] = True  # 512 mm3
        reference[30:41, 30:41, 30:41] = True  # 1331 mm3
        reference[45:60, 45:60, 45:60] = True  # 3375 mm3
        counts = annotated_recall_counts(reference, reference, voxel_mm3=1.0)
        self.assertEqual(counts["annotated_components_total"], 5)
        for tag in ("lt100", "100to500", "500to1000", "1000to3000", "ge3000"):
            self.assertEqual(counts[f"annotated_components_{tag}_total"], 1)
            self.assertEqual(counts[f"annotated_components_{tag}_detected"], 1)

    def test_invalid_inputs_fail_closed(self):
        mask = np.zeros((4, 4, 4), dtype=bool)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            annotated_recall_counts(mask, mask[:3], voxel_mm3=1.0)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            annotated_recall_counts(mask, mask, voxel_mm3=0.0)


if __name__ == "__main__":
    unittest.main()
