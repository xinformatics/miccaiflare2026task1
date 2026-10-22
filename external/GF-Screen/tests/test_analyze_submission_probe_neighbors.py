import unittest

from analyze_submission_probe_neighbors import nearest_neighbors


class ProbeNeighborTests(unittest.TestCase):
    def test_orders_complete_pairs_by_public_metric_distance(self):
        history = [
            {"model": "far", "configuration": "x", "public_dsc": "0.1", "public_nsd": "0.1", "codabench_dsc": "0.2", "codabench_nsd": "0.2"},
            {"model": "near", "configuration": "y", "public_dsc": "0.6", "public_nsd": "0.5", "codabench_dsc": "0.7", "codabench_nsd": "0.6"},
            {"model": "incomplete", "configuration": "z", "public_dsc": "", "public_nsd": "", "codabench_dsc": "0.9", "codabench_nsd": "0.9"},
        ]
        result = nearest_neighbors(history, 0.61, 0.51, 1)
        self.assertEqual(result[0]["model"], "near")

    def test_rejects_more_neighbors_than_complete_pairs(self):
        history = [
            {"model": "one", "configuration": "x", "public_dsc": "0.6", "public_nsd": "0.5", "codabench_dsc": "0.7", "codabench_nsd": "0.6"},
        ]
        with self.assertRaises(ValueError):
            nearest_neighbors(history, 0.6, 0.5, 2)


if __name__ == "__main__":
    unittest.main()
