import unittest

import numpy as np

from proposal_component_instances import (
    instance_mask,
    source_separated_instances,
)


class ProposalComponentInstanceTests(unittest.TestCase):
    def test_union_bridge_does_not_merge_source_specific_components(self):
        first = np.zeros((9, 9, 9), dtype=bool)
        second = np.zeros_like(first)
        first[2:4, 3:5, 3:5] = True
        second[4:6, 3:5, 3:5] = True

        maps, instances = source_separated_instances([first, second])

        self.assertEqual(len(instances), 2)
        np.testing.assert_array_equal(instance_mask(maps, instances[0]), first)
        np.testing.assert_array_equal(instance_mask(maps, instances[1]), second)

    def test_exact_duplicate_is_removed_and_provenance_is_retained(self):
        first = np.zeros((8, 8, 8), dtype=bool)
        first[2:5, 1:4, 3:6] = True

        maps, instances = source_separated_instances([first, first.copy()])

        self.assertEqual(len(maps), 2)
        self.assertEqual(len(instances), 1)
        self.assertEqual(
            instances[0]["equivalent_components"],
            [
                {"source_index": 0, "source_component_id": 1},
                {"source_index": 1, "source_component_id": 1},
            ],
        )
        np.testing.assert_array_equal(instance_mask(maps, instances[0]), first)

    def test_overlapping_nonidentical_components_are_retained(self):
        first = np.zeros((8, 8, 8), dtype=bool)
        second = np.zeros_like(first)
        first[2:5, 2:5, 2:5] = True
        second[2:6, 2:5, 2:5] = True

        _, instances = source_separated_instances([first, second])

        self.assertEqual([entry["component_id"] for entry in instances], [1, 2])
        self.assertEqual([entry["voxels"] for entry in instances], [27, 36])

    def test_inventory_is_deterministic(self):
        first = np.zeros((8, 8, 8), dtype=bool)
        second = np.zeros_like(first)
        first[1, 1, 1] = True
        first[6, 6, 6] = True
        second[2:4, 2:4, 2:4] = True

        _, left = source_separated_instances([first, second])
        _, right = source_separated_instances([first.copy(), second.copy()])

        for lhs, rhs in zip(left, right):
            self.assertEqual(lhs["component_id"], rhs["component_id"])
            self.assertEqual(lhs["source_index"], rhs["source_index"])
            self.assertEqual(lhs["source_component_id"], rhs["source_component_id"])
            self.assertEqual(lhs["equivalent_components"], rhs["equivalent_components"])
            np.testing.assert_array_equal(lhs["centroid_voxel"], rhs["centroid_voxel"])


if __name__ == "__main__":
    unittest.main()
