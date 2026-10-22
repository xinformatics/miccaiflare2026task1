import unittest

import torch

from models.glance import crop_labels_with_coords, select_crops_from_proposal


def legacy_selection(proposal, coords, crop_size, minimum):
    crops = crop_labels_with_coords(proposal, coords, crop_size=crop_size)
    return crops.gt(0).flatten(1).sum(1).ge(minimum)


class ProposalGuidanceTests(unittest.TestCase):
    def test_integral_counts_match_legacy_for_3d_proposal(self):
        generator = torch.Generator().manual_seed(20260820)
        proposal = torch.rand((13, 15, 11), generator=generator).gt(0.82)
        crop_size = (5, 6, 4)
        coords = [(0, 0, 0), (8, 9, 7), (3, 2, 1), (7, 1, 5)]
        for minimum in (1, 7, 31, 120):
            actual = select_crops_from_proposal(
                proposal, coords, crop_size=crop_size, min_voxels=minimum
            )
            expected = legacy_selection(proposal, coords, crop_size, minimum)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_integral_counts_match_legacy_for_multichannel_proposal(self):
        proposal = torch.zeros((2, 9, 10, 8))
        proposal[0, 0:4, 0:4, 0:4] = 1
        proposal[1, 2:7, 3:9, 1:6] = 1
        crop_size = (4, 5, 3)
        coords = [(0, 0, 0), (5, 5, 5), (2, 3, 1)]
        for minimum in (1, 10, 40, 80):
            actual = select_crops_from_proposal(
                proposal, coords, crop_size=crop_size, min_voxels=minimum
            )
            expected = legacy_selection(proposal, coords, crop_size, minimum)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_coordinate_list_returns_empty_boolean_tensor(self):
        selected = select_crops_from_proposal(
            torch.zeros((1, 4, 4, 4)), [], crop_size=(2, 2, 2)
        )
        self.assertEqual(selected.dtype, torch.bool)
        self.assertEqual(tuple(selected.shape), (0,))

    def test_out_of_bounds_coordinate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            select_crops_from_proposal(
                torch.zeros((1, 4, 4, 4)), [(3, 0, 0)], crop_size=(2, 2, 2)
            )


if __name__ == "__main__":
    unittest.main()
