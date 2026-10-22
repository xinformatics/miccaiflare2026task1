import unittest

import torch

from candidate_refiner_source_conditioning import (
    DEFAULT_SOURCE_CONDITION_GROUPS,
    append_source_condition_batch,
    append_source_condition_channels,
    source_condition_index,
    validate_source_condition_groups,
)


class CandidateRefinerSourceConditioningTests(unittest.TestCase):
    def test_lung_colours_collapse_to_one_semantic_channel(self):
        groups = DEFAULT_SOURCE_CONDITION_GROUPS
        expected = groups.index("lung_cells")
        for colour in range(8):
            self.assertEqual(
                source_condition_index(f"lung_cell{colour}", groups), expected
            )

    def test_single_patch_appends_exact_one_hot_volume(self):
        image = torch.zeros(3, 2, 3, 4)
        conditioned = append_source_condition_channels(
            image, "currentgf", DEFAULT_SOURCE_CONDITION_GROUPS
        )
        self.assertEqual(tuple(conditioned.shape), (8, 2, 3, 4))
        self.assertTrue(torch.equal(conditioned[:3], image))
        one_hot = conditioned[3:].mean(dim=(1, 2, 3))
        torch.testing.assert_close(
            one_hot,
            torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
        )

    def test_batch_conditioning_preserves_row_alignment(self):
        images = torch.randn(2, 3, 2, 2, 2)
        conditioned = append_source_condition_batch(
            images,
            ["epoch105", "lung_cell7"],
            DEFAULT_SOURCE_CONDITION_GROUPS,
        )
        self.assertEqual(tuple(conditioned.shape), (2, 8, 2, 2, 2))
        torch.testing.assert_close(conditioned[:, :3], images)
        self.assertTrue(torch.all(conditioned[0, 3] == 1))
        self.assertTrue(torch.all(conditioned[1, 7] == 1))
        self.assertEqual(float(conditioned[:, 3:].sum()), 16.0)

    def test_unknown_or_duplicate_sources_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown semantic group"):
            source_condition_index("new_teacher", DEFAULT_SOURCE_CONDITION_GROUPS)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            validate_source_condition_groups(("currentgf", "currentgf"))
        with self.assertRaisesRegex(ValueError, "does not align"):
            append_source_condition_batch(
                torch.zeros(2, 3, 2, 2, 2),
                ["currentgf"],
                DEFAULT_SOURCE_CONDITION_GROUPS,
            )


if __name__ == "__main__":
    unittest.main()
