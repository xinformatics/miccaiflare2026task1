import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from materialize_topk_prompt_replacement import gated_replacement
from prepare_topk_component_prompts import select_components


class TopKPromptTest(unittest.TestCase):
    def test_selects_largest_eligible_component(self):
        mask = np.zeros((12, 12, 12), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        mask[6:10, 6:10, 6:10] = True
        selected, records = select_components(
            mask, voxel_mm3=2.0, top_k=1, min_mm3=0.0, max_mm3=1000.0
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["voxels"], 64)
        self.assertEqual(int(selected.sum()), 64)
        self.assertFalse(np.any(selected[1:3, 1:3, 1:3]))

    def test_accepted_replacement_preserves_unprompted_anchor(self):
        anchor = np.zeros((12, 12, 12), dtype=bool)
        anchor[1:4, 1:4, 1:4] = True
        anchor[8:10, 8:10, 8:10] = True
        prompt = np.zeros_like(anchor)
        prompt[1:4, 1:4, 1:4] = True
        refined = np.zeros_like(anchor)
        refined[1:5, 1:5, 1:5] = True
        result, audit = gated_replacement(
            anchor, prompt, refined, 0.25, 0.25, 0.25, 4.0
        )
        self.assertTrue(audit["accepted"])
        self.assertTrue(np.all(result[8:10, 8:10, 8:10]))
        self.assertTrue(np.all(result[1:5, 1:5, 1:5]))

    def test_failed_gate_returns_anchor_bit_exact(self):
        anchor = np.zeros((8, 8, 8), dtype=bool)
        anchor[1:3, 1:3, 1:3] = True
        prompt = anchor.copy()
        refined = np.zeros_like(anchor)
        refined[5:7, 5:7, 5:7] = True
        result, audit = gated_replacement(
            anchor, prompt, refined, 0.25, 0.25, 0.25, 4.0
        )
        self.assertFalse(audit["accepted"])
        np.testing.assert_array_equal(result, anchor)

    def test_prompt_must_be_subset_of_anchor(self):
        anchor = np.zeros((4, 4, 4), dtype=bool)
        prompt = np.zeros_like(anchor)
        prompt[1, 1, 1] = True
        with self.assertRaises(ValueError):
            gated_replacement(anchor, prompt, prompt, 0, 0, 0, 10)


if __name__ == "__main__":
    unittest.main()
