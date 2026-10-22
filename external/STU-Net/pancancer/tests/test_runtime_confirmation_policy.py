import sys
import unittest
from pathlib import Path

import numpy as np


PANCANCER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANCANCER))

from runtime_confirmation_policy import confirmation_state, exact_early_exit_mask


class RuntimeConfirmationPolicyTests(unittest.TestCase):
    def test_empty_anchor_exits_with_exact_empty_mask(self):
        anchor = np.zeros((5, 5, 5), dtype=bool)
        result = exact_early_exit_mask(anchor, np.ones_like(anchor))
        self.assertIsNotNone(result)
        self.assertFalse(result.any())
        self.assertFalse(confirmation_state(anchor, anchor)["requires_stunet"])

    def test_all_components_confirmed_exits_with_exact_anchor(self):
        anchor = np.zeros((8, 8, 8), dtype=bool)
        anchor[1:3, 1:3, 1:3] = True
        anchor[5:7, 5:7, 5:7] = True
        freetumor = np.zeros_like(anchor)
        freetumor[1, 1, 1] = True
        freetumor[6, 6, 6] = True
        state = confirmation_state(anchor, freetumor)
        self.assertEqual(state["anchor_component_count"], 2)
        self.assertEqual(state["freetumor_confirmed_component_count"], 2)
        self.assertFalse(state["requires_stunet"])
        self.assertTrue(np.array_equal(exact_early_exit_mask(anchor, freetumor), anchor))

    def test_unconfirmed_component_requires_stunet(self):
        anchor = np.zeros((8, 8, 8), dtype=bool)
        anchor[1:3, 1:3, 1:3] = True
        anchor[5:7, 5:7, 5:7] = True
        freetumor = np.zeros_like(anchor)
        freetumor[1, 1, 1] = True
        state = confirmation_state(anchor, freetumor)
        self.assertEqual(state["freetumor_confirmed_component_count"], 1)
        self.assertTrue(state["requires_stunet"])
        self.assertIsNone(exact_early_exit_mask(anchor, freetumor))

    def test_26_connectivity_matches_deployed_rule(self):
        anchor = np.zeros((4, 4, 4), dtype=bool)
        anchor[1, 1, 1] = True
        anchor[2, 2, 2] = True
        freetumor = np.zeros_like(anchor)
        freetumor[1, 1, 1] = True
        state = confirmation_state(anchor, freetumor)
        self.assertEqual(state["anchor_component_count"], 1)
        self.assertFalse(state["requires_stunet"])

    def test_shape_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            confirmation_state(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)))


if __name__ == "__main__":
    unittest.main()
