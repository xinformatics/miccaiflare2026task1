from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "docker_submission" / "winner_mask_recipe.py"
SPEC = importlib.util.spec_from_file_location("winner_mask_recipe", MODULE)
recipe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = recipe
SPEC.loader.exec_module(recipe)


class WinnerMaskRecipeTests(unittest.TestCase):
    def test_historical_filter_uses_six_connectivity(self):
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[1, 1, 1] = True
        mask[2, 2, 2] = True
        result = recipe.filter_components_physical_6(mask, (1, 1, 1), 2)
        self.assertFalse(result.any())

    def test_confirmation_keeps_whole_component(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 2, 2] = True
        support = np.zeros_like(mask)
        support[1, 2, 2] = True
        self.assertTrue(np.array_equal(recipe.confirm_whole_components_26(mask, support), mask))

    def test_guided_rule_imports_only_supported_disjoint_component(self):
        base = np.zeros((12, 12, 12), dtype=bool)
        base[1:3, 1:3, 1:3] = True
        guided = np.zeros_like(base)
        guided[1:3, 1:3, 1:3] = True  # overlaps base, therefore ineligible
        guided[6:10, 6:10, 6:10] = True
        prompt = np.zeros_like(base)
        prompt[6:8, 6:10, 6:10] = True  # 50% and 32 mm3
        rejected = recipe.add_guided_components(
            base, guided, prompt, (1, 1, 1), minimum_support_mm3=33
        )
        accepted = recipe.add_guided_components(
            base, guided, prompt, (1, 1, 1), minimum_support_mm3=32
        )
        self.assertEqual(int(rejected.sum()), int(base.sum()))
        self.assertEqual(int(accepted.sum()), int(base.sum()) + 64)

    def test_touching_support_imports_complete_component(self):
        anchor = np.zeros((8, 8, 8), dtype=bool)
        anchor[2, 2, 2] = True
        support = np.zeros_like(anchor)
        support[2:6, 2, 2] = True
        support[7, 7, 7] = True
        expected = anchor | support
        expected[7, 7, 7] = False
        self.assertTrue(
            np.array_equal(recipe.union_touching_support_components_26(anchor, support), expected)
        )

    def test_quantized_p70_is_strict(self):
        anchor = np.zeros((4, 4, 4), dtype=bool)
        anchor[1, 1, 1] = True
        free = np.zeros_like(anchor)
        probability = np.zeros_like(anchor, dtype=np.uint16)
        probability[1, 1, 1] = np.uint16(45874)
        self.assertFalse(
            recipe.confirm_with_freetumor_or_stunet_p70(anchor, free, probability).any()
        )
        probability[1, 1, 1] = np.uint16(45875)
        self.assertTrue(
            recipe.confirm_with_freetumor_or_stunet_p70(anchor, free, probability).any()
        )


if __name__ == "__main__":
    unittest.main()
