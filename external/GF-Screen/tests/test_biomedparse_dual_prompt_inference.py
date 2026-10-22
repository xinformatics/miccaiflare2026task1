from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "BiomedParse"
    / "miccai_dual_prompt_inference.py"
)
SPEC = importlib.util.spec_from_file_location("miccai_dual_prompt_inference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DualPromptInferenceTests(unittest.TestCase):
    def test_prompts_are_separate_and_ordered(self):
        self.assertEqual(
            MODULE.PROMPT_ROUTES,
            (("whole_body_lesion", "whole-body lesion"), ("tumor", "Tumor")),
        )
        self.assertNotIn("[SEP]", "".join(prompt for _, prompt in MODULE.PROMPT_ROUTES))

    def test_output_contract_matches_single_prompt_runner(self):
        segmentation = np.zeros((2, 3, 4), dtype=np.uint8)
        segmentation[0, 1, 2] = 1
        fields = MODULE.output_fields(
            {0.5: segmentation}, np.asarray([[0.7]], dtype=np.float16), 1.25,
            "Tumor", [0.5]
        )
        self.assertEqual(
            set(fields),
            {"segs_e050", "segs", "object_existence", "existence_thresholds", "inference_seconds", "prompts"},
        )
        self.assertTrue(np.array_equal(fields["segs"], fields["segs_e050"]))
        self.assertEqual(fields["prompts"].tolist(), ["Tumor"])

    def test_historical_first_threshold_controls_segs_alias(self):
        masks = {
            0.3: np.ones((1, 1, 2), dtype=np.uint8),
            0.5: np.zeros((1, 1, 2), dtype=np.uint8),
            0.7: np.zeros((1, 1, 2), dtype=np.uint8),
        }
        fields = MODULE.output_fields(
            masks, np.asarray([[0.9]], dtype=np.float16), 0.1,
            "whole-body lesion", [0.3, 0.5, 0.7]
        )
        self.assertTrue(np.array_equal(fields["segs"], fields["segs_e030"]))
        self.assertFalse(np.array_equal(fields["segs"], fields["segs_e050"]))

    def test_atomic_output_and_completion_are_fail_closed(self):
        segmentation = np.ones((1, 2, 3), dtype=np.uint8)
        fields = MODULE.output_fields(
            {0.5: segmentation}, np.asarray([[0.9]], dtype=np.float16), 0.1,
            "whole-body lesion", [0.5]
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "case.npz"
            MODULE.atomic_save_npz(destination, fields)
            self.assertTrue(MODULE.output_complete(destination, [0.5], "whole-body lesion"))
            self.assertFalse(MODULE.output_complete(destination, [0.5], "Tumor"))
            self.assertFalse(any(Path(directory).glob("*.partial.npz")))


if __name__ == "__main__":
    unittest.main()
