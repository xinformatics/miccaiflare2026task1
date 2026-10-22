import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analyze_component_center_sampling_headroom.py"
)
SPEC = importlib.util.spec_from_file_location("component_center_headroom", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ComponentCenterSamplingHeadroomTests(unittest.TestCase):
    def test_component_sizes_use_26_connectivity(self):
        segmentation = np.zeros((1, 4, 4, 4), dtype=np.int16)
        segmentation[0, 0, 0, 0] = 1
        segmentation[0, 1, 1, 1] = 1
        segmentation[0, 3, 3, 3] = 1
        self.assertEqual(MODULE.component_sizes(segmentation).tolist(), [2, 1])

    def test_component_uniform_increases_small_lesion_selection_mass(self):
        sizes = np.asarray([10, 1_000], dtype=np.int64)
        voxel, component = MODULE.selection_mass(sizes)
        small_bin = MODULE.bin_index(10 * MODULE.VOXEL_MM3, MODULE.LESION_MM3_BOUNDS)
        self.assertLess(voxel[small_bin], component[small_bin])
        self.assertAlmostEqual(float(voxel.sum()), 1.0)
        self.assertAlmostEqual(float(component.sum()), 1.0)

    def test_analysis_has_no_validation_hidden_or_upload_path(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "Validation-Public",
            "Validation-Hidden",
            "codabench.org",
            "requests.post",
            "curl ",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
