import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "audit_partial_box_scope.py"
SPEC = importlib.util.spec_from_file_location("audit_partial_box_scope", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PartialBoxScopeAuditTest(unittest.TestCase):
    def test_separated_components_expose_global_box_overreach(self):
        mask = np.zeros((32, 32, 32), dtype=np.bool_)
        mask[5:7, 5:7, 5:7] = True
        mask[25:27, 25:27, 25:27] = True
        result = MODULE.box_scope(mask)
        self.assertEqual(result["components"], 2)
        self.assertGreater(result["global_over_component_union"], 2.0)
        self.assertLess(
            result["component_box_union_voxels"], result["global_box_voxels"]
        )

    def test_touching_diagonal_voxels_use_declared_26_connectivity(self):
        mask = np.zeros((8, 8, 8), dtype=np.bool_)
        mask[3, 3, 3] = True
        mask[4, 4, 4] = True
        self.assertEqual(MODULE.box_scope(mask)["components"], 1)

    def test_empty_mask_is_explicit(self):
        result = MODULE.box_scope(np.zeros((4, 4, 4), dtype=np.bool_))
        self.assertEqual(result["components"], 0)
        self.assertEqual(result["global_over_component_union"], 0.0)


if __name__ == "__main__":
    unittest.main()
