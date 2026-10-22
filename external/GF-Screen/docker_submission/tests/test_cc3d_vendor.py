from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import cc3d
import numpy as np


VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "cc3d.py"
SPEC = importlib.util.spec_from_file_location("deployment_cc3d", VENDOR)
deployment_cc3d = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(deployment_cc3d)


def component_sizes(labels: np.ndarray) -> list[int]:
    return sorted(np.bincount(labels.ravel())[1:].tolist())


class VendorCc3dTest(unittest.TestCase):
    def test_binary_component_partitions_match_real_cc3d(self) -> None:
        rng = np.random.default_rng(20260827)
        masks = [
            rng.random((31, 29, 17)) > 0.93,
            rng.random((19, 23, 13)) > 0.72,
            np.zeros((11, 12, 13), dtype=bool),
        ]
        masks[2][1:4, 2:6, 3:8] = True
        masks[2][4, 6, 8] = True
        for connectivity in (6, 18, 26):
            for mask in masks:
                expected, expected_count = cc3d.connected_components(
                    mask.astype(np.uint8), connectivity=connectivity, return_N=True
                )
                observed, observed_count = deployment_cc3d.connected_components(
                    mask, connectivity=connectivity, return_N=True
                )
                self.assertEqual(observed.dtype, np.uint32)
                self.assertEqual(observed_count, expected_count)
                self.assertEqual(component_sizes(observed), component_sizes(expected))

    def test_rejects_nonbinary_or_unsupported_surface(self) -> None:
        with self.assertRaises(ValueError):
            deployment_cc3d.connected_components(np.full((2, 2, 2), 2))
        with self.assertRaises(TypeError):
            deployment_cc3d.connected_components(np.zeros((2, 2, 2)), delta=1)
        with self.assertRaises(NotImplementedError):
            deployment_cc3d.statistics(np.zeros((2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
