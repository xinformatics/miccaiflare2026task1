import unittest
import json
import tempfile
from pathlib import Path

import cc3d
import numpy as np

from generate_lung_cell_proposals import (
    COLOURS,
    _run_case,
    atomic_nifti_temporary,
    cell_colour,
    native_cell_box,
    paint_cells_by_colour,
)


class LungCellProposalTests(unittest.TestCase):
    def test_atomic_temporary_is_not_visible_as_a_final_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            colour = Path(temporary) / "colour0"
            colour.mkdir()
            destination = colour / "case.nii.gz"
            staging = atomic_nifti_temporary(destination)
            staging.touch()
            self.assertEqual(staging.parent.name, ".atomic_temporary")
            self.assertEqual(list(colour.glob("*.nii.gz")), [])

    def test_resume_requires_atomic_sidecar_and_all_colour_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = "case.nii.gz"
            row = {
                "case": name,
                "cells": 3,
                "cells_by_colour": [1, 1, 1, 0, 0, 0, 0, 0],
                "maximum_component_axis_extent_mm": 48.0,
                "image_sha256": "frozen",
                "reused": False,
            }
            for colour in range(COLOURS):
                destination = root / f"colour{colour}" / name
                destination.parent.mkdir(parents=True)
                destination.touch()
            marker = root / "case_complete" / f"{name}.json"
            marker.parent.mkdir()
            marker.write_text(json.dumps(row), encoding="utf-8")
            reused = _run_case(
                {
                    "image": root / "does-not-need-to-exist.nii.gz",
                    "prediction_name": name,
                    "output_root": root,
                    "target_spacing_mm": 2.5,
                    "cell_stride_mm": 48.0,
                    "resume": True,
                }
            )
            self.assertTrue(reused["reused"])
            self.assertEqual(reused["cells"], 3)

    def test_cell_extent_is_bounded_in_physical_space(self):
        spacing = np.asarray([0.8, 1.3, 3.0])
        box = native_cell_box(np.asarray([2, 3, 1]), spacing, 48.0, (256, 256, 96))
        extents = np.asarray(
            [(axis.stop - axis.start) * spacing[index] for index, axis in enumerate(box)]
        )
        self.assertTrue(np.all(extents <= 48.0 + 1e-8))
        self.assertTrue(np.all(extents > 0))

    def test_eight_colours_keep_same_source_cells_disconnected(self):
        cells = [
            np.asarray([x, y, z])
            for x in range(4)
            for y in range(4)
            for z in range(4)
        ]
        masks = paint_cells_by_colour((192, 192, 192), np.ones(3), cells, 48.0)
        self.assertEqual(sum(int(mask.any()) for mask in masks), 8)
        self.assertEqual(sum(int(mask.sum()) for mask in masks), 192**3)
        for colour, mask in enumerate(masks):
            expected = sum(cell_colour(cell) == colour for cell in cells)
            _, observed = cc3d.connected_components(
                mask.astype(np.uint8), connectivity=26, return_N=True
            )
            self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
