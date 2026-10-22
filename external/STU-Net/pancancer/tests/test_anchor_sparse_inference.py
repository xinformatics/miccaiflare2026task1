import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PANCANCER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANCANCER))

from predict_stunet_anchor_sparse import (
    dependency_from_preprocessed_anchor,
    dependency_coverage_is_exact,
    probabilities_to_nifti_order,
    quantized_threshold_mask,
    select_dependency_slicers,
)


def grid_slicers(shape=(9, 10, 11), patch=(5, 5, 5), stride=(2, 3, 3)):
    slicers = []
    for x in range(0, shape[0] - patch[0] + 1, stride[0]):
        for y in range(0, shape[1] - patch[1] + 1, stride[1]):
            for z in range(0, shape[2] - patch[2] + 1, stride[2]):
                slicers.append(
                    (slice(None), slice(x, x + 5), slice(y, y + 5), slice(z, z + 5))
                )
    return slicers


class AnchorSparseInferenceTests(unittest.TestCase):
    def test_probability_export_is_normalized_to_nifti_axis_order(self):
        reference_shape = (3, 4, 5)
        nifti_order = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        self.assertIs(
            probabilities_to_nifti_order(nifti_order, reference_shape),
            nifti_order,
        )
        nnunet_order = nifti_order.transpose(0, 3, 2, 1)
        restored = probabilities_to_nifti_order(nnunet_order, reference_shape)
        self.assertTrue(np.array_equal(restored, nifti_order))
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            probabilities_to_nifti_order(np.zeros((2, 3, 4, 6)), reference_shape)

    def test_selected_windows_have_exact_dependency_multiplicity(self):
        shape = (9, 10, 11)
        dependency = np.zeros(shape, dtype=bool)
        dependency[1:3, 1:3, 1:3] = True
        full = grid_slicers(shape)
        selected = select_dependency_slicers(full, dependency)
        self.assertLess(len(selected), len(full))
        self.assertTrue(dependency_coverage_is_exact(full, selected, dependency))

    def test_excluding_any_contributing_window_breaks_proof(self):
        dependency = np.zeros((9, 10, 11), dtype=bool)
        dependency[4, 5, 5] = True
        full = grid_slicers()
        selected = select_dependency_slicers(full, dependency)
        self.assertGreater(len(selected), 1)
        self.assertFalse(dependency_coverage_is_exact(full, selected[1:], dependency))

    def test_empty_dependency_selects_no_windows(self):
        dependency = np.zeros((9, 10, 11), dtype=bool)
        self.assertEqual(select_dependency_slicers(grid_slicers(), dependency), [])

    def test_invalid_dependency_rank_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "three-dimensional"):
            select_dependency_slicers([], np.zeros((2, 2), dtype=bool))

    def test_p70_uses_the_exact_deployed_uint16_quantization(self):
        values = np.array(
            [45874 / 65535, 45875 / 65535, 0.700001, np.nextafter(0.7, 1.0)],
            dtype=np.float32,
        )
        observed = quantized_threshold_mask(values, 0.70)
        quantized = np.rint(np.clip(values, 0, 1) * 65535).astype(np.uint16)
        expected = quantized.astype(np.float32) / 65535 > 0.70
        self.assertTrue(np.array_equal(observed, expected))

    def test_quantized_threshold_rejects_nonfinite_input(self):
        with self.assertRaisesRegex(ValueError, "invalid probability"):
            quantized_threshold_mask(np.array([np.nan]), 0.70)

    def test_nonempty_anchor_erased_by_preprocessing_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "erased a nonempty GF anchor"):
            dependency_from_preprocessed_anchor(np.zeros((4, 4, 4)), True)

    def test_dependency_halo_contains_preprocessed_anchor(self):
        anchor = np.zeros((7, 7, 7), dtype=bool)
        anchor[3, 3, 3] = True
        dependency = dependency_from_preprocessed_anchor(anchor, True)
        self.assertTrue(dependency[anchor].all())
        self.assertGreater(int(dependency.sum()), int(anchor.sum()))

    def test_sparse_and_full_sliding_window_logits_match_on_dependency(self):
        class DeterministicNetwork(torch.nn.Module):
            def forward(self, value):
                return torch.cat((value * 0.25 - 0.1, value * 0.75 + 0.2), dim=1)

        def predictor(dependency=None):
            instance = __import__(
                "predict_stunet_anchor_sparse",
                fromlist=["AnchorSparsePredictor"],
            ).AnchorSparsePredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=False,
                device=torch.device("cpu"),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
            instance.configuration_manager = SimpleNamespace(patch_size=(5, 6, 7))
            instance.label_manager = SimpleNamespace(num_segmentation_heads=2)
            instance.allowed_mirroring_axes = None
            instance.network = DeterministicNetwork()
            instance.dependency_mask = dependency
            return instance

        values = torch.arange(1 * 9 * 10 * 11, dtype=torch.float32).reshape(1, 9, 10, 11)
        dependency = np.zeros((9, 10, 11), dtype=bool)
        dependency[1:3, 1:4, 1:5] = True
        full_predictor = predictor()
        sparse_predictor = predictor(dependency)
        full = full_predictor.predict_sliding_window_return_logits(values)
        sparse = sparse_predictor.predict_sliding_window_return_logits(values)
        self.assertLess(sparse_predictor.selected_tile_count, sparse_predictor.full_tile_count)
        self.assertTrue(
            torch.equal(full[:, torch.from_numpy(dependency)], sparse[:, torch.from_numpy(dependency)])
        )

    def test_source_is_public_only_and_never_packages_or_uploads(self):
        source = (PANCANCER / "predict_stunet_anchor_sparse.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('choices=("public",)', source)
        self.assertIn('"deployment_authorized": False', source)
        self.assertIn('"model_initialization_seconds":', source)
        self.assertIn('"wall_seconds":', source)
        self.assertIn('"public_labels_read": False', source)
        self.assertIn('"hidden_read": False', source)
        for forbidden in ("Validation-Hidden", "codabench.org", "requests.post", "curl "):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
