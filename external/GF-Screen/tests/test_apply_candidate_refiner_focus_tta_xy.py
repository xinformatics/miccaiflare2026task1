import tempfile
import unittest
import json
from pathlib import Path

import torch

import apply_candidate_refiner
from apply_candidate_refiner_focus_tta_xy import (
    EXPECTED_DEPLOYMENT_SHA256,
    FocusXYTTA,
    TTA_AXES,
    TTA_DIMENSIONS,
    TTA_MARKER_NAME,
    parse_tta_targets,
    sha256,
    tta_loader,
    write_tta_marker,
)
from focus_candidate_refiner import foreground_logit


class FocusXYTTADeploymentTests(unittest.TestCase):
    def test_wrapper_is_xy_locked_and_pins_deployment_source(self):
        self.assertEqual(TTA_AXES, "xy")
        self.assertEqual(TTA_DIMENSIONS, (2, 3))
        self.assertEqual(
            sha256(Path(apply_candidate_refiner.__file__)),
            EXPECTED_DEPLOYMENT_SHA256,
        )

    def test_probability_average_restores_each_flip(self):
        class Directional(torch.nn.Module):
            candidate_refiner_input_mode = "soft_lung_seed"

            def forward(self, images):
                x = torch.arange(
                    images.shape[2], dtype=images.dtype, device=images.device
                ).view(1, 1, -1, 1, 1)
                y = torch.arange(
                    images.shape[3], dtype=images.dtype, device=images.device
                ).view(1, 1, 1, -1, 1)
                foreground = images[:, :1] + x + 2 * y
                return torch.cat((torch.zeros_like(foreground), foreground), dim=1)

        images = torch.randn(2, 3, 4, 5, 6)
        base = Directional()

        def probability(view):
            return torch.sigmoid(foreground_logit(base(view)))

        expected = probability(images)
        for dimension in TTA_DIMENSIONS:
            expected = expected + torch.flip(
                probability(torch.flip(images, dims=(dimension,))),
                dims=(dimension,),
            )
        expected = expected / 3
        wrapped = FocusXYTTA(base)
        observed = torch.sigmoid(foreground_logit(wrapped(images)))
        self.assertEqual(wrapped.candidate_refiner_input_mode, "soft_lung_seed")
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)

    def test_loader_wraps_focus_model_and_preserves_input_mode(self):
        class Focus(torch.nn.Module):
            candidate_refiner_input_mode = "soft_seed"

            def forward(self, images):
                foreground = images[:, :1]
                return torch.cat((torch.zeros_like(foreground), foreground), dim=1)

        calls = []

        def loader(path, device):
            calls.append((path, device))
            return Focus().eval()

        wrapped = tta_loader(loader, {Path("checkpoint.pt")})(
            Path("checkpoint.pt"), torch.device("cpu")
        )
        self.assertIsInstance(wrapped, FocusXYTTA)
        self.assertEqual(wrapped.candidate_refiner_input_mode, "soft_seed")
        self.assertEqual(calls, [(Path("checkpoint.pt"), torch.device("cpu"))])

    def test_marker_binds_wrapper_and_inference_fingerprint(self):
        wrapper = Path(__import__(
            "apply_candidate_refiner_focus_tta_xy"
        ).__file__).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            plan = {
                "inference_fingerprint": "abc123",
                "identity": {
                    "implementation": [
                        {"path": str(wrapper), "sha256": sha256(wrapper)}
                    ]
                },
            }
            complete = {"inference_fingerprint": "abc123"}
            (work / "INFERENCE_PLAN.json").write_text(json.dumps(plan))
            (work / "INFERENCE_COMPLETE.json").write_text(json.dumps(complete))
            marker = write_tta_marker(work, ["focus_weak_sources_tta_xy"])
            self.assertEqual(marker["tta_axes"], "xy")
            self.assertEqual(marker["tta_views"], 3)
            self.assertEqual(marker["inference_fingerprint"], "abc123")
            self.assertEqual(
                marker["tta_focus_routes"], ["focus_weak_sources_tta_xy"]
            )
            self.assertFalse(marker["public_read"])
            self.assertTrue((work / TTA_MARKER_NAME).is_file())

    def test_marker_rejects_non_tta_inference_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            plan = {
                "inference_fingerprint": "abc123",
                "identity": {
                    "implementation": [
                        {"path": "/tmp/not-the-wrapper.py", "sha256": "bad"}
                    ]
                },
            }
            (work / "INFERENCE_PLAN.json").write_text(json.dumps(plan))
            (work / "INFERENCE_COMPLETE.json").write_text(
                json.dumps({"inference_fingerprint": "abc123"})
            )
            with self.assertRaisesRegex(ValueError, "not bound"):
                write_tta_marker(work, ["global"])

    def test_route_target_parser_selects_only_tta_suffixed_route(self):
        targets, routes = parse_tta_targets(
            [
                "infer",
                "--focus-contour-route",
                "focus_weak_sources_tta_xy=/tmp/weak.pt",
                "--focus-contour-route",
                "focus_lung_cells_notta=/tmp/lung.pt",
            ]
        )
        self.assertEqual(targets, {Path("/tmp/weak.pt")})
        self.assertEqual(routes, ["focus_weak_sources_tta_xy"])

    def test_route_target_parser_rejects_shared_tta_and_notta_path(self):
        with self.assertRaisesRegex(ValueError, "distinct hardlink"):
            parse_tta_targets(
                [
                    "--focus-contour-route",
                    "focus_weak_sources_tta_xy=/tmp/shared.pt",
                    "--focus-contour-route",
                    "focus_lung_cells_notta=/tmp/shared.pt",
                ]
            )

    def test_loader_leaves_unselected_focus_model_unwrapped(self):
        class Focus(torch.nn.Module):
            def forward(self, images):
                return torch.cat((images[:, :1], images[:, :1]), dim=1)

        model = Focus()
        observed = tta_loader(lambda path, device: model, {Path("other.pt")})(
            Path("checkpoint.pt"), torch.device("cpu")
        )
        self.assertIs(observed, model)


if __name__ == "__main__":
    unittest.main()
