import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from apply_candidate_refiner import (
    apply_local_replacement,
    ensemble_refiner_batch,
    initialize_output,
    infer,
    local_target_grid,
    load_model,
    materialize,
    parse_named_checkpoint_routes,
    parse_source_contour_routes,
    parse_source_thresholds,
    prepare,
)
from candidate_refiner_model import CandidateRefiner
from prepare_candidate_refiner_patches import (
    crop_affine,
    resample_array,
    resample_seed_prompt,
)


class ApplyCandidateRefinerTests(unittest.TestCase):
    def test_infer_routes_classifiers_and_records_exact_provenance(self):
        class Refiner(torch.nn.Module):
            def __init__(self, class_logit, segmentation_logit):
                super().__init__()
                self.class_logit = class_logit
                self.segmentation_logit = segmentation_logit

            def forward(self, images):
                return (
                    torch.full((len(images),), self.class_logit),
                    torch.full(
                        (len(images), 1, *images.shape[2:]),
                        self.segmentation_logit,
                    ),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            patches = work / "patches"
            patches.mkdir(parents=True)
            records = []
            for patch_id, source in (("p0", "currentgf"), ("p1", "lung_cell0")):
                patch_path = patches / f"{patch_id}.npz"
                np.savez_compressed(
                    patch_path, image=np.zeros((3, 2, 2, 2), dtype=np.float32)
                )
                records.append(
                    {
                        "patch_id": patch_id,
                        "patch": str(patch_path),
                        "source_name": source,
                    }
                )
            (work / "manifest.json").write_text(
                json.dumps({"records": records}), encoding="utf-8"
            )
            base = root / "base.pt"
            mature = root / "mature.pt"
            lung = root / "lung.pt"
            for path in (base, mature, lung):
                path.write_bytes(path.stem.encode())

            models = {
                "base.pt": Refiner(0.0, 2.0),
                "mature.pt": Refiner(3.0, -5.0),
                "lung.pt": Refiner(-3.0, 5.0),
            }

            def load(path, _device):
                return models[Path(path).name]

            args = SimpleNamespace(
                work=work,
                checkpoint=[base],
                contour_checkpoint=[],
                candidate_contour_checkpoint=[],
                candidate_contour_route=[],
                focus_contour_route=[],
                source_contour_route=[],
                classifier_route=[f"mature={mature}", f"lung_cells={lung}"],
                source_classifier_route=[
                    "currentgf=mature",
                    "lung_cell0=lung_cells",
                ],
                classifier_fusion=None,
                batch_size=2,
                workers=0,
                uncompressed_predictions=True,
                resume=True,
            )
            with patch("apply_candidate_refiner.load_model", side_effect=load), patch(
                "apply_candidate_refiner.torch.cuda.is_available", return_value=False
            ):
                infer(args)
            observed = []
            for patch_id in ("p0", "p1"):
                prediction_path = work / "predictions" / f"{patch_id}.npz"
                with zipfile.ZipFile(prediction_path) as stored:
                    self.assertTrue(
                        all(
                            item.compress_type == zipfile.ZIP_STORED
                            for item in stored.infolist()
                        )
                    )
                with np.load(
                    prediction_path, allow_pickle=False
                ) as archive:
                    observed.append(float(archive["classification_probability"]))
                    np.testing.assert_allclose(
                        archive["segmentation_probability"],
                        float(torch.sigmoid(torch.tensor(2.0))),
                        rtol=2e-3,
                    )
            np.testing.assert_allclose(
                observed,
                torch.tensor([3.0, -3.0]).sigmoid().numpy(),
                rtol=1e-6,
            )
            marker = json.loads((work / "INFERENCE_COMPLETE.json").read_text())
            self.assertFalse(marker["prediction_cache_compressed"])
            self.assertFalse(marker["prediction_cache_fsync"])
            self.assertEqual(marker["reused_predictions"], 0)
            self.assertEqual(marker["written_predictions"], 2)
            mtimes = {
                patch_id: (work / "predictions" / f"{patch_id}.npz").stat().st_mtime_ns
                for patch_id in ("p0", "p1")
            }
            with patch("apply_candidate_refiner.load_model", side_effect=load), patch(
                "apply_candidate_refiner.torch.cuda.is_available", return_value=False
            ), patch(
                "apply_candidate_refiner.ensemble_refiner_batch",
                side_effect=AssertionError("valid resumed predictions were recomputed"),
            ):
                infer(args)
            resumed = json.loads((work / "INFERENCE_COMPLETE.json").read_text())
            self.assertEqual(resumed["reused_predictions"], 2)
            self.assertEqual(resumed["written_predictions"], 0)
            self.assertEqual(
                mtimes,
                {
                    patch_id: (work / "predictions" / f"{patch_id}.npz").stat().st_mtime_ns
                    for patch_id in ("p0", "p1")
                },
            )
            unchanged = (work / "predictions" / "p0.npz").stat().st_mtime_ns
            (work / "predictions" / "p1.npz").write_bytes(b"corrupt")
            with patch("apply_candidate_refiner.load_model", side_effect=load), patch(
                "apply_candidate_refiner.torch.cuda.is_available", return_value=False
            ):
                infer(args)
            repaired = json.loads((work / "INFERENCE_COMPLETE.json").read_text())
            self.assertEqual(repaired["reused_predictions"], 1)
            self.assertEqual(repaired["written_predictions"], 1)
            self.assertEqual(
                (work / "predictions" / "p0.npz").stat().st_mtime_ns,
                unchanged,
            )
            with np.load(
                work / "predictions" / "p1.npz", allow_pickle=False
            ) as archive:
                self.assertEqual(
                    archive["segmentation_probability"].shape, (1, 2, 2, 2)
                )
            self.assertEqual(set(marker["routed_classifiers"]), {"mature", "lung_cells"})
            self.assertEqual(
                marker["source_classifier_routes"],
                {"currentgf": "mature", "lung_cell0": "lung_cells"},
            )

    def test_infer_mixes_b32_and_focus_contour_routes_and_records_provenance(self):
        class Refiner(torch.nn.Module):
            def __init__(self, segmentation_logit):
                super().__init__()
                self.segmentation_logit = segmentation_logit

            def forward(self, images):
                return (
                    torch.full((len(images),), 2.0),
                    torch.full_like(images[:, :1], self.segmentation_logit),
                )

        class Focus(torch.nn.Module):
            candidate_refiner_input_mode = "soft_lung_seed"

            def forward(self, images):
                background = torch.zeros_like(images[:, :1])
                foreground = torch.full_like(background, -2.0)
                return torch.cat((background, foreground), dim=1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            patches = work / "patches"
            patches.mkdir(parents=True)
            records = []
            for patch_id, source in (("p0", "currentgf"), ("p1", "lung_cell0")):
                patch_path = patches / f"{patch_id}.npz"
                np.savez_compressed(
                    patch_path, image=np.zeros((3, 2, 2, 2), dtype=np.float32)
                )
                records.append(
                    {"patch_id": patch_id, "patch": str(patch_path), "source_name": source}
                )
            (work / "manifest.json").write_text(
                json.dumps({"records": records}), encoding="utf-8"
            )
            base, focus = root / "base.pt", root / "focus.pt"
            for path in (base, focus):
                path.write_bytes(path.stem.encode())

            args = SimpleNamespace(
                work=work,
                checkpoint=[base],
                contour_checkpoint=[],
                candidate_contour_checkpoint=[],
                candidate_contour_route=[f"b32={base}"],
                focus_contour_route=[f"focus={focus}"],
                source_contour_route=["currentgf=b32", "lung_cell0=focus"],
                classifier_route=[],
                source_classifier_route=[],
                classifier_fusion=None,
                batch_size=2,
                workers=0,
            )
            with patch(
                "apply_candidate_refiner.load_model", return_value=Refiner(2.0)
            ) as candidate_loader, patch(
                "apply_candidate_refiner.load_focus_contour_model", return_value=Focus()
            ), patch("apply_candidate_refiner.torch.cuda.is_available", return_value=False):
                infer(args)
            self.assertEqual(candidate_loader.call_count, 1)
            observed = []
            for patch_id in ("p0", "p1"):
                with np.load(
                    work / "predictions" / f"{patch_id}.npz", allow_pickle=False
                ) as archive:
                    observed.append(float(archive["segmentation_probability"].flat[0]))
            np.testing.assert_allclose(
                observed, torch.tensor([2.0, -2.0]).sigmoid().numpy(), rtol=2e-3
            )
            marker = json.loads((work / "INFERENCE_COMPLETE.json").read_text())
            self.assertEqual(marker["contour_source"], "mixed_routed_candidate_refiner_focus")
            self.assertEqual(set(marker["routed_candidate_contours"]), {"b32"})
            self.assertEqual(set(marker["routed_focus_contours"]), {"focus"})
            self.assertEqual(
                marker["source_contour_routes"],
                {"currentgf": "b32", "lung_cell0": "focus"},
            )

    def test_infer_applies_and_records_qualified_focus_classifier_fusion(self):
        class Classifier(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images)), torch.zeros_like(images[:, :1])

        class Focus(torch.nn.Module):
            candidate_refiner_input_mode = "soft_lung_seed"

            def forward(self, images):
                background = torch.zeros_like(images[:, :1])
                foreground = torch.full_like(background, float(torch.logit(torch.tensor(0.9))))
                return torch.cat((background, foreground), dim=1)

        from focus_classifier_fusion import DEPLOYMENT_FEATURE_NAMES

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            patches = work / "patches"
            patches.mkdir(parents=True)
            image = np.zeros((3, 4, 4, 4), dtype=np.float32)
            image[2, 1:3, 1:3, 1:3] = 1
            patch_path = patches / "p0.npz"
            np.savez_compressed(patch_path, image=image)
            record = {
                "patch_id": "p0",
                "patch": str(patch_path),
                "source_name": "currentgf",
                "seed_mm3": 27.0,
                "max_source_support": 1.0,
                "mean_source_support": 1.0,
            }
            (work / "manifest.json").write_text(
                json.dumps({"records": [record]}), encoding="utf-8"
            )
            classification_checkpoint = root / "class.pt"
            contour_checkpoint = root / "focus.pt"
            classification_checkpoint.write_bytes(b"class")
            contour_checkpoint.write_bytes(b"focus")
            count = len(DEPLOYMENT_FEATURE_NAMES)
            coefficient = [0.0] * count
            coefficient[1] = 2.0  # Focus foreground maximum.
            fusion = root / "fusion.json"
            fusion.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "selection_basis": "cross-fitted exact707 grouped held-out patch classification only",
                        "decision": "QUALIFY_FOCUS_CLASSIFIER_FUSION",
                        "checks": {
                            "average_precision_gain": True,
                            "f0p5_gain": True,
                            "source_nonregression": True,
                        },
                        "model": {"type": "standardized logistic regression", "C": 0.1},
                        "index_sha256": "a" * 64,
                        "provenance": [
                            {
                                "fold": fold,
                                "b32_prediction_sha256": "b" * 64,
                                "focus_prediction_sha256": "c" * 64,
                            }
                            for fold in range(5)
                        ],
                        "public_read": False,
                        "hidden_read": False,
                        "submission_created": False,
                        "feature_names": list(DEPLOYMENT_FEATURE_NAMES),
                        "focus_input_mode": "soft_lung_seed",
                        "deployment_model": {
                            "scaler_mean": [0.0] * count,
                            "scaler_scale": [1.0] * count,
                            "coefficient": coefficient,
                            "intercept": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                work=work,
                checkpoint=[classification_checkpoint],
                contour_checkpoint=[contour_checkpoint],
                candidate_contour_checkpoint=[],
                candidate_contour_route=[],
                focus_contour_route=[],
                source_contour_route=[],
                classifier_fusion=fusion,
                batch_size=1,
                workers=0,
            )
            with patch("apply_candidate_refiner.load_model", return_value=Classifier()), patch(
                "apply_candidate_refiner.load_focus_contour_model", return_value=Focus()
            ), patch("apply_candidate_refiner.torch.cuda.is_available", return_value=False):
                infer(args)
            with np.load(work / "predictions" / "p0.npz", allow_pickle=False) as archive:
                probability = float(archive["classification_probability"])
            self.assertAlmostEqual(probability, float(torch.sigmoid(torch.tensor(1.8))), places=5)
            marker = json.loads((work / "INFERENCE_COMPLETE.json").read_text())
            self.assertEqual(marker["classifier_fusion"]["decision"], str(fusion.resolve()))
            self.assertEqual(marker["classifier_fusion"]["focus_input_mode"], "soft_lung_seed")

    def test_focus_contours_can_be_routed_by_proposal_family(self):
        class Refiner(torch.nn.Module):
            def forward(self, images):
                return torch.full((len(images),), 2.0), torch.zeros_like(images[:, :1])

        class Focus(torch.nn.Module):
            candidate_refiner_input_mode = "soft_lung_seed"

            def __init__(self, foreground_logit):
                super().__init__()
                self.foreground_logit = foreground_logit

            def forward(self, images):
                self.channels = images.shape[1]
                background = torch.zeros_like(images[:, :1])
                foreground = torch.full_like(background, self.foreground_logit)
                return torch.cat((background, foreground), dim=1)

        mature, discovery = Focus(2.0), Focus(-2.0)
        classification, segmentation = ensemble_refiner_batch(
            [Refiner()],
            torch.zeros(3, 3, 2, 2, 2),
            source_names=["currentgf", "lung_cell0", "epoch105"],
            focus_contour_models_by_route={
                "mature": [mature],
                "discovery": [discovery],
            },
            source_contour_routes={
                "currentgf": "mature",
                "epoch105": "mature",
                "lung_cell0": "discovery",
            },
        )
        torch.testing.assert_close(
            classification, torch.full((3,), torch.sigmoid(torch.tensor(2.0)))
        )
        torch.testing.assert_close(
            segmentation[:, 0, 0, 0, 0],
            torch.tensor([2.0, -2.0, 2.0]).sigmoid(),
        )
        self.assertEqual(mature.channels, 3)
        self.assertEqual(discovery.channels, 3)

    def test_candidate_and_focus_contours_can_share_one_complete_route_map(self):
        class Refiner(torch.nn.Module):
            def __init__(self, segmentation_logit):
                super().__init__()
                self.segmentation_logit = segmentation_logit

            def forward(self, images):
                return (
                    torch.full((len(images),), 2.0),
                    torch.full_like(images[:, :1], self.segmentation_logit),
                )

        class Focus(torch.nn.Module):
            candidate_refiner_input_mode = "soft_lung_seed"

            def forward(self, images):
                background = torch.zeros_like(images[:, :1])
                foreground = torch.full_like(background, -2.0)
                return torch.cat((background, foreground), dim=1)

        classification, segmentation = ensemble_refiner_batch(
            [Refiner(-5.0)],
            torch.zeros(3, 3, 2, 2, 2),
            source_names=["currentgf", "lung_cell0", "epoch105"],
            candidate_contour_models_by_route={"b32_mature": [Refiner(2.0)]},
            focus_contour_models_by_route={"focus_discovery": [Focus()]},
            source_contour_routes={
                "currentgf": "b32_mature",
                "epoch105": "b32_mature",
                "lung_cell0": "focus_discovery",
            },
        )
        torch.testing.assert_close(
            classification, torch.full((3,), torch.sigmoid(torch.tensor(2.0)))
        )
        torch.testing.assert_close(
            segmentation[:, 0, 0, 0, 0],
            torch.tensor([2.0, -2.0, 2.0]).sigmoid(),
        )

    def test_mixed_contour_routes_require_disjoint_route_names(self):
        class Refiner(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images)), torch.zeros_like(images[:, :1])

        with self.assertRaisesRegex(ValueError, "route names overlap"):
            ensemble_refiner_batch(
                [Refiner()],
                torch.zeros(1, 3, 2, 2, 2),
                source_names=["currentgf"],
                candidate_contour_models_by_route={"shared": [Refiner()]},
                focus_contour_models_by_route={"shared": [Refiner()]},
                source_contour_routes={"currentgf": "shared"},
            )

    def test_candidate_contours_can_be_routed_by_proposal_source(self):
        class Refiner(torch.nn.Module):
            def __init__(self, class_logit, segmentation_logit):
                super().__init__()
                self.class_logit = class_logit
                self.segmentation_logit = segmentation_logit

            def forward(self, images):
                return (
                    torch.full((len(images),), self.class_logit),
                    torch.full(
                        (len(images), 1, *images.shape[2:]),
                        self.segmentation_logit,
                    ),
                )

        images = torch.zeros(3, 3, 2, 2, 2)
        classification, segmentation = ensemble_refiner_batch(
            [Refiner(3.0, -5.0)],
            images,
            source_names=["currentgf", "lung_cell0", "epoch105"],
            candidate_contour_models_by_route={
                "mature": [Refiner(-4.0, 2.0)],
                "discovery": [Refiner(-4.0, -2.0)],
            },
            source_contour_routes={
                "currentgf": "mature",
                "epoch105": "mature",
                "lung_cell0": "discovery",
            },
        )
        torch.testing.assert_close(
            classification, torch.full((3,), torch.sigmoid(torch.tensor(3.0)))
        )
        expected = torch.tensor([2.0, -2.0, 2.0]).sigmoid()
        torch.testing.assert_close(
            segmentation[:, 0, 0, 0, 0], expected
        )

    def test_classifiers_can_be_routed_while_contours_remain_global(self):
        class Refiner(torch.nn.Module):
            def __init__(self, class_logit, segmentation_logit):
                super().__init__()
                self.class_logit = class_logit
                self.segmentation_logit = segmentation_logit

            def forward(self, images):
                return (
                    torch.full((len(images),), self.class_logit),
                    torch.full(
                        (len(images), 1, *images.shape[2:]),
                        self.segmentation_logit,
                    ),
                )

        images = torch.zeros(3, 3, 2, 2, 2)
        classification, segmentation = ensemble_refiner_batch(
            [Refiner(0.0, 2.0)],
            images,
            source_names=["currentgf", "lung_cell0", "epoch105"],
            classifier_models_by_route={
                "mature": [Refiner(3.0, -5.0)],
                "lung_cells": [Refiner(-3.0, 5.0)],
            },
            source_classifier_routes={
                "currentgf": "mature",
                "epoch105": "mature",
                "lung_cell0": "lung_cells",
            },
        )
        torch.testing.assert_close(
            classification,
            torch.tensor([3.0, -3.0, 3.0]).sigmoid(),
        )
        torch.testing.assert_close(
            segmentation,
            torch.full_like(segmentation, torch.sigmoid(torch.tensor(2.0))),
        )

    def test_routed_classifier_coverage_fails_closed(self):
        class Refiner(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images)), torch.zeros_like(images[:, :1])

        with self.assertRaisesRegex(ValueError, "invalid routed classifier coverage"):
            ensemble_refiner_batch(
                [Refiner()],
                torch.zeros(1, 3, 2, 2, 2),
                source_names=["currentgf"],
                classifier_models_by_route={"mature": [Refiner()]},
                source_classifier_routes={"epoch105": "mature"},
            )

    def test_routed_contour_coverage_and_route_parsing_fail_closed(self):
        class Refiner(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images)), torch.zeros_like(images[:, :1])

        with self.assertRaisesRegex(ValueError, "invalid routed contour coverage"):
            ensemble_refiner_batch(
                [Refiner()],
                torch.zeros(1, 3, 2, 2, 2),
                source_names=["currentgf"],
                candidate_contour_models_by_route={"mature": [Refiner()]},
                source_contour_routes={"epoch105": "mature"},
            )
        routes = parse_named_checkpoint_routes(
            ["mature=/a.pt", "mature=/b.pt", "discovery=/c.pt"]
        )
        self.assertEqual(routes["mature"], [Path("/a.pt"), Path("/b.pt")])
        self.assertEqual(
            parse_source_contour_routes(
                ["currentgf=mature", "lung_cell0=discovery"]
            ),
            {"currentgf": "mature", "lung_cell0": "discovery"},
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_source_contour_routes(
                ["currentgf=mature", "currentgf=discovery"]
            )

    def test_candidate_classifier_and_contour_checkpoints_can_be_decoupled(self):
        class Refiner(torch.nn.Module):
            def __init__(self, class_logit, segmentation_logit):
                super().__init__()
                self.class_logit = class_logit
                self.segmentation_logit = segmentation_logit

            def forward(self, images):
                return (
                    torch.full((len(images),), self.class_logit),
                    torch.full(
                        (len(images), 1, *images.shape[2:]),
                        self.segmentation_logit,
                    ),
                )

        images = torch.zeros(2, 3, 3, 3, 3)
        classifier = Refiner(class_logit=3.0, segmentation_logit=-4.0)
        contour = Refiner(class_logit=-5.0, segmentation_logit=2.0)
        classification, segmentation = ensemble_refiner_batch(
            [classifier], images, candidate_contour_models=[contour]
        )
        torch.testing.assert_close(
            classification, torch.full((2,), torch.sigmoid(torch.tensor(3.0)))
        )
        torch.testing.assert_close(
            segmentation,
            torch.full(
                (2, 1, 3, 3, 3), torch.sigmoid(torch.tensor(2.0))
            ),
        )

    def test_two_contour_override_families_fail_closed(self):
        class Refiner(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images)), torch.zeros_like(images[:, :1])

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ensemble_refiner_batch(
                [Refiner()],
                torch.zeros(1, 3, 2, 2, 2),
                contour_models=[Refiner()],
                candidate_contour_models=[Refiner()],
            )

    def test_source_conditioned_checkpoint_round_trip_binds_input_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "conditioned.pt"
            groups = [
                "epoch105",
                "currentgf",
                "autopet_weak",
                "lndb_weak",
                "lung_cells",
            ]
            original = CandidateRefiner(in_channels=8, base_channels=4)
            torch.save(
                {
                    "state_dict": original.state_dict(),
                    "config": {
                        "base_channels": 4,
                        "source_conditioning": True,
                        "source_condition_groups": groups,
                    },
                },
                checkpoint,
            )
            restored = load_model(checkpoint, torch.device("cpu"))
            self.assertEqual(restored.encoder0.conv1.in_channels, 8)
            self.assertEqual(
                tuple(restored.candidate_refiner_source_condition_groups),
                tuple(groups),
            )
            images = torch.zeros(2, 3, 8, 8, 8)
            classification, segmentation = ensemble_refiner_batch(
                [restored], images, source_names=["epoch105", "lung_cell6"]
            )
            self.assertEqual(tuple(classification.shape), (2,))
            self.assertEqual(tuple(segmentation.shape), (2, 1, 8, 8, 8))

    def test_prompt_masked_checkpoint_round_trip_binds_classifier_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "prompted.pt"
            original = CandidateRefiner(
                base_channels=4, prompt_masked_classifier=True
            )
            torch.save(
                {
                    "state_dict": original.state_dict(),
                    "config": {
                        "base_channels": 4,
                        "prompt_masked_classifier": True,
                        "prompt_channel": 2,
                    },
                },
                checkpoint,
            )
            restored = load_model(checkpoint, torch.device("cpu"))
            self.assertTrue(restored.prompt_masked_classifier)
            self.assertTrue(restored.candidate_refiner_prompt_masked_classifier)
            self.assertEqual(restored.prompt_channel, 2)
            self.assertEqual(restored.candidate_refiner_prompt_channel, 2)
            self.assertEqual(restored.classifier[-1].in_features, 128)
            images = torch.rand(2, 3, 16, 16, 16)
            images[:, 2, 4:12, 4:12, 4:12] = 1
            classification, segmentation = restored(images)
            self.assertEqual(tuple(classification.shape), (2,))
            self.assertEqual(tuple(segmentation.shape), (2, 1, 16, 16, 16))

    def test_prompt_masked_checkpoint_rejects_input_semantic_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "wrong_prompt_channel.pt"
            original = CandidateRefiner(
                base_channels=4, prompt_masked_classifier=True, prompt_channel=1
            )
            torch.save(
                {
                    "state_dict": original.state_dict(),
                    "config": {
                        "base_channels": 4,
                        "prompt_masked_classifier": True,
                        "prompt_channel": 1,
                    },
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "fixed .* input layout"):
                load_model(checkpoint, torch.device("cpu"))

    def test_source_conditioned_classifier_receives_matching_one_hot_channels(self):
        class Conditioned(torch.nn.Module):
            candidate_refiner_source_condition_groups = (
                "epoch105",
                "currentgf",
                "autopet_weak",
                "lndb_weak",
                "lung_cells",
            )

            def forward(self, images):
                self.observed = images.detach().clone()
                return torch.zeros(len(images)), torch.zeros_like(images[:, :1])

        images = torch.zeros(2, 3, 2, 2, 2)
        model = Conditioned()
        ensemble_refiner_batch(
            [model], images, source_names=["currentgf", "lung_cell3"]
        )
        self.assertEqual(tuple(model.observed.shape), (2, 8, 2, 2, 2))
        self.assertTrue(torch.all(model.observed[0, 4] == 1))
        self.assertTrue(torch.all(model.observed[1, 7] == 1))
        with self.assertRaisesRegex(ValueError, "source names are required"):
            ensemble_refiner_batch([model], images)

    def test_focus_contour_can_replace_segmentation_without_changing_classifier(self):
        class Small(torch.nn.Module):
            def forward(self, images):
                batch = len(images)
                return (
                    torch.full((batch,), 2.0),
                    torch.full((batch, 1, *images.shape[2:]), -2.0),
                )

        class Focus(torch.nn.Module):
            def forward(self, images):
                self.observed_channels = images.shape[1]
                logits = torch.zeros((len(images), 2, *images.shape[2:]))
                logits[:, 1] = 4.0
                return logits

        images = torch.zeros((2, 3, 4, 4, 4))
        small = Small()
        focus = Focus()
        classification, legacy = ensemble_refiner_batch([small], images)
        hybrid_classification, hybrid = ensemble_refiner_batch(
            [small], images, [focus]
        )
        torch.testing.assert_close(hybrid_classification, classification)
        self.assertTrue(torch.all(hybrid > legacy))
        self.assertEqual(focus.observed_channels, 1)

    def test_prompt_focus_receives_soft_tissue_and_seed_channels(self):
        class Small(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images)), torch.zeros_like(images[:, :1])

        class PromptFocus(torch.nn.Module):
            candidate_refiner_input_mode = "soft_seed"

            def __init__(self):
                super().__init__()
                self.observed = None

            def forward(self, images):
                self.observed = images.detach().clone()
                return torch.cat((torch.zeros_like(images[:, :1]), images[:, 1:2]), dim=1)

        images = torch.zeros(2, 3, 4, 4, 4)
        images[:, 0] = 0.25
        images[:, 1] = 0.75
        images[:, 2, 1:3, 1:3, 1:3] = 1
        focus = PromptFocus()
        _, contour = ensemble_refiner_batch([Small()], images, [focus])
        self.assertEqual(tuple(focus.observed.shape), (2, 2, 4, 4, 4))
        self.assertTrue(torch.all(focus.observed[:, 0] == 0.25))
        self.assertTrue(torch.equal(focus.observed[:, 1], images[:, 2]))
        self.assertGreater(float(contour[:, :, 1:3, 1:3, 1:3].mean()), 0.5)

    def test_dual_window_prompt_focus_receives_all_cached_channels(self):
        class Small(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images)), torch.zeros_like(images[:, :1])

        class PromptFocus(torch.nn.Module):
            candidate_refiner_input_mode = "soft_lung_seed"

            def __init__(self):
                super().__init__()
                self.observed = None

            def forward(self, images):
                self.observed = images.detach().clone()
                return torch.cat((torch.zeros_like(images[:, :1]), images[:, 2:3]), dim=1)

        images = torch.zeros(2, 3, 4, 4, 4)
        images[:, 0] = 0.25
        images[:, 1] = 0.75
        images[:, 2, 1:3, 1:3, 1:3] = 1
        focus = PromptFocus()
        _, contour = ensemble_refiner_batch([Small()], images, [focus])
        self.assertEqual(tuple(focus.observed.shape), (2, 3, 4, 4, 4))
        self.assertTrue(torch.equal(focus.observed, images))
        self.assertGreater(float(contour[:, :, 1:3, 1:3, 1:3].mean()), 0.5)

    def test_source_threshold_parser_is_strict(self):
        self.assertEqual(
            parse_source_thresholds(["epoch105=0.4", "currentgf=0.7"]),
            {"epoch105": 0.4, "currentgf": 0.7},
        )
        with self.assertRaises(ValueError):
            parse_source_thresholds(["missing_separator"])
        with self.assertRaises(ValueError):
            parse_source_thresholds(["source=1.0"])
        with self.assertRaises(ValueError):
            parse_source_thresholds(["source=0.4", "source=0.5"])

    def test_source_thresholds_require_known_source_separated_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            manifest_path = work / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "component_policy": "union",
                        "source_names": ["epoch105", "currentgf"],
                        "records": [],
                        "case_records": [],
                    }
                )
            )
            with self.assertRaisesRegex(
                ValueError, "require a source-separated manifest"
            ):
                materialize(
                    SimpleNamespace(
                        work=work,
                        source_classification_threshold=["currentgf=0.6"],
                    )
                )

            manifest_path.write_text(
                json.dumps(
                    {
                        "component_policy": "source_separate",
                        "source_names": ["epoch105", "currentgf"],
                        "records": [],
                        "case_records": [],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "unknown sources"):
                materialize(
                    SimpleNamespace(
                        work=work,
                        source_classification_threshold=["other=0.6"],
                    )
                )

    def test_empty_base_mode_drops_base_without_mutating_it(self):
        base = np.zeros((4, 4, 4), dtype=bool)
        base[1, 1, 1] = True
        empty = initialize_output(base, True)
        copied = initialize_output(base, False)
        self.assertFalse(empty.any())
        self.assertTrue(copied[1, 1, 1])
        copied[1, 1, 1] = False
        self.assertTrue(base[1, 1, 1])

    def test_local_round_trip_contains_patch(self):
        source_affine = np.diag((2.0, 2.0, 3.0, 1.0))
        source_shape = (40, 42, 30)
        center = np.asarray((40.0, 42.0, 45.0))
        patch_shape = (16, 16, 12)
        patch_affine = crop_affine(center, patch_shape, (1.5, 1.5, 3.0))
        patch = np.zeros(patch_shape, dtype=np.float32)
        patch[5:11, 5:11, 3:9] = 1.0
        slices, local_affine, local_shape = local_target_grid(
            patch_shape, patch_affine, source_shape, source_affine
        )
        mapped = resample_array(
            patch, patch_affine, local_affine, local_shape, 0, 0.0
        )
        self.assertGreater(int(mapped.sum()), 0)
        self.assertTrue(all(axis.stop > axis.start for axis in slices))
        self.assertTrue(all(value > 0 for value in local_shape))

    def test_local_replacement_preserves_base_outside_crop(self):
        output = np.zeros((8, 8, 8), dtype=bool)
        output[1:7, 3:5, 3:5] = True
        slices = (slice(2, 6), slice(2, 6), slice(2, 6))
        base_local = output[slices].copy()
        refined = np.zeros((4, 4, 4), dtype=bool)
        refined[1:3, 1:3, 1:3] = True
        apply_local_replacement(output, slices, base_local, refined)
        self.assertTrue(output[1, 3, 3])
        self.assertTrue(output[6, 3, 3])
        self.assertFalse(output[2, 3, 3])
        self.assertTrue(output[3, 3, 3])

    def test_prepare_local_bbox_seed_matches_full_volume_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            proposals = root / "proposals"
            base_dir = root / "base"
            work = root / "work"
            for directory in (images, proposals, base_dir):
                directory.mkdir()
            affine = np.diag((1.2, 1.7, 2.1, 1.0))
            shape = (30, 28, 26)
            image = np.full(shape, 20.0, dtype=np.float32)
            proposal = np.zeros(shape, dtype=np.uint8)
            proposal[9:15, 7:13, 11:18] = 1
            base = np.zeros(shape, dtype=np.uint8)
            nib.save(nib.Nifti1Image(image, affine), images / "case_0000.nii.gz")
            nib.save(nib.Nifti1Image(proposal, affine), proposals / "case.nii.gz")
            nib.save(nib.Nifti1Image(base, affine), base_dir / "case.nii.gz")
            args = SimpleNamespace(
                    images=images,
                    proposal=[f"source={proposals}"],
                    base=base_dir,
                    only_disconnected=False,
                    only_connected=False,
                    work=work,
                    shape="20,20,20",
                    spacing="1.5,1.5,1.5",
                    min_mm3=1.0,
                    max_mm3=1e12,
                    bbox_margin_mm=0.0,
                    tile_stride_mm=0.0,
                    case_limit=None,
                    resume=False,
                )
            prepare(args)
            manifest = json.loads((work / "manifest.json").read_text())
            self.assertEqual(manifest["candidates"], 1)
            record = manifest["records"][0]
            with np.load(record["patch"], allow_pickle=False) as archive:
                observed = archive["image"][2].astype(np.float32)
                patch_affine = archive["patch_affine"]
            expected, fallback = resample_seed_prompt(
                proposal, affine, patch_affine, (20, 20, 20), 1
            )
            self.assertFalse(fallback)
            np.testing.assert_allclose(observed, expected, atol=5e-4, rtol=0)
            original = Path(record["patch"])
            original_bytes = original.read_bytes()
            fixed_ns = 1_700_000_000_000_000_000
            os.utime(original, ns=(fixed_ns, fixed_ns))
            args.resume = True
            prepare(args)
            self.assertEqual(original.stat().st_mtime_ns, fixed_ns)
            self.assertEqual(original.read_bytes(), original_bytes)
            resumed = json.loads((work / "manifest.json").read_text())
            self.assertEqual(resumed["reused_patches"], 1)
            self.assertEqual(resumed["written_patches"], 0)

    def test_source_separate_prepare_and_materialize_share_component_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            proposal_a = root / "proposal_a"
            proposal_b = root / "proposal_b"
            base_dir = root / "base"
            work = root / "work"
            output = root / "output"
            for directory in (images, proposal_a, proposal_b, base_dir):
                directory.mkdir()
            affine = np.eye(4)
            shape = (20, 20, 20)
            image = np.zeros(shape, dtype=np.float32)
            first = np.zeros(shape, dtype=np.uint8)
            second = np.zeros(shape, dtype=np.uint8)
            first[3:5, 4:6, 4:6] = 1
            second[5:7, 4:6, 4:6] = 1
            image_path = images / "case_0000.nii.gz"
            first_path = proposal_a / "case.nii.gz"
            second_path = proposal_b / "case.nii.gz"
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(first, affine), first_path)
            nib.save(nib.Nifti1Image(second, affine), second_path)
            nib.save(
                nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine),
                base_dir / "case.nii.gz",
            )

            prepare(
                SimpleNamespace(
                    images=images,
                    proposal=[f"a={proposal_a}", f"b={proposal_b}"],
                    base=base_dir,
                    only_disconnected=False,
                    only_connected=False,
                    work=work,
                    shape="12,12,12",
                    spacing="1,1,1",
                    min_mm3=1.0,
                    max_mm3=1e12,
                    bbox_margin_mm=0.0,
                    tile_stride_mm=0.0,
                    case_limit=None,
                    component_policy="source_separate",
                )
            )
            manifest = json.loads((work / "manifest.json").read_text())
            self.assertEqual(manifest["component_policy"], "source_separate")
            self.assertEqual(manifest["candidates"], 2)
            self.assertEqual(
                [record["source_name"] for record in manifest["records"]],
                ["a", "b"],
            )
            for record in manifest["records"]:
                with np.load(record["patch"], allow_pickle=False) as archive:
                    self.assertAlmostEqual(float(archive["image"][2].max()), 1.0)
            (work / "predictions").mkdir()
            for record in manifest["records"]:
                np.savez_compressed(
                    work / "predictions" / f"{record['patch_id']}.npz",
                    classification_probability=np.asarray(0.99, dtype=np.float32),
                    segmentation_probability=np.zeros(
                        (1, 12, 12, 12), dtype=np.float16
                    ),
                )

            materialize(
                SimpleNamespace(
                    work=work,
                    base=base_dir,
                    output=output,
                    empty_base=True,
                    classification_threshold=0.5,
                    segmentation_threshold=0.5,
                    mode="seed",
                    min_seed_coverage=0.0,
                    min_refined_overlap=0.0,
                    min_volume_ratio=0.0,
                    max_volume_ratio=100.0,
                    min_base_dice=0.0,
                    min_base_seed_coverage=0.0,
                    source_classification_threshold=None,
                )
            )
            prediction = np.asanyarray(
                nib.load(output / "case.nii.gz").dataobj
            ).astype(bool)
            np.testing.assert_array_equal(prediction, first | second)
            audit = (output.with_name(output.name + ".components.csv")).read_text()
            self.assertIn("case__c0001", audit)
            self.assertIn("case__c0002", audit)

            gated_output = root / "gated_output"
            materialize(
                SimpleNamespace(
                    work=work,
                    base=base_dir,
                    output=gated_output,
                    empty_base=True,
                    classification_threshold=0.5,
                    source_classification_threshold=["b=0.995"],
                    segmentation_threshold=0.5,
                    mode="seed",
                    min_seed_coverage=0.0,
                    min_refined_overlap=0.0,
                    min_volume_ratio=0.0,
                    max_volume_ratio=100.0,
                    min_base_dice=0.0,
                    min_base_seed_coverage=0.0,
                )
            )
            gated = np.asanyarray(
                nib.load(gated_output / "case.nii.gz").dataobj
            ).astype(bool)
            np.testing.assert_array_equal(gated, first.astype(bool))

    def test_tiled_materialization_unions_accepted_tile_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            proposals = root / "proposals"
            base_dir = root / "base"
            work = root / "work"
            output = root / "output"
            for directory in (images, proposals, base_dir, work / "predictions"):
                directory.mkdir(parents=True)
            affine = np.eye(4)
            shape = (20, 20, 20)
            image = np.zeros(shape, dtype=np.float32)
            proposal = np.zeros(shape, dtype=np.uint8)
            proposal[5:7, 5:7, 5:7] = 1
            base = np.zeros(shape, dtype=np.uint8)
            image_path = images / "case_0000.nii.gz"
            proposal_path = proposals / "case.nii.gz"
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(proposal, affine), proposal_path)
            nib.save(nib.Nifti1Image(base, affine), base_dir / "case.nii.gz")

            records = []
            for tile_index, center in enumerate((6.0, 12.0)):
                patch_id = f"case__c0001__t{tile_index:03d}"
                patch_path = work / f"{patch_id}.npz"
                patch_affine = crop_affine(
                    np.asarray((center, 6.0, 6.0)), (8, 8, 8), (1.0, 1.0, 1.0)
                )
                np.savez_compressed(patch_path, patch_affine=patch_affine)
                segmentation = np.zeros((1, 8, 8, 8), dtype=np.float16)
                segmentation[0, 3:5, 3:5, 3:5] = 1
                np.savez_compressed(
                    work / "predictions" / f"{patch_id}.npz",
                    classification_probability=np.asarray(0.99, dtype=np.float32),
                    segmentation_probability=segmentation,
                )
                records.append(
                    {
                        "patch_id": patch_id,
                        "case": "case",
                        "component_id": 1,
                        "tile_index": tile_index,
                        "seed_mm3": 8.0,
                        "patch": str(patch_path),
                    }
                )
            manifest = {
                "tile_stride_mm": 6.0,
                "records": records,
                "case_records": [
                    {
                        "case": "case",
                        "image": str(image_path),
                        "source_proposals": [str(proposal_path)],
                    }
                ],
            }
            (work / "manifest.json").write_text(json.dumps(manifest))
            materialize(
                SimpleNamespace(
                    work=work,
                    base=base_dir,
                    output=output,
                    empty_base=False,
                    classification_threshold=0.5,
                    segmentation_threshold=0.5,
                    mode="refined",
                    min_seed_coverage=0.0,
                    min_refined_overlap=0.0,
                    min_volume_ratio=0.0,
                    max_volume_ratio=100.0,
                    min_base_dice=0.0,
                    min_base_seed_coverage=0.0,
                )
            )
            prediction = np.asanyarray(nib.load(output / "case.nii.gz").dataobj) > 0
            self.assertGreater(int(prediction[:10].sum()), 0)
            self.assertGreater(int(prediction[10:].sum()), 0)

            union_output = root / "union_output"
            materialize(
                SimpleNamespace(
                    work=work,
                    base=base_dir,
                    output=union_output,
                    empty_base=True,
                    classification_threshold=0.5,
                    segmentation_threshold=0.5,
                    mode="seed_union",
                    min_seed_coverage=0.0,
                    min_refined_overlap=0.0,
                    min_volume_ratio=0.0,
                    max_volume_ratio=100.0,
                    min_base_dice=0.0,
                    min_base_seed_coverage=0.0,
                    source_classification_threshold=None,
                )
            )
            union_prediction = np.asanyarray(
                nib.load(union_output / "case.nii.gz").dataobj
            ) > 0
            self.assertTrue(np.all(union_prediction[proposal > 0]))
            self.assertTrue(np.all(union_prediction[prediction]))

            intersection_output = root / "intersection_output"
            materialize(
                SimpleNamespace(
                    work=work,
                    base=base_dir,
                    output=intersection_output,
                    empty_base=True,
                    classification_threshold=0.5,
                    segmentation_threshold=0.5,
                    mode="seed_intersection",
                    min_seed_coverage=0.0,
                    min_refined_overlap=0.0,
                    min_volume_ratio=0.0,
                    max_volume_ratio=100.0,
                    min_base_dice=0.0,
                    min_base_seed_coverage=0.0,
                    source_classification_threshold=None,
                )
            )
            intersection_prediction = np.asanyarray(
                nib.load(intersection_output / "case.nii.gz").dataobj
            ) > 0
            np.testing.assert_array_equal(
                intersection_prediction,
                prediction & (proposal > 0),
            )


if __name__ == "__main__":
    unittest.main()
