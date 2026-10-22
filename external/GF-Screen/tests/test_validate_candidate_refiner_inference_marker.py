import json
import tempfile
import unittest
from pathlib import Path

from validate_candidate_refiner_inference_marker import sha256, validate


class InferenceMarkerValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def file(self, name, content):
        path = self.root / name
        path.write_bytes(content)
        return path

    def test_source_routed_marker_binds_every_checkpoint_and_source(self):
        classification = self.file("class.pt", b"class")
        mature = self.file("mature.pt", b"mature")
        discovery = self.file("discovery.pt", b"discovery")
        mapping = {
            "epoch105": "mature",
            "currentgf": "mature",
            "lung_cell0": "discovery",
        }
        payload = {
            "candidates": 3,
            "checkpoints": [str(classification.resolve())],
            "checkpoint_sha256": [sha256(classification)],
            "source_condition_groups": [[]],
            "contour_checkpoints": [],
            "contour_checkpoint_sha256": [],
            "contour_input_modes": [],
            "candidate_contour_checkpoints": [],
            "candidate_contour_checkpoint_sha256": [],
            "routed_candidate_contours": {
                "mature": {
                    "checkpoints": [str(mature.resolve())],
                    "checkpoint_sha256": [sha256(mature)],
                    "source_condition_groups": [[]],
                },
                "discovery": {
                    "checkpoints": [str(discovery.resolve())],
                    "checkpoint_sha256": [sha256(discovery)],
                    "source_condition_groups": [[]],
                },
            },
            "source_contour_routes": dict(sorted(mapping.items())),
        }
        checks = validate(
            payload,
            [classification],
            routed_checkpoints={"mature": [mature], "discovery": [discovery]},
            source_to_route=mapping,
            expected_candidates=3,
        )
        self.assertTrue(all(checks.values()))
        discovery.write_bytes(b"tampered")
        checks = validate(
            payload,
            [classification],
            routed_checkpoints={"mature": [mature], "discovery": [discovery]},
            source_to_route=mapping,
            expected_candidates=3,
        )
        self.assertFalse(checks["routed_checkpoints"])

    def test_focus_routed_marker_binds_modes_and_checkpoints(self):
        classification = self.file("class_focus.pt", b"class")
        mature = self.file("mature_focus.pt", b"mature-focus")
        lung = self.file("lung_focus.pt", b"lung-focus")
        mapping = {"epoch105": "mature", "lung_cell0": "lung_cells"}
        payload = {
            "candidates": 2,
            "checkpoints": [str(classification.resolve())],
            "checkpoint_sha256": [sha256(classification)],
            "source_condition_groups": [[]],
            "contour_checkpoints": [],
            "contour_checkpoint_sha256": [],
            "contour_input_modes": [],
            "candidate_contour_checkpoints": [],
            "candidate_contour_checkpoint_sha256": [],
            "routed_candidate_contours": {},
            "routed_focus_contours": {
                "mature": {
                    "checkpoints": [str(mature.resolve())],
                    "checkpoint_sha256": [sha256(mature)],
                    "input_modes": ["soft_lung_seed"],
                },
                "lung_cells": {
                    "checkpoints": [str(lung.resolve())],
                    "checkpoint_sha256": [sha256(lung)],
                    "input_modes": ["soft_lung_seed"],
                },
            },
            "source_contour_routes": dict(sorted(mapping.items())),
        }
        checks = validate(
            payload,
            [classification],
            focus_routed_checkpoints={"mature": [mature], "lung_cells": [lung]},
            focus_routed_input_mode="soft_lung_seed",
            source_to_route=mapping,
            expected_candidates=2,
        )
        self.assertTrue(all(checks.values()))
        checks = validate(
            payload,
            [classification],
            focus_routed_checkpoints={"mature": [mature], "lung_cells": [lung]},
            focus_routed_input_mode="soft_seed",
            source_to_route=mapping,
            expected_candidates=2,
        )
        self.assertFalse(checks["routed_focus_checkpoints"])

    def test_mixed_candidate_and_focus_routes_are_bound_together(self):
        classification = self.file("mixed_class.pt", b"class")
        candidate = self.file("mixed_b32.pt", b"b32")
        focus = self.file("mixed_focus.pt", b"focus")
        mapping = {
            "epoch105": "b32_mature",
            "currentgf": "b32_mature",
            "lung_cell0": "focus_discovery",
        }
        payload = {
            "candidates": 3,
            "checkpoints": [str(classification.resolve())],
            "checkpoint_sha256": [sha256(classification)],
            "source_condition_groups": [[]],
            "contour_checkpoints": [],
            "contour_checkpoint_sha256": [],
            "contour_input_modes": [],
            "candidate_contour_checkpoints": [],
            "candidate_contour_checkpoint_sha256": [],
            "routed_candidate_contours": {
                "b32_mature": {
                    "checkpoints": [str(candidate.resolve())],
                    "checkpoint_sha256": [sha256(candidate)],
                    "source_condition_groups": [[]],
                }
            },
            "routed_focus_contours": {
                "focus_discovery": {
                    "checkpoints": [str(focus.resolve())],
                    "checkpoint_sha256": [sha256(focus)],
                    "input_modes": ["soft_lung_seed"],
                }
            },
            "source_contour_routes": dict(sorted(mapping.items())),
        }
        checks = validate(
            payload,
            [classification],
            routed_checkpoints={"b32_mature": [candidate]},
            focus_routed_checkpoints={"focus_discovery": [focus]},
            focus_routed_input_mode="soft_lung_seed",
            source_to_route=mapping,
            expected_candidates=3,
        )
        self.assertTrue(all(checks.values()))

        with self.assertRaisesRegex(ValueError, "route names overlap"):
            validate(
                payload,
                [classification],
                routed_checkpoints={"shared": [candidate]},
                focus_routed_checkpoints={"shared": [focus]},
                focus_routed_input_mode="soft_lung_seed",
                source_to_route={"currentgf": "shared"},
            )

    def test_legacy_and_global_contour_markers_remain_supported(self):
        classification = self.file("class.pt", b"class")
        contour = self.file("contour.pt", b"contour")
        common = {
            "checkpoints": [str(classification.resolve())],
            "checkpoint_sha256": [sha256(classification)],
            "contour_checkpoints": [],
            "contour_checkpoint_sha256": [],
            "contour_input_modes": [],
            "routed_candidate_contours": {},
            "source_contour_routes": {},
        }
        legacy = {
            **common,
            "candidate_contour_checkpoints": [],
            "candidate_contour_checkpoint_sha256": [],
        }
        self.assertTrue(all(validate(legacy, [classification]).values()))
        candidate = {
            **common,
            "candidate_contour_checkpoints": [str(contour.resolve())],
            "candidate_contour_checkpoint_sha256": [sha256(contour)],
        }
        self.assertTrue(
            all(
                validate(
                    candidate,
                    [classification],
                    candidate_contour_checkpoints=[contour],
                ).values()
            )
        )

    def test_mixed_override_families_fail_closed(self):
        classification = self.file("class.pt", b"class")
        contour = self.file("contour.pt", b"contour")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate(
                {},
                [classification],
                contour_checkpoints=[contour],
                routed_checkpoints={"mature": [contour]},
                source_to_route={"currentgf": "mature"},
            )

    def test_classifier_fusion_identity_is_bound_to_global_focus_cache(self):
        classification = self.file("class_fusion.pt", b"class")
        contour = self.file("focus_fusion.pt", b"focus")
        fusion = self.file(
            "fusion.json",
            json.dumps({"decision": "QUALIFY_FOCUS_CLASSIFIER_FUSION"}).encode(),
        )
        payload = {
            "checkpoints": [str(classification.resolve())],
            "checkpoint_sha256": [sha256(classification)],
            "contour_checkpoints": [str(contour.resolve())],
            "contour_checkpoint_sha256": [sha256(contour)],
            "contour_input_modes": ["soft_lung_seed"],
            "candidate_contour_checkpoints": [],
            "candidate_contour_checkpoint_sha256": [],
            "routed_candidate_contours": {},
            "routed_focus_contours": {},
            "source_contour_routes": {},
            "classifier_fusion": {
                "decision": str(fusion.resolve()),
                "decision_sha256": sha256(fusion),
                "focus_input_mode": "soft_lung_seed",
                "feature_names": ["b32_logit"],
            },
        }
        checks = validate(
            payload,
            [classification],
            contour_checkpoints=[contour],
            contour_input_mode="soft_lung_seed",
            classifier_fusion=fusion,
        )
        self.assertTrue(all(checks.values()))
        fusion.write_bytes(b"tampered")
        checks = validate(
            payload,
            [classification],
            contour_checkpoints=[contour],
            contour_input_mode="soft_lung_seed",
            classifier_fusion=fusion,
        )
        self.assertFalse(checks["classifier_fusion"])
        with self.assertRaisesRegex(ValueError, "global Focus"):
            validate(payload, [classification], classifier_fusion=fusion)

    def test_routed_classifier_marker_binds_checkpoints_and_sources(self):
        classification = self.file("class_base.pt", b"base")
        mature = self.file("class_mature.pt", b"mature")
        lung = self.file("class_lung.pt", b"lung")
        mapping = {
            "epoch105": "mature",
            "currentgf": "mature",
            "lung_cell0": "lung_cells",
        }
        payload = {
            "candidates": 3,
            "checkpoints": [str(classification.resolve())],
            "checkpoint_sha256": [sha256(classification)],
            "source_condition_groups": [[]],
            "contour_checkpoints": [],
            "contour_checkpoint_sha256": [],
            "contour_input_modes": [],
            "candidate_contour_checkpoints": [],
            "candidate_contour_checkpoint_sha256": [],
            "routed_candidate_contours": {},
            "routed_focus_contours": {},
            "source_contour_routes": {},
            "routed_classifiers": {
                "mature": {
                    "checkpoints": [str(mature.resolve())],
                    "checkpoint_sha256": [sha256(mature)],
                    "source_condition_groups": [[]],
                },
                "lung_cells": {
                    "checkpoints": [str(lung.resolve())],
                    "checkpoint_sha256": [sha256(lung)],
                    "source_condition_groups": [[]],
                },
            },
            "source_classifier_routes": dict(sorted(mapping.items())),
        }
        checks = validate(
            payload,
            [classification],
            expected_candidates=3,
            routed_classifier_checkpoints={
                "mature": [mature],
                "lung_cells": [lung],
            },
            source_to_classifier_route=mapping,
        )
        self.assertTrue(all(checks.values()))
        lung.write_bytes(b"tampered")
        checks = validate(
            payload,
            [classification],
            expected_candidates=3,
            routed_classifier_checkpoints={
                "mature": [mature],
                "lung_cells": [lung],
            },
            source_to_classifier_route=mapping,
        )
        self.assertFalse(checks["routed_classifier_checkpoints"])

    def test_routed_classifier_map_must_cover_route_inventory(self):
        classification = self.file("class_map.pt", b"base")
        mature = self.file("class_map_mature.pt", b"mature")
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate(
                {},
                [classification],
                routed_classifier_checkpoints={"mature": [mature]},
                source_to_classifier_route={"lung_cell0": "lung_cells"},
            )


if __name__ == "__main__":
    unittest.main()
