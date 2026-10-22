import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from validate_candidate_refiner_grid_fold import validate_fold


class ValidateCandidateRefinerGridFoldTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.expected = self.root / "labels"
        self.expected.mkdir()
        for case in ("case_a", "case_b"):
            (self.expected / f"{case}.nii.gz").write_bytes(b"gt")
        self.base = self.root / "base"
        self.base.mkdir()
        self.grid = self.root / "grid.json"
        self.configuration = {
            "tag": "candidate",
            "classification_threshold": 0.75,
            "segmentation_threshold": 0.9,
            "mode": "refined",
            "source_classification_thresholds": {},
            "source_segmentation_thresholds": {},
            "source_deployment_modes": {},
        }
        self.grid.write_text(
            json.dumps({"configurations": [self.configuration]}), encoding="utf-8"
        )
        self.output_root = self.root / "output"
        self.fold = self.output_root / "candidate" / "fold0"
        self.fold.mkdir(parents=True)
        for case in ("case_a", "case_b"):
            (self.fold / f"{case}.nii.gz").write_bytes(b"prediction")
        self.audit = self.fold.with_name("fold0.components.csv")
        with self.audit.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("case", "accepted"))
            writer.writeheader()
            writer.writerow({"case": "case_a", "accepted": 1})
            writer.writerow({"case": "case_b", "accepted": 0})
        self.marker = self.fold / "CANDIDATE_REFINER_COMPLETE.json"
        self.write_marker()

    def tearDown(self):
        self.temporary.cleanup()

    def write_marker(self):
        self.marker.write_text(
            json.dumps(
                {
                    "cases": 2,
                    "candidates": 2,
                    "accepted": 1,
                    "classification_threshold": 0.75,
                    "source_classification_thresholds": {},
                    "segmentation_threshold": 0.9,
                    "fallback_segmentation_threshold": 0.85,
                    "source_segmentation_thresholds": {},
                    "source_deployment_modes": {},
                    "mode": "refined",
                    "empty_base": True,
                    "component_policy": "source_separate",
                    "gates": {
                        "min_seed_coverage": 0.0,
                        "min_refined_overlap": 0.0,
                        "min_volume_ratio": 0.0,
                        "max_volume_ratio": 1e12,
                    },
                    "base": str(self.base.resolve()),
                    "audit": str(self.audit.resolve()),
                    "materializer": "shared_case_grid_v1",
                    "configuration_grid": str(self.grid.resolve()),
                }
            ),
            encoding="utf-8",
        )

    def validate(self):
        return validate_fold(
            self.grid,
            self.output_root,
            0,
            self.expected,
            self.base,
            0.85,
            True,
            {
                "min_seed_coverage": 0.0,
                "min_refined_overlap": 0.0,
                "min_volume_ratio": 0.0,
                "max_volume_ratio": 1e12,
            },
        )

    def test_accepts_complete_fold(self):
        result = self.validate()
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["configurations"][0]["accepted"], 1)

    def test_rejects_payload_newer_than_marker(self):
        marker_time = self.marker.stat().st_mtime_ns
        os.utime(self.audit, ns=(marker_time + 1_000_000, marker_time + 1_000_000))
        with self.assertRaisesRegex(ValueError, "stale marker"):
            self.validate()

    def test_rejects_truncated_audit(self):
        self.audit.write_text("case,accepted\ncase_a,1\n", encoding="utf-8")
        self.write_marker()
        with self.assertRaisesRegex(ValueError, "candidate count mismatch"):
            self.validate()

    def test_rejects_case_coverage_mismatch(self):
        (self.fold / "case_b.nii.gz").unlink()
        self.write_marker()
        with self.assertRaisesRegex(ValueError, "output coverage mismatch"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
