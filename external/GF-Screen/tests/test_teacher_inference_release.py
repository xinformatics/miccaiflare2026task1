import json
import re
import tempfile
import unittest
from pathlib import Path

from inference_naming import prediction_name
from mark_teacher_inference_complete import THRESHOLDS, mark_complete


ROOT = Path(__file__).resolve().parents[1]
DIRECT = (ROOT / "run_focus_encoder_notta_direct_public_20260825.sh").read_text()
WATCHER = (ROOT / "watch_teacher_inference_complete_20260825.sh").read_text()


class TeacherInferenceReleaseTest(unittest.TestCase):
    def build_fixture(self, root):
        images = root / "images"
        teacher = root / "teacher"
        images.mkdir()
        names = []
        for index in range(8):
            source = images / f"case_{index:02d}_0000.nii.gz"
            source.write_bytes(f"source-{index}".encode())
            names.append(prediction_name(source.name))
        for threshold in THRESHOLDS:
            directory = teacher / f"lesion_{threshold}"
            directory.mkdir(parents=True)
            for index, name in enumerate(names):
                (directory / name).write_bytes(
                    f"prediction-{threshold}-{index}".encode()
                )
        checkpoint = root / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        audit = root / "audit.json"
        audit.write_text(json.dumps({
            "status": "PASS_WITH_EXCLUSIONS",
            "unlabeled_files": 8,
            "usable_unique_files": 2,
        }))
        return images, teacher, checkpoint, audit

    def test_exact_inventories_create_scheduling_only_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, teacher, checkpoint, audit = self.build_fixture(root)
            # A writer may have an in-flight or orphaned intermediate. It is
            # not a completed prediction and must not affect exact release.
            (teacher / "lesion_0.97" / "case_99_0000_trans.nii.gz").write_bytes(
                b"partial"
            )
            output = root / "TERMINAL.json"
            payload = mark_complete(
                images, teacher, checkpoint, audit, output, expected_count=8
            )
            self.assertEqual(
                payload["decision"],
                "TEACHER_INFERENCE_COMPLETE_GPU_RELEASE_PENDING",
            )
            self.assertFalse(payload["pseudo_materialization_complete"])
            self.assertFalse(payload["public_read"])
            self.assertFalse(payload["hidden_read"])
            self.assertTrue(all(
                item["members"] == 8
                for item in payload["threshold_inventories"].values()
            ))

    def test_missing_prediction_rejects_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, teacher, checkpoint, audit = self.build_fixture(root)
            (teacher / "lesion_0.97" / "case_07.nii.gz").unlink()
            with self.assertRaisesRegex(ValueError, "set mismatch"):
                mark_complete(
                    images, teacher, checkpoint, audit,
                    root / "TERMINAL.json", expected_count=8,
                )

    def test_direct_route_waits_for_release_not_pseudo_materialization(self):
        self.assertIn("TEACHER_INFERENCE_COMPLETE.json", DIRECT)
        self.assertIn("TEACHER_INFERENCE_COMPLETE_GPU_RELEASE_PENDING", DIRECT)
        self.assertNotIn("PSEUDO_LABELS_COMPLETE.json", DIRECT)
        self.assertIn("WEAK707_PUBLIC_GPU0_MAX_MIB=4096", DIRECT)
        self.assertIn("WEAK707_PUBLIC_GPU1_MAX_MIB=1024", DIRECT)
        self.assertIn("wait_any_gpu", (
            ROOT / "run_candidate_refiner_v3_weak707_public_20260823.sh"
        ).read_text())
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", DIRECT, flags=re.DOTALL)
        self.assertTrue(blocks)
        for index, block in enumerate(blocks):
            compile(block, f"direct-heredoc-{index}", "exec")

    def test_direct_route_uses_current_champion_dsc_first_gate(self):
        self.assertIn("fixed_public.metrics.csv", DIRECT)
        self.assertIn("select_dsc_first_nonregression.py", DIRECT)
        self.assertIn("--min-dsc-gain 0.015", DIRECT)
        self.assertIn("--min-dsc-win-rate 0.55", DIRECT)
        self.assertIn("--min-nsd-gain 0.0", DIRECT)
        self.assertIn("--min-nsd-uncertainty-bound -0.005", DIRECT)
        self.assertIn('gate.get("decision") == "PROMOTE"', DIRECT)

    def test_direct_route_binds_non_promotional_lesion_size_diagnostic(self):
        self.assertIn("analyze_lesion_errors.py", DIRECT)
        self.assertIn("direct lesion-error diagnostic identity mismatch", DIRECT)
        self.assertIn("diagnostic_only_not_used_for_promotion", DIRECT)
        self.assertLess(
            DIRECT.index("select_dsc_first_nonregression.py"),
            DIRECT.index('gate.get("decision") == "PROMOTE"'),
        )

    def test_release_watcher_is_label_and_upload_blind(self):
        lowered = WATCHER.lower()
        self.assertNotIn("validation-public", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("codabench", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn("! -name '*_trans.nii.gz'", WATCHER)


if __name__ == "__main__":
    unittest.main()
