from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "select_qualified_module_sources.py"


def write_saved_manifest(root: Path, *, gate: str = "PROMOTE", cases: int = 50) -> Path:
    checkpoint = root / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    metrics = root / "metrics.csv"
    metrics.write_text("case,DSC,NSD\n", encoding="utf-8")
    robust = root / "robust.json"
    robust.write_text(json.dumps({"cases": cases, "decision": gate}), encoding="utf-8")
    submission = root / "candidate.zip"
    with zipfile.ZipFile(submission, "w") as archive:
        for index in range(100):
            archive.writestr(f"case_{index:03d}.nii.gz", b"prediction")
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    manifest = root / "saved.tsv"
    fieldnames = [
        "rank", "tag", "family", "checkpoint", "glance_threshold",
        "lesion_threshold", "DSC", "NSD", "metrics", "robust", "gate",
        "submission", "sha256",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "rank": 1,
                "tag": "candidate",
                "family": "test",
                "checkpoint": checkpoint,
                "glance_threshold": 0.5,
                "lesion_threshold": 0.5,
                "DSC": 0.61,
                "NSD": 0.52,
                "metrics": metrics,
                "robust": robust,
                "gate": gate,
                "submission": submission,
                "sha256": digest,
            }
        )
    return manifest


class QualifiedModuleSourceTests(unittest.TestCase):
    def run_selector(self, root: Path, focus: Path, glance: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--base",
                str(root / "base.pt"),
                "--focus-manifest",
                str(focus),
                "--glance-manifest",
                str(glance),
                "--output",
                str(root / "selection.json"),
            ],
            capture_output=True,
            text=True,
        )

    def test_uses_qualified_focus_and_base_glance_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.pt").write_bytes(b"base")
            focus_dir = root / "focus"
            focus_dir.mkdir()
            focus = write_saved_manifest(focus_dir)
            result = self.run_selector(root, focus, root / "missing-glance.tsv")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(report["focus"]["status"], "qualified_saved_source")
            self.assertEqual(report["focus"]["submission_member_count"], 100)
            self.assertEqual(report["focus"]["lesion_threshold"], 0.5)
            self.assertEqual(
                report["glance"]["status"],
                "fallback_base_no_qualified_manifest",
            )
            self.assertEqual(report["glance"]["checkpoint"], str((root / "base.pt").resolve()))
            self.assertEqual(report["glance"]["glance_threshold"], 0.5)

    def test_rejects_gate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.pt").write_bytes(b"base")
            focus_dir = root / "focus"
            focus_dir.mkdir()
            focus = write_saved_manifest(focus_dir, gate="PROMOTE")
            robust = focus_dir / "robust.json"
            robust.write_text(json.dumps({"cases": 50, "decision": "KEEP_ANCHOR"}), encoding="utf-8")
            result = self.run_selector(root, focus, root / "missing-glance.tsv")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest/robust gate mismatch", result.stderr)

    def test_rejects_incomplete_robustness_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.pt").write_bytes(b"base")
            focus_dir = root / "focus"
            focus_dir.mkdir()
            focus = write_saved_manifest(focus_dir, cases=49)
            result = self.run_selector(root, focus, root / "missing-glance.tsv")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not cover 50 cases", result.stderr)

    def test_rejects_submission_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.pt").write_bytes(b"base")
            focus_dir = root / "focus"
            focus_dir.mkdir()
            focus = write_saved_manifest(focus_dir)
            with focus.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
                fieldnames = list(rows[0])
            rows[0]["sha256"] = "0" * 64
            with focus.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            result = self.run_selector(root, focus, root / "missing-glance.tsv")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("submission SHA-256 mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
