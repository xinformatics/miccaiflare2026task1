import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def make_zip(path: Path, payload: bytes) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("FLARE-Val_0001.nii.gz", payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_saved_manifest(
    path: Path, tag: str, dsc: float, nsd: float, archive: Path, digest: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "rank", "tag", "family", "checkpoint", "glance_threshold",
        "lesion_threshold", "DSC", "NSD", "metrics", "robust", "gate",
        "submission", "sha256",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "rank": 1,
                "tag": tag,
                "family": "focus",
                "checkpoint": "checkpoint.pt",
                "glance_threshold": 0.5,
                "lesion_threshold": 0.5,
                "DSC": dsc,
                "NSD": nsd,
                "metrics": "metrics.json",
                "robust": "robust.json",
                "gate": "PROMOTE",
                "submission": archive,
                "sha256": digest,
            }
        )


class SubmissionQueueRefreshAndRankTests(unittest.TestCase):
    def test_refresh_discovers_manifests_and_deduplicates_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_sweeps = root / "public_sweeps"
            first_zip = root / "first.zip"
            second_zip = root / "second.zip"
            first_sha = make_zip(first_zip, b"first")
            second_sha = make_zip(second_zip, b"second")
            first_manifest = public_sweeps / "one" / "saved_submission.tsv"
            second_manifest = public_sweeps / "two" / "saved_focus_submission.tsv"
            write_saved_manifest(first_manifest, "first", 0.71, 0.64, first_zip, first_sha)
            write_saved_manifest(second_manifest, "second", 0.72, 0.60, second_zip, second_sha)
            ready = root / "READY.tsv"
            audit = root / "READY.audit.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "refresh_all_saved_candidates_20260822.py"),
                    "--manifest-root", str(public_sweeps),
                    "--manifest", str(first_manifest),
                    "--submitted", str(root / "submitted.tsv"),
                    "--output", str(ready),
                    "--audit-output", str(audit),
                    "--expected-count", "1",
                    "--max-count", "3",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            with ready.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["tag"] for row in rows], ["second", "first"])
            report = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(report["manifest_count"], 2)
            self.assertEqual(report["manifest_rows"], 2)

    def test_calibrated_rank_uses_predicted_hidden_dsc_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.tsv"
            with history.open("w", newline="", encoding="utf-8") as handle:
                fields = (
                    "public_dsc", "public_nsd", "codabench_dsc", "codabench_nsd"
                )
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                for value in (0.4, 0.5, 0.6, 0.7, 0.8):
                    writer.writerow(
                        {
                            "public_dsc": value,
                            "public_nsd": value,
                            "codabench_dsc": value,
                            "codabench_nsd": value,
                        }
                    )
            candidates = root / "candidates.tsv"
            fields = ("tag", "family", "DSC", "NSD", "submission", "sha256")
            with candidates.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerow(
                    {"tag": "dsc_target", "family": "focus", "DSC": 0.8,
                     "NSD": 0.5, "submission": "a.zip", "sha256": "a"}
                )
                writer.writerow(
                    {"tag": "balanced", "family": "focus", "DSC": 0.7,
                     "NSD": 0.7, "submission": "b.zip", "sha256": "b"}
                )
            output = root / "ranked.tsv"
            json_output = root / "ranked.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "rank_submission_candidates_calibrated.py"),
                    "--history", str(history),
                    "--candidates", str(candidates),
                    "--output", str(output),
                    "--json-output", str(json_output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["tag"] for row in rows], ["dsc_target", "balanced"])
            report = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertIn("DSC first", report["diagnostics"]["ranking_objective"])


if __name__ == "__main__":
    unittest.main()
