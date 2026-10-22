import csv
import gzip
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from rank_submission_shortlist import (
    archive_foreground_counts,
    choose,
    foreground_drift,
    load_candidates,
)
from build_three_submission_shortlist import CandidateSpec, partition_available


class SubmissionShortlistTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.anchor = self.root / "anchor.csv"
        self._metrics(self.anchor, 0.50, 0.40)

    def tearDown(self):
        self.temporary.cleanup()

    def _metrics(self, path, dsc, nsd):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["Name", "Lesion_DSC", "Lesion_NSD"])
            writer.writeheader()
            for index in range(2):
                writer.writerow({"Name": f"case{index}", "Lesion_DSC": dsc, "Lesion_NSD": nsd})

    def _zip(self, path, token):
        with zipfile.ZipFile(path, "w") as archive:
            for index in range(1, 4):
                info = zipfile.ZipInfo(
                    f"FLARE-Val_{index:04d}.nii.gz", date_time=(2020, 1, 1, 0, 0, 0)
                )
                archive.writestr(info, f"{token}-{index}")

    def _candidate(self, name, family, dsc, nsd, decision="PROMOTE", submitted=False, token=None):
        metrics = self.root / f"{name}.csv"
        archive = self.root / f"{name}.zip"
        gate = self.root / f"{name}.json"
        self._metrics(metrics, dsc, nsd)
        self._zip(archive, token or name)
        gate.write_text(json.dumps({"decision": decision}))
        return {
            "name": name,
            "family": family,
            "metrics": metrics.name,
            "submission": archive.name,
            "gate": gate.name,
            "submitted": str(submitted).lower(),
        }

    def test_promoted_diverse_candidates_precede_fallbacks(self):
        rows = [
            self._candidate("focus_best", "focus", 0.70, 0.60),
            self._candidate("focus_second", "focus", 0.68, 0.59),
            self._candidate("guided", "guided", 0.66, 0.62),
            self._candidate("fallback", "tta", 0.72, 0.61, decision="KEEP_ANCHOR"),
        ]
        manifest = self.root / "manifest.tsv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        candidates, means = load_candidates(manifest, self.anchor, 2, 3)
        self.assertEqual(means, {"Lesion_DSC": 0.5, "Lesion_NSD": 0.4})
        selected = choose(candidates, 3)
        self.assertEqual([item.name for item in selected], ["focus_best", "guided", "focus_second"])
        self.assertTrue(all(item.status == "PROMOTED" for item in selected))

    def test_near_dsc_nsd_hedge_precedes_family_diversity(self):
        rows = [
            self._candidate("dsc_best", "focus", 0.700, 0.600),
            self._candidate("nsd_hedge", "focus", 0.698, 0.640),
            self._candidate("guided", "guided", 0.690, 0.620),
            self._candidate("far_nsd", "tta", 0.680, 0.650),
        ]
        manifest = self.root / "manifest.tsv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        candidates, _ = load_candidates(manifest, self.anchor, 2, 3)
        selected = choose(candidates, 3)
        self.assertEqual(
            [item.name for item in selected],
            ["dsc_best", "nsd_hedge", "guided"],
        )

    def test_previously_submitted_and_duplicate_archives_are_excluded(self):
        rows = [
            self._candidate("submitted", "old", 0.80, 0.70, submitted=True),
            self._candidate("first", "focus", 0.70, 0.60, token="same"),
            self._candidate("duplicate", "guided", 0.69, 0.61, token="same"),
        ]
        manifest = self.root / "manifest.tsv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        candidates, _ = load_candidates(manifest, self.anchor, 2, 3)
        selected = choose(candidates, 3)
        self.assertEqual([item.name for item in selected], ["first"])
        duplicate = next(item for item in candidates if item.name == "duplicate")
        self.assertEqual(duplicate.status, "EXCLUDE_DUPLICATE_ARCHIVE")

    def test_fallback_preserves_nsd_before_chasing_dsc(self):
        rows = [
            self._candidate("stable", "focus", 0.55, 0.41, decision="KEEP_ANCHOR"),
            self._candidate("dsc_only", "guided", 0.75, 0.39, decision="KEEP_ANCHOR"),
        ]
        manifest = self.root / "manifest.tsv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        candidates, _ = load_candidates(manifest, self.anchor, 2, 3)
        selected = choose(candidates, 2)
        self.assertEqual([item.name for item in selected], ["stable", "dsc_only"])
        self.assertEqual(selected[0].status, "FALLBACK_NONREGRESS_GATE_UNCERTAIN")
        self.assertEqual(selected[1].status, "FALLBACK_NSD_REGRESSION")

    def test_hidden_foreground_drift_is_a_broad_catastrophe_guard(self):
        stable = foreground_drift([0, 100, 1000, 10000], [0, 110, 900, 12000])
        self.assertTrue(stable["hidden_sanity_pass"])
        drifted = foreground_drift([0, 100, 1000, 10000], [10000, 1, 1, 1])
        self.assertFalse(drifted["hidden_sanity_pass"])

    def test_missing_historical_archive_is_audited_without_blocking_ready_candidate(self):
        ready_metrics = self.root / "ready.csv"
        ready_archive = self.root / "ready.zip"
        ready_gate = self.root / "ready.json"
        self._metrics(ready_metrics, 0.70, 0.60)
        self._zip(ready_archive, "ready")
        ready_gate.write_text(json.dumps({"decision": "PROMOTE"}))
        missing_archive = self.root / "never_created.zip"
        available, unavailable = partition_available(
            [
                CandidateSpec(
                    "ready", "focus", ready_metrics, ready_archive, ready_gate
                ),
                CandidateSpec(
                    "stale", "nnunet", ready_metrics, missing_archive, ready_gate
                ),
            ]
        )
        self.assertEqual([item.name for item in available], ["ready"])
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0]["name"], "stale")
        self.assertEqual(unavailable[0]["missing"], "submission")
        self.assertEqual(unavailable[0]["status"], "EXCLUDE_UNAVAILABLE_ASSET")

    def test_archive_foreground_counts_streams_binary_nifti(self):
        archive_path = self.root / "masks.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for index, count in enumerate((3, 7), 1):
                array = np.zeros((4, 5, 6), dtype=np.uint8)
                array.reshape(-1)[:count] = 1
                nii = self.root / f"mask{index}.nii"
                nii_gz = self.root / f"mask{index}.nii.gz"
                nib.save(nib.Nifti1Image(array, np.eye(4)), nii)
                with nii.open("rb") as source, gzip.open(nii_gz, "wb") as output:
                    output.write(source.read())
                archive.write(nii_gz, f"FLARE-Val_{index:04d}.nii.gz")
        self.assertEqual(archive_foreground_counts(archive_path), [3, 7])


if __name__ == "__main__":
    unittest.main()
