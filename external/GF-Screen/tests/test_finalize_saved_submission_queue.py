import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from finalize_saved_submission_queue import finalize


def make_zip(
    path: Path,
    payloads: tuple[bytes, bytes],
    compression: int = zipfile.ZIP_STORED,
) -> str:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr("FLARE-Val_0001.nii.gz", payloads[0])
        archive.writestr("FLARE-Val_0002.nii.gz", payloads[1])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path: Path, rows: list[dict[str, str]], backup: bool = False) -> None:
    fields = [
        "rank", "tag", "family", "checkpoint", "glance_threshold",
        "lesion_threshold", "DSC", "NSD", "metrics", "robust",
    ]
    if backup:
        fields.append("gate")
    fields.extend(["submission", "sha256"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class FinalizeQueueTests(unittest.TestCase):
    def test_strict_precedes_backup_and_content_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "strict.zip"
            duplicate = root / "duplicate.zip"
            backup = root / "backup.zip"
            first_sha = make_zip(first, (b"a", b"b"))
            duplicate_sha = make_zip(
                duplicate, (b"a", b"b"), compression=zipfile.ZIP_DEFLATED
            )
            backup_sha = make_zip(backup, (b"c", b"d"))
            base = {
                "checkpoint": "c", "glance_threshold": "0.5",
                "lesion_threshold": "0.5", "DSC": "0.6", "NSD": "0.5",
                "metrics": "m", "robust": "r",
            }
            strict_manifest = root / "strict.tsv"
            backup_manifest = root / "backup.tsv"
            write_manifest(strict_manifest, [
                dict(base, rank="1", tag="strict", family="focus", submission=str(first), sha256=first_sha),
            ])
            write_manifest(backup_manifest, [
                dict(base, rank="1", tag="duplicate", family="glance", gate="fallback", submission=str(duplicate), sha256=duplicate_sha),
                dict(base, rank="2", tag="backup", family="cross", gate="fallback", submission=str(backup), sha256=backup_sha),
            ], backup=True)
            output = root / "READY.tsv"
            audit = root / "audit.json"
            ready = finalize(
                strict_manifest, backup_manifest, root / "missing.tsv",
                output, audit, expected_count=2, max_count=3,
            )
            self.assertEqual([row["tag"] for row in ready], ["strict", "backup"])
            self.assertEqual(ready[0]["priority"], "strict")
            report = json.loads(audit.read_text())
            self.assertEqual(report["ready_count"], 2)
            self.assertIn("prediction-content-equivalent", report["rejected"][0]["reason"])

    def test_previously_submitted_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "candidate.zip"
            digest = make_zip(archive, (b"a", b"b"))
            manifest = root / "strict.tsv"
            write_manifest(manifest, [{
                "rank": "1", "tag": "candidate", "family": "focus",
                "checkpoint": "c", "glance_threshold": "0.5",
                "lesion_threshold": "0.5", "DSC": "0.6", "NSD": "0.5",
                "metrics": "m", "robust": "r", "submission": str(archive),
                "sha256": digest,
            }])
            submitted = root / "submitted.tsv"
            submitted.write_text(f"{digest}\told.zip\n")
            ready = finalize(
                manifest, root / "missing.tsv", submitted,
                root / "READY.tsv", root / "audit.json", 2, 3,
            )
            self.assertEqual(ready, [])

    def test_candidates_are_ranked_by_balanced_evidence_within_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weaker = root / "weaker.zip"
            stronger = root / "stronger.zip"
            weaker_sha = make_zip(weaker, (b"w1", b"w2"))
            stronger_sha = make_zip(stronger, (b"s1", b"s2"))
            manifest = root / "backup.tsv"
            common = {
                "checkpoint": "c", "glance_threshold": "0.5",
                "lesion_threshold": "0.5", "metrics": "m", "robust": "r",
                "gate": "fallback",
            }
            write_manifest(manifest, [
                dict(common, rank="1", tag="weaker", family="cross",
                     DSC="0.59", NSD="0.50", submission=str(weaker),
                     sha256=weaker_sha),
                dict(common, rank="2", tag="stronger", family="focus",
                     DSC="0.61", NSD="0.51", submission=str(stronger),
                     sha256=stronger_sha),
            ], backup=True)
            ready = finalize(
                root / "missing.tsv", manifest, root / "submitted.tsv",
                root / "READY.tsv", root / "audit.json", 2, 3,
            )
            self.assertEqual([row["tag"] for row in ready], ["stronger", "weaker"])

    def test_promoted_candidates_are_ranked_dsc_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            balanced = root / "balanced.zip"
            dsc_target = root / "dsc_target.zip"
            balanced_sha = make_zip(balanced, (b"b1", b"b2"))
            dsc_target_sha = make_zip(dsc_target, (b"d1", b"d2"))
            manifest = root / "strict.tsv"
            common = {
                "checkpoint": "c", "glance_threshold": "0.5",
                "lesion_threshold": "0.5", "metrics": "m", "robust": "r",
            }
            write_manifest(manifest, [
                dict(common, rank="1", tag="balanced", family="focus",
                     DSC="0.70", NSD="0.65", submission=str(balanced),
                     sha256=balanced_sha),
                dict(common, rank="2", tag="dsc_target", family="focus",
                     DSC="0.72", NSD="0.60", submission=str(dsc_target),
                     sha256=dsc_target_sha),
            ])
            ready = finalize(
                manifest, root / "missing.tsv", root / "submitted.tsv",
                root / "READY.tsv", root / "audit.json", 2, 3,
            )
            self.assertEqual(
                [row["tag"] for row in ready], ["dsc_target", "balanced"]
            )

    def test_repairs_shell_manifest_with_embedded_crlf_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "candidate.zip"
            digest = make_zip(archive, (b"a", b"b"))
            manifest = root / "backup.tsv"
            fields = [
                "rank", "tag", "family", "checkpoint", "glance_threshold",
                "lesion_threshold", "DSC", "NSD", "metrics", "robust",
                "gate", "submission", "sha256",
            ]
            logical = [
                "1", "candidate", "focus", "c", "0.5", "0.5", "0.61",
                "0.51", "m", "r", "NEAR_ANCHOR_BACKUP",
            ]
            # Reproduce the CR retained by bash at the old plan-row boundary.
            manifest.write_bytes(
                ("\t".join(fields) + "\n" + "\t".join(logical)).encode()
                + b"\r\t"
                + str(archive).encode()
                + b"\t"
                + digest.encode()
                + b"\n"
            )
            ready = finalize(
                root / "missing.tsv", manifest, root / "submitted.tsv",
                root / "READY.tsv", root / "audit.json", 2, 3,
            )
            self.assertEqual(len(ready), 1)
            self.assertEqual(ready[0]["tag"], "candidate")
            self.assertEqual(ready[0]["sha256"], digest)


if __name__ == "__main__":
    unittest.main()
