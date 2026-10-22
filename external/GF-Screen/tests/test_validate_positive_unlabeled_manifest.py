import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_positive_unlabeled_manifest import validate_contract


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class ValidatePositiveUnlabeledManifestTest(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        unlabeled = root / "unlabeled"
        images = [unlabeled / f"case_{index}.nii.gz" for index in range(3)]
        for path in images:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        label = root / "labels" / "case_0.nii.gz"
        label.parent.mkdir(parents=True)
        label.write_bytes(b"pseudo-label")
        label_identity = identity(label)
        train_audit = root / "train_audit.json"
        write_json(
            train_audit,
            {
                "status": "PASS_WITH_EXCLUSIONS",
                "unlabeled_root": str(unlabeled.resolve()),
                "unlabeled_files": 3,
                "usable_unique_files": 2,
                "excluded_unlabeled_files": [str(images[2].resolve())],
            },
        )
        pseudo_audit = root / "pseudo_audit.json"
        write_json(
            pseudo_audit,
            {
                "images": 2,
                "excluded_exact_overlaps": 1,
                "excluded_unlabeled_files": [str(images[2].resolve())],
                "positive_cases": 1,
                "empty_or_rejected_cases": 1,
                "cases": [
                    {
                        "image": str(images[0].resolve()),
                        "label": str(label.resolve()),
                        "label_sha256": label_identity["sha256"],
                        "label_bytes": label.stat().st_size,
                        "components": 1,
                    },
                    {
                        "image": str(images[1].resolve()),
                        "label": None,
                        "components": 0,
                    },
                ],
            },
        )
        manifest = root / "manifest.json"
        write_json(
            manifest,
            {
                "training": [
                    {
                        "image": str(images[0].resolve()),
                        "label": str(label.resolve()),
                        "label_sha256": label_identity["sha256"],
                        "label_bytes": label.stat().st_size,
                        "pseudo_label": True,
                    }
                ]
            },
        )
        terminal = root / "terminal.json"
        write_json(
            terminal,
            {
                "status": "COMPLETE",
                "decision": "PSEUDO_LABELS_READY_FOR_LONG_TRAINING",
                "train_unlabel_audit": identity(train_audit),
                "balanced_pseudo_audit": identity(pseudo_audit),
            },
        )
        return terminal, manifest, pseudo_audit

    def test_exact_positive_subset_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.build_fixture(Path(temporary))
            payload = validate_contract(*paths)
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["excluded_images"], 1)
            self.assertEqual(payload["positive_cases"], 1)
            self.assertEqual(payload["source_cohort_counts"], {"__root__": 2})
            self.assertEqual(payload["positive_cohort_counts"], {"__root__": 1})
            self.assertEqual(
                payload["empty_or_rejected_cohort_counts"], {"__root__": 1}
            )

    def test_excluded_image_substitution_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terminal, manifest, pseudo_audit = self.build_fixture(root)
            audit = json.loads(pseudo_audit.read_text())
            audit["cases"][1]["image"] = str(
                (root / "unlabeled" / "case_2.nii.gz").resolve()
            )
            write_json(pseudo_audit, audit)
            terminal_payload = json.loads(terminal.read_text())
            terminal_payload["balanced_pseudo_audit"] = identity(pseudo_audit)
            write_json(terminal, terminal_payload)
            with self.assertRaisesRegex(ValueError, "exactly the usable source set"):
                validate_contract(terminal, manifest, pseudo_audit)

    def test_changed_pseudo_label_bytes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terminal, manifest, pseudo_audit = self.build_fixture(root)
            label = Path(json.loads(manifest.read_text())["training"][0]["label"])
            label.write_bytes(b"tampered-pseudo-label")
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                validate_contract(terminal, manifest, pseudo_audit)

    def test_source_cohort_without_positive_case_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terminal, manifest, pseudo_audit = self.build_fixture(root)
            extra = root / "unlabeled" / "cohort_b" / "case_3.nii.gz"
            extra.parent.mkdir(parents=True)
            extra.touch()

            terminal_payload = json.loads(terminal.read_text())
            train_audit = Path(terminal_payload["train_unlabel_audit"]["path"])
            train_payload = json.loads(train_audit.read_text())
            train_payload["unlabeled_files"] = 4
            train_payload["usable_unique_files"] = 3
            write_json(train_audit, train_payload)

            pseudo_payload = json.loads(pseudo_audit.read_text())
            pseudo_payload["images"] = 3
            pseudo_payload["empty_or_rejected_cases"] = 2
            pseudo_payload["cases"].append(
                {"image": str(extra.resolve()), "label": None, "components": 0}
            )
            write_json(pseudo_audit, pseudo_payload)
            terminal_payload["train_unlabel_audit"] = identity(train_audit)
            terminal_payload["balanced_pseudo_audit"] = identity(pseudo_audit)
            write_json(terminal, terminal_payload)

            with self.assertRaisesRegex(ValueError, "no positive pseudo cases"):
                validate_contract(terminal, manifest, pseudo_audit)


if __name__ == "__main__":
    unittest.main()
