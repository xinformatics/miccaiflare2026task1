import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "audit_standalone_public_20260824.py"
SPEC = importlib.util.spec_from_file_location("standalone_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_metrics(path: Path, rows: list[tuple[str, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD"))
        writer.writerows(rows)


class StandaloneAuditTest(unittest.TestCase):
    def test_exact_case_validation_and_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels"
            labels.mkdir()
            for name in ("a", "b"):
                (labels / f"{name}.nii.gz").touch()
            anchor = root / "anchor.csv"
            better = root / "public_sweeps" / "standalone_prompt_banks_20260821" / "metrics" / "better.metrics.csv"
            worse = root / "public_sweeps" / "standalone_prompt_banks_20260821" / "metrics" / "worse.metrics.csv"
            write_metrics(anchor, [("a", 0.8, 0.7), ("b", 0.6, 0.5)])
            write_metrics(better, [("b", 0.5, 0.4), ("a", 0.7, 0.6)])
            write_metrics(worse, [("a", 0.1, 0.2), ("b", 0.2, 0.1)])
            report = MODULE.summarize(
                [worse, better], labels, anchor, bootstrap=100, seed=1
            )
            self.assertEqual(report["cases"], 2)
            self.assertEqual(report["candidate_count"], 2)
            self.assertTrue(report["best"]["name"].endswith("better"))
            self.assertEqual(report["decision"], "KEEP_BIOMEDPARSE_AS_PROPOSAL_SOURCE_ONLY")

    def test_rejects_missing_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels"
            labels.mkdir()
            (labels / "a.nii.gz").touch()
            (labels / "b.nii.gz").touch()
            metrics = root / "metrics.csv"
            write_metrics(metrics, [("a", 0.5, 0.5)])
            with self.assertRaisesRegex(ValueError, "case mismatch"):
                MODULE.read_metrics(metrics, MODULE.expected_cases(labels))


if __name__ == "__main__":
    unittest.main()
