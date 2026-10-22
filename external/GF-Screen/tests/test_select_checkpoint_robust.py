import csv
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from select_checkpoint_robust import main


class SelectCheckpointRobustTests(unittest.TestCase):
    def write_metrics(
        self,
        path: Path,
        names: list[str],
        dsc: list[float],
        nsd: list[float],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["Name", "Lesion_DSC", "Lesion_NSD"]
            )
            writer.writeheader()
            for name, dsc_value, nsd_value in zip(names, dsc, nsd):
                writer.writerow(
                    {
                        "Name": name,
                        "Lesion_DSC": dsc_value,
                        "Lesion_NSD": nsd_value,
                    }
                )

    def run_selector(
        self,
        anchor: Path,
        candidate: Path,
        output: Path,
        *,
        minimum_gain: float = 0.005,
    ) -> dict:
        argv = [
            "select_checkpoint_robust.py",
            "--anchor",
            str(anchor),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
            "--bootstrap",
            "1000",
            "--seed",
            "20260824",
            "--min-dsc-gain",
            str(minimum_gain),
            "--min-nsd-gain",
            str(minimum_gain),
            "--min-win-rate",
            "0.52",
            "--max-catastrophic-rate",
            "0.10",
        ]
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            main()
        return json.loads(output.read_text(encoding="utf-8"))

    def test_promotes_uniform_paired_dsc_and_nsd_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [f"case_{index:03d}" for index in range(30)]
            anchor, candidate, output = (
                root / "anchor.csv",
                root / "candidate.csv",
                root / "decision.json",
            )
            self.write_metrics(anchor, names, [0.5] * 30, [0.4] * 30)
            self.write_metrics(candidate, names, [0.51] * 30, [0.41] * 30)
            result = self.run_selector(anchor, candidate, output)
            self.assertEqual(result["decision"], "PROMOTE")
            self.assertTrue(all(result["checks"].values()))
            self.assertGreater(result["bootstrap_95ci"]["Lesion_DSC"][0], 0)
            self.assertGreater(result["bootstrap_95ci"]["Lesion_NSD"][0], 0)
            self.assertEqual(result["schema_version"], 1)
            self.assertEqual(
                result["anchor_metrics_sha256"],
                hashlib.sha256(anchor.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["candidate_metrics_sha256"],
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
            )
            implementation = Path(result["implementation"])
            self.assertEqual(
                result["implementation_sha256"],
                hashlib.sha256(implementation.read_bytes()).hexdigest(),
            )
            self.assertEqual(result["criteria"]["min_dsc_gain"], 0.005)

    def test_rejects_dsc_only_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [f"case_{index:03d}" for index in range(30)]
            anchor, candidate, output = (
                root / "anchor.csv",
                root / "candidate.csv",
                root / "decision.json",
            )
            self.write_metrics(anchor, names, [0.5] * 30, [0.4] * 30)
            self.write_metrics(candidate, names, [0.51] * 30, [0.4] * 30)
            result = self.run_selector(anchor, candidate, output)
            self.assertEqual(result["decision"], "KEEP_ANCHOR")
            self.assertTrue(result["checks"]["dsc_practical_gain"])
            self.assertFalse(result["checks"]["nsd_practical_gain"])
            self.assertFalse(result["checks"]["nsd_ci_positive"])

    def test_rejects_excess_catastrophic_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [f"case_{index:03d}" for index in range(20)]
            anchor, candidate, output = (
                root / "anchor.csv",
                root / "candidate.csv",
                root / "decision.json",
            )
            self.write_metrics(anchor, names, [0.5] * 20, [0.5] * 20)
            values = [0.9] * 17 + [0.1] * 3
            self.write_metrics(candidate, names, values, values)
            result = self.run_selector(anchor, candidate, output)
            self.assertGreater(result["paired_gain"]["Lesion_DSC"], 0.005)
            self.assertFalse(result["checks"]["tail_regression_guard"])
            self.assertEqual(result["decision"], "KEEP_ANCHOR")

    def test_case_inventory_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchor, candidate, output = (
                root / "anchor.csv",
                root / "candidate.csv",
                root / "decision.json",
            )
            self.write_metrics(anchor, ["a", "b"], [0.5, 0.5], [0.5, 0.5])
            self.write_metrics(candidate, ["a", "c"], [0.6, 0.6], [0.6, 0.6])
            with self.assertRaisesRegex(ValueError, "case mismatch"):
                self.run_selector(anchor, candidate, output)

    def test_nonfinite_or_out_of_range_metrics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchor, candidate, output = (
                root / "anchor.csv",
                root / "candidate.csv",
                root / "decision.json",
            )
            self.write_metrics(anchor, ["a", "b"], [0.5, 0.5], [0.5, 0.5])
            self.write_metrics(candidate, ["a", "b"], [float("nan"), 1.1], [0.6, 0.6])
            with self.assertRaisesRegex(ValueError, "invalid Lesion_DSC"):
                self.run_selector(anchor, candidate, output)


if __name__ == "__main__":
    unittest.main()
