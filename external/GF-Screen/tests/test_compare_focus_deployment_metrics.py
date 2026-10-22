import csv
import tempfile
import unittest
from pathlib import Path

from compare_focus_deployment_metrics import compare


def write_rows(path: Path, rows: list[tuple[str, float, float, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD", "pred_fg_voxels"))
        writer.writerows(rows)


class FocusDeploymentComparisonTests(unittest.TestCase):
    def test_accepts_locked_rounding_and_voxel_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); anchor = root / "anchor.csv"; candidate = root / "candidate.csv"
            write_rows(anchor, [("a", .5, .4, 2_600_000), ("b", .7, .6, 2_700_000)])
            write_rows(candidate, [("a", .5001, .3999, 2_600_006), ("b", .6999, .6001, 2_699_994)])
            result = compare(anchor, candidate)
            self.assertFalse(result["exact_metric_rows"])
            self.assertTrue(result["deployment_equivalent"])

    def test_rejects_material_metric_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); anchor = root / "anchor.csv"; candidate = root / "candidate.csv"
            write_rows(anchor, [("a", .5, .4, 1_000_000)])
            write_rows(candidate, [("a", .5002, .4, 1_000_000)])
            self.assertFalse(compare(anchor, candidate)["deployment_equivalent"])

    def test_rejects_excess_voxel_drift_even_if_metrics_round_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); anchor = root / "anchor.csv"; candidate = root / "candidate.csv"
            write_rows(anchor, [("a", .5, .4, 5_000_000)])
            write_rows(candidate, [("a", .5, .4, 5_000_009)])
            self.assertFalse(compare(anchor, candidate)["deployment_equivalent"])


if __name__ == "__main__":
    unittest.main()
