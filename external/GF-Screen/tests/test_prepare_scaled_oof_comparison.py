import csv
import json
import tempfile
import unittest
from pathlib import Path

from prepare_scaled_oof_comparison import prepare


class PrepareScaledOOFComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps({"items": [{"name": name} for name in ("a", "b", "c")]})
        )

    def tearDown(self):
        self.temporary.cleanup()

    def calibration(self, name, values):
        metrics = self.root / f"{name}.csv"
        with metrics.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD"))
            writer.writerows(values)
        dsc = sum(row[1] for row in values) / len(values)
        nsd = sum(row[2] for row in values) / len(values)
        calibration = self.root / f"{name}.json"
        calibration.write_text(
            json.dumps(
                {
                    "cases": len(values),
                    "Lesion_DSC": dsc,
                    "Lesion_NSD": nsd,
                    "metrics": str(metrics),
                    "classification_threshold": 0.8,
                    "segmentation_threshold": 0.6,
                    "deployment_mode": "seed_union",
                }
            )
        )
        return calibration

    def test_selects_strongest_baseline_and_subsets_scaled_metrics(self):
        weak = self.calibration("weak", [("a", .4, .4), ("b", .5, .5), ("c", .6, .6)])
        strong = self.calibration("strong", [("a", .6, .5), ("b", .7, .6), ("c", .8, .7)])
        scaled = self.calibration(
            "scaled",
            [("a", .7, .6), ("b", .8, .7), ("c", .9, .8), ("d", .5, .5), ("e", .5, .5)],
        )
        payload = prepare(
            self.manifest, scaled, [("weak", weak), ("strong", strong)], self.root / "out"
        )
        self.assertEqual(payload["selected_baseline"], "strong")
        with Path(payload["scaled_subset_metrics"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["Name"] for row in rows], ["a", "b", "c"])
        self.assertFalse(payload["public_or_hidden_data_read"])

    def test_rejects_baseline_coverage_mismatch(self):
        bad = self.calibration("bad", [("a", .5, .5), ("b", .5, .5), ("x", .5, .5)])
        scaled = self.calibration(
            "scaled", [("a", .5, .5), ("b", .5, .5), ("c", .5, .5), ("d", .5, .5)]
        )
        with self.assertRaisesRegex(ValueError, "baseline coverage mismatch"):
            prepare(self.manifest, scaled, [("bad", bad)], self.root / "out")

    def test_anchor_selected_scale_comparison_uses_standalone_metrics(self):
        baseline = self.calibration(
            "baseline", [("a", .4, .4), ("b", .5, .5), ("c", .6, .6)]
        )
        standalone = self.calibration(
            "standalone_scaled",
            [("a", .5, .4), ("b", .6, .5), ("c", .7, .6), ("d", .4, .3), ("e", .4, .3)],
        )
        composed = self.calibration(
            "composed_scaled",
            [("a", .9, .9), ("b", .9, .9), ("c", .9, .9), ("d", .9, .9), ("e", .9, .9)],
        )
        payload = json.loads(composed.read_text())
        payload["calibration_deployment"] = "anchor_currentgf"
        payload["standalone_calibration"] = str(standalone)
        composed.write_text(json.dumps(payload))
        result = prepare(
            self.manifest, composed, [("baseline", baseline)], self.root / "out"
        )
        self.assertEqual(result["scaled_selected_deployment"], "anchor_currentgf")
        self.assertEqual(result["scaled_comparison_deployment"], "standalone")
        self.assertEqual(result["scaled_comparison_calibration"], str(standalone.resolve()))
        with Path(result["scaled_subset_metrics"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertAlmostEqual(float(rows[0]["Lesion_DSC"]), 0.5)


if __name__ == "__main__":
    unittest.main()
