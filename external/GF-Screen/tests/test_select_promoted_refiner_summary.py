import csv
import json
import tempfile
import unittest
from pathlib import Path

from select_promoted_refiner_summary import FIELDS, select_promoted


class SelectPromotedRefinerSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.metrics = self.root / "metrics.csv"
        self.metrics.write_text("Name,Lesion_DSC,Lesion_NSD\ncase,0.7,0.6\n")

    def tearDown(self):
        self.temporary.cleanup()

    def row(
        self,
        name,
        dsc,
        nsd,
        decision="PROMOTE",
        mode="refined",
        deployment="standalone",
    ):
        robust = self.root / f"{name}.json"
        robust.write_text(
            json.dumps(
                {
                    "decision": decision,
                    "cases": 1,
                    "candidate_mean": {"Lesion_DSC": dsc, "Lesion_NSD": nsd},
                    "checks": {"strict_gate": decision == "PROMOTE"},
                }
            )
        )
        return {
            "variant": name,
            "classification_threshold": "0.8",
            "segmentation_threshold": "0.6",
            "deployment_mode": mode,
            "deployment": deployment,
            "DSC": str(dsc),
            "NSD": str(nsd),
            "combined": str((dsc + nsd) / 2),
            "metrics": str(self.metrics),
            "robust": str(robust),
            "decision": decision,
        }

    def write(self, rows):
        path = self.root / "summary.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_selects_highest_dsc_then_nsd(self):
        summary = self.write(
            [self.row("first", 0.70, 0.62), self.row("best", 0.71, 0.60)]
        )
        self.assertEqual(select_promoted(summary, 1)["variant"], "best")

    def test_no_promotion_returns_none(self):
        summary = self.write([self.row("held", 0.72, 0.64, "KEEP_ANCHOR")])
        self.assertIsNone(select_promoted(summary, 1))

    def test_explicit_fallback_selects_only_best_anchor_row(self):
        summary = self.write(
            [
                self.row("standalone", 0.90, 0.90, "KEEP_ANCHOR"),
                self.row(
                    "anchor_lower", 0.70, 0.66, "KEEP_ANCHOR", deployment="anchor"
                ),
                self.row(
                    "anchor_best", 0.71, 0.65, "KEEP_ANCHOR", deployment="anchor"
                ),
            ]
        )
        selected = select_promoted(summary, 1, allow_anchor_fallback=True)
        self.assertEqual(selected["variant"], "anchor_best")

    def test_strict_promotion_precedes_anchor_fallback(self):
        summary = self.write(
            [
                self.row(
                    "fallback", 0.90, 0.90, "KEEP_ANCHOR", deployment="anchor"
                ),
                self.row("promoted", 0.70, 0.60, "PROMOTE"),
            ]
        )
        selected = select_promoted(summary, 1, allow_anchor_fallback=True)
        self.assertEqual(selected["variant"], "promoted")

    def test_rejects_summary_sidecar_decision_mismatch(self):
        row = self.row("bad", 0.72, 0.64)
        robust_path = Path(row["robust"])
        robust = json.loads(robust_path.read_text())
        robust["decision"] = "KEEP_ANCHOR"
        robust_path.write_text(json.dumps(robust))
        summary = self.write([row])
        with self.assertRaisesRegex(ValueError, "decision mismatch"):
            select_promoted(summary, 1)

    def test_rejects_unknown_mode(self):
        summary = self.write([self.row("bad", 0.72, 0.64, mode="invented")])
        with self.assertRaisesRegex(ValueError, "invalid deployment mode"):
            select_promoted(summary, 1)

    def test_rejects_metrics_coverage_mismatch(self):
        summary = self.write([self.row("bad", 0.72, 0.64)])
        with self.assertRaisesRegex(ValueError, "metrics coverage mismatch"):
            select_promoted(summary, 2)

    def test_rejects_promote_with_failed_check(self):
        row = self.row("bad", 0.72, 0.64)
        robust_path = Path(row["robust"])
        robust = json.loads(robust_path.read_text())
        robust["checks"]["strict_gate"] = False
        robust_path.write_text(json.dumps(robust))
        summary = self.write([row])
        with self.assertRaisesRegex(ValueError, "failed checks"):
            select_promoted(summary, 1)


if __name__ == "__main__":
    unittest.main()
