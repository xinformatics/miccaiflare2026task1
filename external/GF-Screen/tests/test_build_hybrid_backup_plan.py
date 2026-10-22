import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "build_hybrid_backup_plan.py"


class HybridBackupPlanTests(unittest.TestCase):
    def write_family(self, root, family, dsc, nsd, catastrophic=0.0):
        robust = root / f"{family}.robust.json"
        robust.write_text(
            json.dumps(
                {
                    "decision": "KEEP_ANCHOR",
                    "anchor_mean": {"Lesion_DSC": 0.59, "Lesion_NSD": 0.50},
                    "candidate_mean": {"Lesion_DSC": dsc, "Lesion_NSD": nsd},
                    "catastrophic_regression_rate": catastrophic,
                }
            )
        )
        summary = root / f"{family}.tsv"
        threshold_name = "glance_threshold" if family == "glance" else "lesion_threshold"
        fields = [
            "variant", "checkpoint_tag", "checkpoint", threshold_name,
            "DSC", "NSD", "metrics", "robust",
        ]
        if family == "cross":
            fields = [
                "variant", "checkpoint", "glance_threshold", "lesion_threshold",
                "DSC", "NSD", "metrics", "robust",
            ]
        checkpoint = root / f"{family}.pt"
        checkpoint.write_bytes(b"checkpoint")
        metrics = root / f"{family}.csv"
        metrics.write_text("Name,Lesion_DSC,Lesion_NSD\ncase,0.5,0.5\n")
        row = {
            "variant": f"complete04_{family}",
            "checkpoint_tag": "checkpoint_epoch_0281",
            "checkpoint": str(checkpoint),
            "glance_threshold": "0.50",
            "lesion_threshold": "0.40",
            "DSC": str(dsc),
            "NSD": str(nsd),
            "metrics": str(metrics),
            "robust": str(robust),
        }
        with summary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow({key: row[key] for key in fields})
        return summary

    def test_selects_near_anchor_and_rejects_large_drop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            focus = self.write_family(root, "focus", 0.592, 0.499)
            glance = self.write_family(root, "glance", 0.58, 0.49)
            cross = self.write_family(root, "cross", 0.591, 0.501)
            output = root / "plan.tsv"
            subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--focus-summary", str(focus),
                    "--glance-summary", str(glance),
                    "--cross-summary", str(cross),
                    "--output", str(output),
                    "--count", "3",
                ],
                check=True,
            )
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual({row["family"] for row in rows}, {"focus", "cross"})
            self.assertTrue(all(row["gate"] == "NEAR_ANCHOR_BACKUP" for row in rows))
            self.assertTrue(all(len(row["tag"]) < 64 for row in rows))


if __name__ == "__main__":
    unittest.main()
