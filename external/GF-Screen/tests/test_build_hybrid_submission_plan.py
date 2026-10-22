from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "build_hybrid_submission_plan.py"


def robust(path: Path, decision: str) -> Path:
    path.write_text(json.dumps({"decision": decision}) + "\n", encoding="utf-8")
    return path


def write_summary(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class HybridSubmissionPlanTests(unittest.TestCase):
    def test_prefers_one_promoted_candidate_per_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            metrics = root / "metrics.csv"
            checkpoint.write_bytes(b"checkpoint")
            metrics.write_text("metric\n", encoding="utf-8")
            promote = robust(root / "promote.json", "PROMOTE")
            keep = robust(root / "keep.json", "KEEP_ANCHOR")

            common = {
                "checkpoint": str(checkpoint),
                "metrics": str(metrics),
                "combined": "0.7",
            }
            focus = root / "focus.tsv"
            write_summary(
                focus,
                [
                    "variant", "checkpoint_tag", "checkpoint", "lesion_threshold",
                    "DSC", "NSD", "combined", "metrics", "robust",
                ],
                [
                    {**common, "variant": "gf_hybrid_complete04", "checkpoint_tag": "checkpoint_epoch_0281", "lesion_threshold": "0.40", "DSC": "0.64", "NSD": "0.55", "robust": str(promote)},
                    {**common, "variant": "gf_hybrid_complete08", "checkpoint_tag": "checkpoint_epoch_0283", "lesion_threshold": "0.40", "DSC": "0.66", "NSD": "0.56", "robust": str(promote)},
                ],
            )
            glance = root / "glance.tsv"
            write_summary(
                glance,
                [
                    "variant", "checkpoint_tag", "checkpoint", "glance_threshold",
                    "DSC", "NSD", "combined", "metrics", "robust",
                ],
                [
                    {**common, "variant": "gf_glance_hybrid04_lr1e4", "checkpoint_tag": "checkpoint_epoch_0285", "glance_threshold": "0.35", "DSC": "0.63", "NSD": "0.54", "robust": str(promote)},
                    {**common, "variant": "gf_glance_hybrid04_lr3e5", "checkpoint_tag": "model", "glance_threshold": "0.50", "DSC": "0.70", "NSD": "0.60", "robust": str(keep)},
                ],
            )
            cross = root / "cross.tsv"
            write_summary(
                cross,
                [
                    "variant", "checkpoint", "glance_threshold", "lesion_threshold",
                    "DSC", "NSD", "metrics", "robust", "decision",
                ],
                [
                    {"variant": "gf_hybrid_cross", "checkpoint": str(checkpoint), "glance_threshold": "0.35", "lesion_threshold": "0.40", "DSC": "0.68", "NSD": "0.58", "metrics": str(metrics), "robust": str(promote), "decision": "PROMOTE"},
                ],
            )
            output = root / "plan.tsv"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--focus-summary", str(focus), "--glance-summary", str(glance), "--cross-summary", str(cross), "--output", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["family"] for row in rows], ["cross", "focus", "glance"])
            self.assertEqual(rows[1]["DSC"], "0.66")
            self.assertEqual(rows[2]["glance_threshold"], "0.35")
            self.assertTrue(all(len(row["tag"]) < 64 for row in rows))


if __name__ == "__main__":
    unittest.main()
