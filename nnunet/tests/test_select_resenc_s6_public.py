import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from nnunet.select_resenc_s6_public import (  # noqa: E402
    ANALYSIS_TO_POLICY,
    parse_epoch_asset,
    select,
)


class SelectResEncS6PublicTests(unittest.TestCase):
    def make_fixture(self, root: Path, winning_dsc: float, winning_nsd: float):
        source_policy = (
            PROJECT_ROOT
            / "nnunet"
            / "RESENC_S6_PUBLIC_SELECTION_POLICY_20260827.json"
        )
        policy = root / "policy.json"
        policy.write_bytes(source_policy.read_bytes())
        analyses, checkpoints = {}, {}
        for epoch in (100, 200, 300, 400, 500):
            metrics = {
                name: {"Lesion_DSC": 0.60, "Lesion_NSD": 0.54}
                for name in ANALYSIS_TO_POLICY
            }
            if epoch == 300:
                metrics["gf_union_resenc_p40_ftu_s10_mm3000"] = {
                    "Lesion_DSC": winning_dsc,
                    "Lesion_NSD": winning_nsd,
                }
            analysis = root / f"analysis_{epoch}.json"
            analysis.write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "case_count": 50,
                        "metrics": metrics,
                        "public_labels_read": True,
                        "hidden_read": False,
                        "submission_created": False,
                        "uploaded": False,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / f"checkpoint_{epoch}.pth"
            checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
            analyses[epoch] = analysis
            checkpoints[epoch] = checkpoint
        return policy, analyses, checkpoints

    def test_selects_only_row_passing_all_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy, analyses, checkpoints = self.make_fixture(
                Path(temporary), 0.65, 0.55
            )
            payload = select(policy, analyses, checkpoints)
            self.assertEqual(payload["decision"], "REVIEW_TRAINED_CANDIDATE")
            self.assertEqual(payload["selected"]["epoch"], 300)
            self.assertEqual(
                payload["selected"]["deployment"],
                "gf_union_resenc_p40_ftu_s10_mm3000",
            )
            self.assertEqual(len(payload["results"]), 20)

    def test_nsd_failure_prevents_hidden_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy, analyses, checkpoints = self.make_fixture(
                Path(temporary), 0.65, 0.53
            )
            payload = select(policy, analyses, checkpoints)
            self.assertEqual(payload["decision"], "NO_HIDDEN_ARCHIVE")
            self.assertIsNone(payload["selected"])
            self.assertEqual(payload["diagnostic_best"]["epoch"], 300)

    def test_missing_endpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy, analyses, checkpoints = self.make_fixture(
                Path(temporary), 0.65, 0.55
            )
            analyses.pop(500)
            with self.assertRaises(ValueError):
                select(policy, analyses, checkpoints)

    def test_epoch_asset_parser(self):
        self.assertEqual(parse_epoch_asset("300=/tmp/c.pth"), (300, Path("/tmp/c.pth")))


if __name__ == "__main__":
    unittest.main()
