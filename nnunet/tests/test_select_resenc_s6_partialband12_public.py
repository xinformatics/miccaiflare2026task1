import json
import tempfile
import unittest
from pathlib import Path

from nnunet.select_resenc_s6_partialband12_public import (
    EXPECTED_EPOCHS,
    EXPECTED_MODEL,
    load_policy,
    select,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "nnunet/RESENC_S6_PARTIALBAND12_PUBLIC_SELECTION_POLICY_20260827.json"


class SelectPartialBand12PublicTest(unittest.TestCase):
    def test_frozen_policy_contract(self):
        payload = load_policy(POLICY)
        self.assertEqual(payload["model"], EXPECTED_MODEL)
        self.assertEqual(payload["checkpoint_epochs"], EXPECTED_EPOCHS)
        self.assertFalse(payload["threshold_search_after_public_read"])
        self.assertFalse(payload["selection"]["geometry_audit_is_selection_authority"])

    def test_selector_requires_gate_and_ranks_dsc_then_nsd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analyses, checkpoints = {}, {}
            deployments = [
                "resenc_standalone_p50",
                "gf_anchor_confirm_either_p70",
                "gf_resenc_boundary3mm_p70p50_ftu",
                "gf_union_resenc_p40_ftu_s10_mm3000",
            ]
            for epoch in EXPECTED_EPOCHS:
                analysis = root / f"analysis_{epoch}.json"
                metrics = {
                    name: {"Lesion_DSC": 0.60, "Lesion_NSD": 0.54}
                    for name in deployments
                }
                if epoch == 50:
                    metrics["gf_anchor_confirm_either_p70"] = {
                        "Lesion_DSC": 0.64,
                        "Lesion_NSD": 0.54,
                    }
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
                    )
                )
                checkpoint = root / f"checkpoint_{epoch}.pth"
                checkpoint.write_bytes(f"epoch-{epoch}".encode())
                analyses[epoch] = analysis
                checkpoints[epoch] = checkpoint
            result = select(POLICY, analyses, checkpoints)
            self.assertEqual(result["decision"], "REVIEW_TRAINED_CANDIDATE")
            self.assertEqual(result["selected"]["epoch"], 50)
            self.assertEqual(
                result["selected"]["deployment"], "gf_anchor_confirm_either_p70"
            )


if __name__ == "__main__":
    unittest.main()
