import json
import sys
import tempfile
import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANCANCER))
from select_recall_loss_escalation import select


def endpoint(path: Path, epoch: int, dsc: float, nsd: float, source: bool) -> None:
    payload = {
        "status": "COMPLETE",
        "evaluation_epoch": epoch,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    metric = {"Lesion_DSC": dsc, "Lesion_NSD": nsd}
    if source:
        payload["results"] = [
            {"configuration": "stunet_standalone", "Lesion_DSC": 0.4, "Lesion_NSD": 0.3},
            {"configuration": "gf_anchor_confirm_either_p70", **metric},
        ]
    else:
        payload["metrics"] = {"gf_anchor_confirm_either_p70": metric}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def diagnostic(root: Path, branch: str, epoch: int, precision: float, recall: float, volume: float, fn_oracle: float, fp_oracle: float) -> None:
    base = root / f"{branch}_epoch_{epoch:04d}"
    lesion = base / "lesion_errors" / "summary.json"
    gap = base / "dsc_nsd_gap" / "summary.json"
    lesion.parent.mkdir(parents=True, exist_ok=True)
    gap.parent.mkdir(parents=True, exist_ok=True)
    lesion.write_text(json.dumps({
        "cases": 50,
        "micro": {"precision": precision, "recall": recall},
        "macro": {"fn_free_oracle_dice": fn_oracle, "fp_free_oracle_dice": fp_oracle},
    }))
    gap.write_text(json.dumps({"cases": 50, "aggregate": {"volume_ratio": volume}}))


class RecallLossEscalationTests(unittest.TestCase):
    def build(self, qualify: bool):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        balanced = root / "balanced"
        source = root / "source"
        diag = root / "diag"
        for epoch in (40, 80):
            endpoint(balanced / f"epoch_{epoch:04d}" / "TERMINAL.json", epoch, 0.62, 0.54, False)
            endpoint(
                source / f"epoch_{epoch:04d}" / "TERMINAL.json",
                epoch,
                0.624 if qualify else 0.619,
                0.539,
                True,
            )
            diagnostic(diag, "balanced", epoch, 0.82, 0.70, 0.86, 0.86, 0.81)
            diagnostic(
                diag,
                "source",
                epoch,
                0.80,
                0.71 if qualify else 0.70,
                0.88,
                0.86,
                0.81,
            )
        return temporary, balanced, source, diag

    def test_qualifies_only_with_causal_recall_response_and_fn_headroom(self):
        temporary, balanced, source, diag = self.build(True)
        try:
            result = select(balanced, source, diag)
            self.assertEqual(result["decision"], "QUALIFY_TVERSKY_A020_B080_FOLLOWUP")
            self.assertTrue(result["selected"]["qualifies"])
            self.assertFalse(result["hidden_read"])
        finally:
            temporary.cleanup()

    def test_dsc_regression_rejects_stronger_loss(self):
        temporary, balanced, source, diag = self.build(False)
        try:
            result = select(balanced, source, diag)
            self.assertEqual(result["decision"], "KEEP_A025_B075_NO_STRONGER_LOSS")
            self.assertFalse(result["selected"]["checks"]["p70_dsc_direction"])
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
