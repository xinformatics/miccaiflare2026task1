from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nnunet.audit_resenc_s6_early_learning import (
    EXPECTED_HISTORICAL_SHA256,
    audit,
    parse_training_log,
)


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "nnunet/RESENC_S6_HISTORICAL_PUBLIC_BASELINE_20260827.json"


def record(epoch: int, train: float, val: float, dice: float) -> str:
    return "\n".join(
        (
            f"2026-08-27 00:00:00: Epoch {epoch}",
            f"2026-08-27 00:00:00: Current learning rate: {0.001 - epoch * 1e-5}",
            f"2026-08-27 00:00:00: train_loss {train}",
            f"2026-08-27 00:00:00: val_loss {val}",
            f"2026-08-27 00:00:00: Pseudo dice [np.float64({dice})]",
            "2026-08-27 00:00:00: Epoch time: 10.0 s",
        )
    )


class ResEncEarlyLearningTests(unittest.TestCase):
    def test_historical_reference_is_frozen(self):
        import hashlib

        self.assertEqual(hashlib.sha256(HISTORICAL.read_bytes()).hexdigest(), EXPECTED_HISTORICAL_SHA256)

    def test_parser_requires_complete_records(self):
        text = record(0, 1.3, 1.2, 0.0) + "\n2026: Epoch 1\n2026: train_loss 1.0\n"
        parsed = parse_training_log(text)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["epoch"], 0)

    def test_passes_after_loss_improvement_and_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "training.txt"
            log.write_text(
                "\n".join(
                    record(epoch, 1.3 - epoch * 0.03, 1.2 - epoch * 0.02, 0.1 if epoch >= 6 else 0.0)
                    for epoch in range(10)
                )
            )
            payload = audit(log, HISTORICAL)
        self.assertEqual(payload["decision"], "PASS_EARLY_LEARNING_SIGNAL")
        self.assertEqual(payload["first_foreground_activation_epoch"], 6)
        self.assertFalse(payload["automatic_training_stop"])

    def test_no_activation_is_review_required_but_never_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "training.txt"
            log.write_text(
                "\n".join(
                    record(epoch, 1.3 - epoch * 0.03, 1.2 - epoch * 0.02, 0.0)
                    for epoch in range(10)
                )
            )
            payload = audit(log, HISTORICAL)
        self.assertEqual(payload["decision"], "REVIEW_REQUIRED_NO_FOREGROUND_ACTIVATION")
        self.assertFalse(payload["automatic_training_stop"])
        self.assertFalse(payload["training_mutated"])

    def test_controller_is_read_only_and_waits_for_epoch10(self):
        source = (ROOT / "nnunet/run_resenc_s6_early_learning_sentinel_20260827.sh").read_text()
        self.assertIn("Epoch 10", source)
        self.assertNotIn("systemctl stop", source)
        self.assertNotIn("systemctl restart", source)
        self.assertNotIn("kill ", source)
        self.assertNotIn("nvidia-smi", source)


if __name__ == "__main__":
    unittest.main()
