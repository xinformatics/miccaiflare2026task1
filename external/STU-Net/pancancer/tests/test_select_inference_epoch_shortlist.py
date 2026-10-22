import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "select_inference_epoch_shortlist.py"
SPEC = importlib.util.spec_from_file_location("select_epoch_shortlist", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SelectInferenceEpochShortlistTest(unittest.TestCase):
    def make_epoch(self, fold: Path, epoch: int, ema: float) -> None:
        checkpoint = fold / f"checkpoint_inference_epoch_{epoch:04d}.pth"
        checkpoint.write_bytes(bytes([epoch % 251]) * (epoch + 1))
        payload = {
            "schema_version": 1,
            "checkpoint": checkpoint.name,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "current_epoch": epoch,
            "trainer_name": "STUNetTrainerPanCancer",
            "internal_partial_label_metrics": {
                "ema_fg_dice": ema,
                "mean_fg_dice": ema,
            },
        }
        checkpoint.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    def test_endpoint_is_always_kept_and_other_rows_are_spaced(self):
        with tempfile.TemporaryDirectory() as directory:
            fold = Path(directory)
            for epoch, ema in [(101, 0.72), (105, 0.76), (109, 0.75), (112, 0.80), (130, 0.79)]:
                self.make_epoch(fold, epoch, ema)
            rows = MODULE.read_inventory(fold, 101, 130)
            selected = MODULE.select_spaced(rows, top_k=3, min_separation=5)
        self.assertEqual([row["epoch"] for row in selected], [105, 112, 130])

    def test_corrupt_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fold = Path(directory)
            self.make_epoch(fold, 101, 0.72)
            checkpoint = fold / "checkpoint_inference_epoch_0101.pth"
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                MODULE.read_inventory(fold, 101, 130)

    def test_interval_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            fold = Path(directory)
            self.make_epoch(fold, 100, 0.99)
            self.make_epoch(fold, 101, 0.70)
            self.make_epoch(fold, 130, 0.80)
            self.make_epoch(fold, 131, 0.99)
            rows = MODULE.read_inventory(fold, 101, 130)
        self.assertEqual([row["epoch"] for row in rows], [101, 130])


if __name__ == "__main__":
    unittest.main()
