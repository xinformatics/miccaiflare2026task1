import json
import tempfile
import unittest
from pathlib import Path

from nnunet.predict_resenc_s6_partialband12_safe import (
    TRAINER_NAME,
    rewrite_terminal,
    terminal_argument,
)


class PredictPartialBand12SafeTest(unittest.TestCase):
    def test_terminal_argument_is_required(self):
        self.assertEqual(terminal_argument(["--terminal", "result.json"]), Path("result.json"))
        with self.assertRaises(ValueError):
            terminal_argument([])

    def test_terminal_rewrite_is_atomic_and_model_specific(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.json"
            path.write_text(
                json.dumps({"status": "COMPLETE", "trainer": TRAINER_NAME}) + "\n"
            )
            rewrite_terminal(path)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["architecture_parameters"], 107_467_050)
            self.assertEqual(
                payload["partial_label_training_policy"], "shape_linf_band_12mm"
            )
            self.assertFalse(list(path.parent.glob(".*.tmp")))

    def test_terminal_rewrite_rejects_other_trainer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.json"
            path.write_text(json.dumps({"status": "COMPLETE", "trainer": "other"}))
            with self.assertRaises(ValueError):
                rewrite_terminal(path)


if __name__ == "__main__":
    unittest.main()
