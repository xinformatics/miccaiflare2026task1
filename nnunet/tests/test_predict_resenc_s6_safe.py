import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from nnunet.predict_resenc_s6_safe import (  # noqa: E402
    TRAINER_NAME,
    nnUNetTrainerResEncS6PanCancerPartialBox,
    rewrite_terminal,
    safe,
    terminal_argument,
)


class PredictResEncS6SafeTests(unittest.TestCase):
    def test_custom_trainer_is_registered_with_shared_safe_resolver(self):
        self.assertIs(
            safe.resolve_trainer(None, TRAINER_NAME, None),
            nnUNetTrainerResEncS6PanCancerPartialBox,
        )

    def test_terminal_argument_is_required_and_exact(self):
        self.assertEqual(
            terminal_argument(["--gpu", "0", "--terminal", "/tmp/result.json"]),
            Path("/tmp/result.json"),
        )
        with self.assertRaises(ValueError):
            terminal_argument(["--gpu", "0"])

    def test_terminal_rewrite_requires_exact_trainer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "TERMINAL.json"
            path.write_text(
                json.dumps(
                    {"status": "COMPLETE", "trainer": TRAINER_NAME, "uploaded": False}
                ),
                encoding="utf-8",
            )
            rewrite_terminal(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["architecture_parameters"], 107_467_050)
            self.assertIn("ResidualEncoderUNet S6", payload["model"])
            self.assertFalse(payload["uploaded"])

            path.write_text(
                json.dumps({"status": "COMPLETE", "trainer": "wrong"}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                rewrite_terminal(path)

    def test_adapter_has_no_upload_or_label_argument(self):
        source = (PROJECT_ROOT / "nnunet" / "predict_resenc_s6_safe.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('add_argument("--labels"', source)
        for forbidden in ("codabench.org", "requests.post", "curl ", "selenium"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
