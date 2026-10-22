import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_focus_candidate_refiner_fullvolume_oof_20260823.sh").read_text(
    encoding="utf-8"
)


class FocusFullVolumeCLISeparationTests(unittest.TestCase):
    def test_input_mode_flags_are_validator_only(self):
        self.assertIn(
            'FOCUS_VALIDATOR_ARGS+=(--focus-routed-input-mode "${focus_input_mode}")',
            SCRIPT,
        )
        self.assertIn(
            'FOCUS_VALIDATOR_ARGS+=(--contour-input-mode "${focus_input_mode}")',
            SCRIPT,
        )
        validator = SCRIPT.split("validate_candidate_refiner_inference_marker.py", 1)[1]
        self.assertIn('"${FOCUS_VALIDATOR_ARGS[@]}"', validator)

        inference = SCRIPT.split('"${root}/apply_candidate_refiner.py" infer', 1)[1]
        inference = inference.split('> "${output}/fold${fold}.infer.log"', 1)[0]
        self.assertNotIn("focus-routed-input-mode", inference)
        self.assertNotIn("contour-input-mode", inference)
        self.assertNotIn("FOCUS_VALIDATOR_ARGS", inference)

    def test_shell_remains_fail_closed(self):
        self.assertIn("set -euo pipefail", SCRIPT)
        self.assertIn("inference_matches", SCRIPT)


if __name__ == "__main__":
    unittest.main()
