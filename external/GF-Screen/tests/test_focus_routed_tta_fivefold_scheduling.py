import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_focus_routed_tta_fivefold_20260824.sh").read_text(
    encoding="utf-8"
)


class FocusRoutedTTAFivefoldSchedulingTests(unittest.TestCase):
    def test_is_strictly_gated_and_local_only(self):
        self.assertIn("PROMOTE_TTA_TO_FIVEFOLD", SCRIPT)
        self.assertIn("QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME", SCRIPT)
        self.assertIn("KEEP_GLOBAL_FOCUS_EPOCHS", SCRIPT)
        self.assertIn("must not suppress", SCRIPT)
        self.assertIn("summarize_focus_routed_tta.py", SCRIPT)
        self.assertIn("--tta-axes xy", SCRIPT)
        self.assertNotIn("val_image", SCRIPT)
        self.assertNotIn("submission.zip", SCRIPT)
        self.assertNotIn("codabench", SCRIPT.lower())
        self.assertIn("EARLY_ACCELERATOR_TERMINAL.json", SCRIPT)
        self.assertIn("anything missing", SCRIPT)

    def test_partitions_folds_across_both_guarded_gpus(self):
        self.assertIn("fold % 2 == 0", SCRIPT)
        self.assertIn('run_sequence 0 "${specifications0[@]}"', SCRIPT)
        self.assertIn('run_sequence 1 "${specifications1[@]}"', SCRIPT)
        self.assertIn('"${used}" -le 4096', SCRIPT)


if __name__ == "__main__":
    unittest.main()
