import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from validate_long_focus_interim_reuse import validate_interim  # noqa: E402


class LongFocusInterimReuseTests(unittest.TestCase):
    def test_exact_existing_public_endpoints_are_reusable(self):
        run = ROOT / "runs/gf_focus_pu_long30_e283_to313_20260825"
        for epoch in (293, 303):
            source = (
                ROOT
                / f"public_sweeps/gf_focus_pu_long30_e{epoch}_interim_20260825"
            )
            payload = validate_interim(
                source, run / f"checkpoint_epoch_{epoch:04d}.pt", epoch
            )
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["cases_per_variant"], 50)
            self.assertFalse(payload["hidden_read"])

    def test_nonpredeclared_epoch_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_interim(Path("/tmp/none"), Path("/tmp/none"), 313)

    def test_trajectory_seeds_only_293_and_303(self):
        source = (
            ROOT / "evaluate_long_focus_pu_trajectory_public_20260825.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("for epoch in 293 303", source)
        self.assertIn("validate_long_focus_interim_reuse.py", source)
        self.assertNotIn("for epoch in 293 303 313; do\n  source=", source)


if __name__ == "__main__":
    unittest.main()
