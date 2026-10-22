import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = (ROOT / "watch_msd_teacher_snapshot_20260825.sh").read_text(
    encoding="utf-8"
)


class MsdTeacherTransitionAuditTests(unittest.TestCase):
    def test_is_read_only_msd_specific_nested_audit(self):
        self.assertIn('train_unlabel/MSD-506', WATCHER)
        self.assertIn('minimum=16', WATCHER)
        self.assertIn('audit_nested_teacher_masks.py', WATCHER)
        for threshold in ("0.85", "0.90", "0.95", "0.97"):
            self.assertIn(f'lesion_{threshold}', WATCHER)
        lowered = WATCHER.lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("codabench", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertNotIn("rm ", lowered)


if __name__ == "__main__":
    unittest.main()
