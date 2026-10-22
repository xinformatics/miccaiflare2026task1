import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_cbg_train_unlabel_direct_20260825.sh").read_text(
    encoding="utf-8"
)
EVALUATOR = (
    ROOT / "evaluate_cbg_unlabeled_recall_additions_public_20260823.sh"
).read_text(encoding="utf-8")
CORE = (
    ROOT / "run_dual_unlabeled_glance_after_allpartial_20260822.sh"
).read_text(encoding="utf-8")


class DirectTrainUnlabelSchedulingTests(unittest.TestCase):
    def test_uses_all_train_unlabel_through_audited_core(self):
        self.assertIn("run_dual_unlabeled_glance_after_allpartial_20260822.sh", SCRIPT)
        self.assertIn('"train_unlabel_expected_cases":2856', SCRIPT)
        self.assertIn('"teacher_empty_policy":"unknown_not_negative"', SCRIPT)
        self.assertIn("UNLABELED_CHECKPOINT", SCRIPT)

    def test_starts_from_current_cbg_family_and_stops_before_short_arms(self):
        self.assertIn("checkpoint_epoch_0283.pt", SCRIPT)
        self.assertIn("gf_touchcomp_0825", SCRIPT)
        self.assertIn("UNLABELED_ARM_BALANCED", SCRIPT)
        self.assertIn("UNLABELED_ARM_CONSERVATIVE", SCRIPT)
        self.assertIn("UNLABELED_GPU0_MAX_MIB=4096", SCRIPT)
        self.assertIn("UNLABELED_GPU1_MAX_MIB=1024", SCRIPT)
        self.assertIn("UNLABELED_TEACHER_WORKERS=1", SCRIPT)
        self.assertIn("UNLABELED_TEACHER_FOCUS_BATCH_SIZE=8", SCRIPT)
        self.assertIn("UNLABELED_PSEUDO_WORKERS=4", SCRIPT)
        self.assertIn("UNLABELED_GENERATE_ONLY=1", SCRIPT)
        self.assertIn("PSEUDO_LABELS_READY_FOR_LONG_TRAINING", SCRIPT)

    def test_memory_sensitive_worker_counts_are_validated_and_forwarded(self):
        self.assertIn('teacher_workers="${UNLABELED_TEACHER_WORKERS:-1}"', CORE)
        self.assertIn(
            'teacher_focus_batch_size="${UNLABELED_TEACHER_FOCUS_BATCH_SIZE:-8}"',
            CORE,
        )
        self.assertIn('pseudo_workers="${UNLABELED_PSEUDO_WORKERS:-4}"', CORE)
        self.assertIn('"${value}" =~ ^[1-9][0-9]*$', CORE)
        self.assertIn('--workers "${teacher_workers}"', CORE)
        self.assertIn('--focus_batch_size "${teacher_focus_batch_size}"', CORE)
        self.assertIn('--workers "${pseudo_workers}"', CORE)
        self.assertNotIn("--workers 2 --focus_batch_size 16", CORE)

    def test_public_is_evaluation_only_and_hidden_is_unreachable(self):
        lowered = SCRIPT.lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("codabench", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"public_used_for_training":false', lowered)
        self.assertIn('"hidden_read":false', lowered)
        self.assertIn('"uploaded":false', lowered)

    def test_recall_evaluator_has_isolated_environment_overrides(self):
        self.assertIn("CBG_UL_DEPLOYED", EVALUATOR)
        self.assertIn("CBG_UL_OUTPUT", EVALUATOR)
        self.assertIn("CBG_UL_BALANCED", EVALUATOR)
        self.assertIn("CBG_UL_CONSERVATIVE", EVALUATOR)


if __name__ == "__main__":
    unittest.main()
