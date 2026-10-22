import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECOVERY = (ROOT / "run_component_meta_recovery_after_sourcestrat_20260826.sh").read_text()
FOCUS = (ROOT / "queue_focus_e303_to313_after_balanced_20260826.sh").read_text()


class ComponentMetaRecoveryQueueTests(unittest.TestCase):
    def test_recovery_is_deauthorized_before_any_source_or_gpu_work(self):
        self.assertIn("LABEL_CONTRACT_CORRECTION_20260825.json", RECOVERY)
        self.assertIn("DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES", RECOVERY)
        guard = RECOVERY.index("DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES")
        wait = RECOVERY.index('while [[ ! -s "${source_public}" ]]')
        launch = RECOVERY.index('/usr/bin/bash "${fusion_controller}" frozen')
        self.assertLess(guard, wait)
        self.assertLess(guard, launch)

    def test_recovery_waits_for_source_and_preserves_gate_order(self):
        self.assertIn("stunetb_sourcestrat_e40_e80_20260826", RECOVERY)
        self.assertIn('"REVIEW_TRAINED_CANDIDATE","NO_HIDDEN_ARCHIVE"', RECOVERY)
        order = [
            RECOVERY.index('/usr/bin/bash "${fusion_controller}" frozen'),
            RECOVERY.index('/usr/bin/bash "${component_controller}"'),
            RECOVERY.index('/usr/bin/bash "${public_controller}"'),
            RECOVERY.index('/usr/bin/bash "${hidden_controller}"'),
        ]
        self.assertEqual(order, sorted(order))

    def test_recovery_is_resource_gated_and_never_uploads(self):
        self.assertIn("a>=30*1024*1024", RECOVERY)
        self.assertIn("g0<=1024 and g1<=1024", RECOVERY)
        self.assertIn('"uploaded":False', RECOVERY)
        self.assertNotIn("codabench.org", RECOVERY.lower())
        self.assertNotIn("requests.post", RECOVERY)
        self.assertNotIn("curl ", RECOVERY)

    def test_recovery_recognizes_component_saver_decision(self):
        self.assertIn('"${hidden_decision}" == CANDIDATE_SAVED', RECOVERY)
        self.assertIn("decision=CANDIDATE_SAVED", RECOVERY)
        self.assertNotIn(
            '"${hidden_decision}" == ARCHIVE_READY_FOR_MANUAL_UPLOAD', RECOVERY
        )

    def test_focus_bypasses_invalid_additive_terminal(self):
        self.assertNotIn("component_meta_recovery_after_sourcestrat_20260826", FOCUS)
        self.assertIn('while [[ ! -s "${aggressive_terminal}" ]]', FOCUS)
        self.assertIn("hidden_labels_read", FOCUS)
        self.assertIn("p.get(\"uploaded\")", FOCUS)


if __name__ == "__main__":
    unittest.main()
