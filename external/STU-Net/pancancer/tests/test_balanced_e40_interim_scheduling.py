import unittest
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / "run_balanced_e40_e80_public_20260826.sh").read_text()


class BalancedE40InterimSchedulingTests(unittest.TestCase):
    def test_interim_is_epoch40_only_and_does_not_finalize_selector(self):
        self.assertIn('interim_e40_only="${BALANCED_INTERIM_E40_ONLY:-0}"', SCRIPT)
        self.assertIn('epochs=(40)', SCRIPT)
        self.assertIn('E40_INTERIM_COMPLETE.json', SCRIPT)
        self.assertIn('INTERIM_E40_PUBLIC_READY_TRAINING_MAY_RESUME', SCRIPT)
        self.assertIn('if [[ "${interim_e40_only}" == 1 ]]', SCRIPT)
        interim_exit = SCRIPT.index('decision":"INTERIM_E40_PUBLIC_READY_TRAINING_MAY_RESUME')
        final_selector = SCRIPT.index('"predeclared_epochs":[40,80]')
        self.assertLess(interim_exit, final_selector)

    def test_interim_keeps_hidden_and_upload_paths_absent(self):
        self.assertNotIn("Validation-Hidden", SCRIPT)
        self.assertNotIn("codabench.org", SCRIPT)
        self.assertIn('"hidden_read":False,"submission_created":False,"uploaded":False', SCRIPT)


if __name__ == "__main__":
    unittest.main()
