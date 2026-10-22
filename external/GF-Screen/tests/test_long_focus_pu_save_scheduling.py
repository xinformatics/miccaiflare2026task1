import unittest
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / "evaluate_long_focus_pu_champion_public_20260825.sh").read_text()
TRAJECTORY = (ROOT / "evaluate_long_focus_pu_trajectory_public_20260825.sh").read_text()
SELECTOR = (ROOT / "select_long_focus_pu_trajectory.py").read_text()
SAVER = (ROOT / "save_long_focus_pu_after_public_20260825.sh").read_text()


class LongFocusPuSaveSchedulingTest(unittest.TestCase):
    def test_public_evaluators_retry_failed_terminals_only(self):
        for name, script in (("child", PUBLIC), ("trajectory", TRAJECTORY)):
            self.assertIn('COMPLETE) exit 0', script, name)
            self.assertIn('FAILED)', script, name)
            self.assertIn('TERMINAL.failed_retry_', script, name)
            self.assertNotIn('[[ -s "${terminal}" ]] && exit 0', script, name)

    def test_failed_controller_terminal_is_not_consumed_as_final(self):
        self.assertIn('COMPLETE|SKIPPED) break', SAVER)
        self.assertIn('waiting for replacement of failed long Focus controller terminal', SAVER)
        self.assertIn("trap 'failure_line=${LINENO}' ERR", SAVER)

    def test_own_failed_terminal_is_preserved_for_resume(self):
        self.assertIn('COMPLETE|SKIPPED) trap - EXIT; exit 0', SAVER)
        self.assertIn('TERMINAL.failed_retry_', SAVER)
        self.assertIn('preserved failed long Focus saver terminal for resume', SAVER)

    def test_embedded_python_blocks_compile(self):
        for name, script in (("saver", SAVER), ("trajectory", TRAJECTORY)):
            blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL)
            self.assertTrue(blocks, name)
            for index, block in enumerate(blocks):
                compile(block, f"{name}-heredoc-{index}", "exec")

    def test_hidden_inference_matches_public_model_configuration(self):
        for token in (
            "--a_min -175 --a_max 250",
            "--focus_batch_size 16",
        ):
            self.assertIn(token, PUBLIC)
            self.assertIn(token, SAVER)
        self.assertIn('--glance_threshold "${glance_threshold}" --lesion_threshold 0.50', PUBLIC)
        self.assertIn('glance_threshold="${LONG_FOCUS_GLANCE_THRESHOLD:-0.20}"', PUBLIC)
        self.assertIn("--glance_threshold 0.20 --lesion_threshold 0.50", SAVER)
        self.assertNotIn("focus_tta", SAVER.lower())

    def test_all_predeclared_deployments_are_reproducible(self):
        for variant in ("deployed500", "deployed1000", "touching"):
            self.assertIn(variant, PUBLIC)
            self.assertIn(variant, SAVER)
        for token in (
            "--fusion a",
            "biomedparse_fusion.py",
            "--min-mm3 0 --max-mm3 2000 --min-distance-mm 0",
            "--min-support-fraction 0.25 --min-support-mm3 100",
            "--support-dilation-mm inf",
            "--configuration touching_component",
        ):
            self.assertIn(token, PUBLIC)
            self.assertIn(token, SAVER)

    def test_public_identity_and_absolute_gate_are_rechecked(self):
        self.assertIn("controller/public terminal identity mismatch", SAVER)
        self.assertIn("public {key} identity mismatch", SAVER)
        self.assertIn('row["gate"]!="PROMOTE"', SAVER)
        self.assertIn("dsc<0.6154316379500749", SAVER)
        self.assertIn("nsd<0.5104225107769019", SAVER)

    def test_selected_trajectory_checkpoint_is_hash_bound(self):
        self.assertIn('public.get("predeclared_epochs")!=[293,303,313]', SAVER)
        self.assertIn("controller final checkpoint identity mismatch", SAVER)
        self.assertIn("selected child terminal identity mismatch", SAVER)
        self.assertIn("selected child checkpoint mismatch", SAVER)
        self.assertIn("selected child summary mismatch", SAVER)
        self.assertIn("selected public metric mismatch", SAVER)
        self.assertIn('checkpoint=run_dir/f"checkpoint_epoch_{epoch:04d}.pt"', SAVER)

    def test_hidden_access_starts_only_after_public_qualification(self):
        gate = SAVER.index('read -r epoch variant minimum dsc nsd checkpoint')
        marker = SAVER.index('"${python_bin}" - "${variant}" "${checkpoint}" "${hidden_started}"')
        hidden_inventory = SAVER.index('for directory in "${hidden_images}"')
        self.assertLess(gate, marker)
        self.assertLess(marker, hidden_inventory)
        self.assertIn('"hidden_read":hidden_started.is_file() or archive is not None', SAVER)
        self.assertIn('"hidden_labels_read":False', SAVER)

    def test_archive_is_exact_100_and_never_uploaded(self):
        self.assertIn("--expected-count 100", SAVER)
        self.assertIn("archive coverage mismatch", SAVER)
        self.assertIn('"uploaded":False', SAVER)
        self.assertNotIn("codabench.org", SAVER.lower())
        self.assertNotIn("curl ", SAVER.lower())
        self.assertNotIn("requests.", SAVER.lower())

    def test_trajectory_is_public_only_and_selects_one_candidate(self):
        lowered = (TRAJECTORY + SELECTOR).lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("package_codabench", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"hidden_read":false', lowered)
        self.assertIn("selected = eligible[0] if eligible else none", lowered)


if __name__ == "__main__":
    unittest.main()
