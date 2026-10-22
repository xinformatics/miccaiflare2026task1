from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAVER = ROOT / "nnunet/save_resenc_s6_dualwindow_after_public_20260827.sh"
GLANCE = ROOT / "external/GF-Screen/run_long_glance_grl_after_focus_20260825.sh"


class GlobalResEncSaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saver = SAVER.read_text(encoding="utf-8")
        cls.glance = GLANCE.read_text(encoding="utf-8")

    def test_hidden_inference_requires_complete_frozen_selector(self):
        self.assertIn("validate_resenc_s6_global_selection.py", self.saver)
        for family in (
            "resenc_s6_partialbox_endpoints_20260827",
            "resenc_s6_positive_unlabeled_endpoints_20260827",
            "resenc_s6_organ_target_endpoints_20260827",
            "resenc_s6_dualwindow_endpoints_20260827",
        ):
            self.assertIn(family, self.saver)
        self.assertIn('if [[ "${selected_json}" == null ]]', self.saver)
        self.assertIn("NO_HIDDEN_ARCHIVE", self.saver)
        self.assertLess(
            self.saver.index('selected_json=$("${python_bin}" "${validator}"'),
            self.saver.index('"status":"STARTED"'),
        )

    def test_hidden_prediction_is_sequential_and_checkpoint_bound(self):
        self.assertIn("for part_id in $(seq 0 99)", self.saver)
        self.assertIn("--num-parts 100", self.saver)
        self.assertIn("--direct-compact-probabilities --sequential", self.saver)
        self.assertIn('p.get("checkpoint",{}).get("sha256")!=sha', self.saver)
        self.assertIn('p.get("trainer")!=trainer', self.saver)
        self.assertIn('"${python_bin}" -u "${predictor}"', self.saver)
        self.assertIn("--dataset-role hidden", self.saver)

    def test_materialization_and_archive_are_exactly_one_frozen_deployment(self):
        self.assertIn('materialize_resenc_s6_deployment.py', self.saver)
        self.assertIn('--deployment "${deployment}"', self.saver)
        self.assertIn('--expected-count 100', self.saver)
        self.assertIn(
            'submissions/rs6_${family}_e${epoch}_${short}_0827.zip',
            self.saver,
        )
        self.assertNotIn("for deployment in", self.saver)

    def test_saver_has_no_upload_capability(self):
        for forbidden in ("codabench.org", "requests.post", "selenium", "curl "):
            self.assertNotIn(forbidden, self.saver)
        self.assertIn('"uploaded":False', self.saver)

    def test_hidden_inference_rechecks_disk_headroom(self):
        self.assertIn(
            "minimum_disk_available_bytes=$((128 * 1024 * 1024 * 1024))",
            self.saver,
        )
        self.assertIn("disk_available=filesystem.f_bavail*filesystem.f_frsize", self.saver)
        self.assertIn("disk>=minimum", self.saver)
        self.assertLess(
            self.saver.index("disk>=minimum"),
            self.saver.index("for part_id in $(seq 0 99)"),
        )

    def test_long_glance_waits_for_save_terminal(self):
        public = self.glance.index('"${resenc_priority_public}"')
        saved = self.glance.index('"${resenc_priority_saved}"')
        gpu = self.glance.index("gpu_memory_mib()")
        self.assertLess(public, gpu)
        self.assertLess(saved, gpu)
        self.assertIn("resenc_s6_dualwindow_saved_20260827/TERMINAL.json", self.glance)


if __name__ == "__main__":
    unittest.main()
