import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESENC_QUEUE = ROOT / "nnunet" / "queue_resenc_s6_partialbox_after_focus_20260827.sh"
GLANCE_QUEUE = ROOT / "external" / "GF-Screen" / "run_long_glance_grl_after_focus_20260825.sh"


class ResEncGpuPriorityHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resenc = RESENC_QUEUE.read_text(encoding="utf-8")
        cls.glance = GLANCE_QUEUE.read_text(encoding="utf-8")

    def test_resenc_waits_for_focus_training_and_save_consumers(self):
        self.assertIn("gf_focus_pu_long30_e283_to313_20260825/TERMINAL.json", self.resenc)
        self.assertIn("gf_focus_pu_long30_trajectory_saved_20260825/TERMINAL.json", self.resenc)
        self.assertIn('while [[ ! -s "${focus}" || ! -s "${focus_saved}" ]]', self.resenc)

    def test_long_glance_cannot_race_resenc_for_idle_gpus(self):
        self.assertIn(
            "resenc_s6_dualwindow_endpoints_20260827/TERMINAL.json",
            self.glance,
        )
        self.assertIn(
            "resenc_s6_dualwindow_saved_20260827/TERMINAL.json",
            self.glance,
        )
        self.assertNotIn(
            'resenc_priority_public="${root}/../../nnunet/public_sweeps/'
            'resenc_s6_organ_target_endpoints_20260827/TERMINAL.json"',
            self.glance,
        )
        self.assertIn(
            '"${resenc_priority_public}" "${resenc_priority_saved}"; do',
            self.glance,
        )
        self.assertLess(
            self.glance.index(
                '"${resenc_priority_public}" "${resenc_priority_saved}"; do'
            ),
            self.glance.index("gpu_memory_mib()"),
        )

    def test_priority_gate_requires_a_successful_or_skipped_terminal(self):
        marker_loop = self.glance[
            self.glance.index("for marker in") : self.glance.index("for file in")
        ]
        self.assertIn("COMPLETE|SKIPPED) break", marker_loop)
        self.assertIn("waiting for replacement of failed upstream terminal", marker_loop)


if __name__ == "__main__":
    unittest.main()
