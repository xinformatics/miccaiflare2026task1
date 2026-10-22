import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "run_candidate_refiner_v3_weak707_training_20260823.sh"
OOF = ROOT / "run_candidate_refiner_v3_weak707_fullvolume_oof_20260823.sh"
PUBLIC = ROOT / "run_candidate_refiner_v3_weak707_public_20260823.sh"
SAVE = ROOT / "save_candidate_refiner_v3_weak707_candidate_20260823.sh"


class Weak707DeploymentSourceSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = {
            name: path.read_text(encoding="utf-8")
            for name, path in {
                "train": TRAIN,
                "oof": OOF,
                "public": PUBLIC,
                "save": SAVE,
            }.items()
        }

    def test_all_twelve_source_identities_are_wired_end_to_end(self):
        fixed = ("epoch105", "currentgf", "autopet_weak", "lndb_weak")
        for stage, text in self.text.items():
            for source in fixed:
                self.assertIn(source, text, f"{stage} is missing {source}")
            self.assertIn("lung_cell${colour}", text, f"{stage} lacks cell loop")
            self.assertIn("0 1 2 3 4 5 6 7", text, f"{stage} lacks eight cells")

    def test_candidate_identity_and_geometry_match(self):
        for stage, text in self.text.items():
            self.assertIn("--component-policy source_separate", text, stage)
            self.assertIn('--bbox-margin-mm "${bbox_margin}"', text, stage)
            self.assertIn("--tile-stride-mm 72", text, stage)
            self.assertIn("--min-mm3 5", text, stage)

    def test_public_and_hidden_recreate_frozen_cell_lattice(self):
        for stage in ("public", "save"):
            text = self.text[stage]
            self.assertIn("generate_lung_cell_proposals.py", text)
            self.assertIn("--target-spacing-mm 2.5", text)
            self.assertIn("--cell-stride-mm 48", text)
            self.assertIn("--resume", text)

    def test_public_and_hidden_recreate_both_weak_prompt_banks(self):
        for stage in ("public", "save"):
            text = self.text[stage]
            self.assertIn("run_selected_weak_prompt_bank.sh", text)
            self.assertIn("run_bank autopet", text)
            self.assertIn("run_bank lndb", text)
            self.assertIn("weak_domain_biomedparse_707_selected_20260823", text)

    def test_deployment_threshold_map_requires_exact_source_inventory(self):
        expected_literal = (
            'expected={"epoch105","currentgf","autopet_weak","lndb_weak"} '
            '| {f"lung_cell{i}" for i in range(8)}'
        )
        for stage in ("oof", "public", "save"):
            self.assertIn(expected_literal, self.text[stage], stage)

    def test_training_budget_and_dynamic_two_gpu_dispatch_are_explicit(self):
        train = self.text["train"]
        self.assertIn("--samples-per-epoch 8192", train)
        # Fold 0 now enters the same dynamic allocator as folds 1--4 so an
        # already-free second device can be claimed immediately.
        self.assertIn("train_remaining_dynamic 0 1 2 3 4", train)
        self.assertIn('used=$(gpu_memory_mib "${gpu}")', train)
        self.assertIn('job_pid[${gpu}]=$!', train)

    def test_exact_full_volume_oof_calibrates_contour_before_deployment_grid(self):
        oof = self.text["oof"]
        self.assertIn("deployment_segmentation_thresholds", oof)
        self.assertIn("segmentation_threshold_sweep.tsv", oof)
        self.assertIn("select_segmentation_threshold_sweep.py", oof)
        self.assertIn("--expected-cases 707", oof)
        self.assertLess(
            oof.index("SEGMENTATION_THRESHOLD_SELECTION.json"),
            oof.index('materialization_grid="${output}/materialization_grid.json"'),
        )

    def test_long_deployment_cache_builders_are_restart_safe(self):
        for stage in ("oof", "public", "save"):
            self.assertIn("--resume", self.text[stage], stage)


if __name__ == "__main__":
    unittest.main()
