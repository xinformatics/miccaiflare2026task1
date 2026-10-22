import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
TRAIN = (ROOT / "run_component_e160_complete_hybrid_continuation_20260826.sh").read_text()
QUEUE = (ROOT / "queue_complete_hybrid_after_focus_20260826.sh").read_text()


class CompleteHybridSchedulingTests(unittest.TestCase):
    def test_exact_parent_and_two_gpu_recipe_are_bound(self):
        self.assertIn("81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74", TRAIN)
        self.assertIn("STUNetTrainerPanCancerComponentBoxE160CompleteHybrid", TRAIN)
        self.assertIn("-num_gpus 2", TRAIN)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", TRAIN)
        self.assertIn("COMPLETE_HYBRID_N_PROC_DA:-0", TRAIN)

    def test_training_is_resume_safe_and_label_role_safe(self):
        self.assertIn("checkpoint_latest.pth", TRAIN)
        self.assertIn("--c", TRAIN)
        self.assertIn("optimizer_state", TRAIN)
        self.assertIn('"public_read":False', TRAIN)
        self.assertIn('"hidden_read":False', TRAIN)
        self.assertIn('"uploaded":False', TRAIN)
        for forbidden in ("Validation-Public", "Validation-Hidden", "codabench", "curl "):
            self.assertNotIn(forbidden, TRAIN)

    def test_launch_binds_native_smoke_to_all_live_inputs(self):
        self.assertIn('inputs.get("trainer_source",{}).get("sha256")!=sha(trainer)', TRAIN)
        self.assertIn('inputs.get("runner_source",{}).get("sha256")!=sha(runner)', TRAIN)
        self.assertIn('inputs.get("audit",{}).get("sha256")!=sha(Path(sys.argv[1]))', TRAIN)
        self.assertIn('inputs.get("plans",{}).get("sha256")!=sha(plans)', TRAIN)
        self.assertIn('inputs.get("dataset_json",{}).get("sha256")!=sha(dataset)', TRAIN)
        self.assertIn('smoke.get("identifier_inventory_sha256")', TRAIN)

    def test_live_smoke_hashes_match_every_bound_input(self):
        smoke = json.loads(
            (ROOT / "analysis/complete_hybrid4x_native_loader_smoke_20260826/TERMINAL.json").read_text()
        )
        inputs = smoke["inputs"]
        paths = {
            "trainer_source": ROOT / "STUNetTrainerPanCancerCompleteHybrid.py",
            "runner_source": ROOT / "run_pancancer_stunet_complete_hybrid.py",
            "audit": ROOT / "analysis/complete_hybrid4x_inventory_20260826.json",
            "plans": PROJECT / "nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans.json",
            "dataset_json": PROJECT / "nnUNet_preprocessed/Dataset501_PanCancer/dataset.json",
        }
        for key, path in paths.items():
            self.assertTrue(path.is_file(), key)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                inputs[key]["sha256"],
                key,
            )

    def test_queue_waits_for_prioritized_additive_decision(self):
        self.assertIn("component_meta_recovery_after_sourcestrat_20260826", QUEUE)
        self.assertIn('while [[ ! -s "${additive}" ]]', QUEUE)
        self.assertIn('p.get("status") not in {"COMPLETE","SKIPPED"}', QUEUE)
        self.assertIn('p.get("hidden_labels_read")', QUEUE)
        self.assertNotIn("gf_focus_pu_long30_trajectory_saved_20260825", QUEUE)
        self.assertIn("MemoryHigh=48G", QUEUE)
        self.assertIn("MemoryMax=64G", QUEUE)
        self.assertIn("Restart=on-failure", QUEUE)

    def test_queue_and_direct_training_are_deauthorized_before_compute(self):
        self.assertIn("LABEL_CONTRACT_CORRECTION_20260825.json", QUEUE)
        self.assertIn("DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES", QUEUE)
        self.assertLess(
            QUEUE.index("DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES"),
            QUEUE.index('while [[ ! -s "${additive}" ]]'),
        )
        self.assertIn("LABEL_CONTRACT_CORRECTION_20260825.json", TRAIN)
        self.assertIn("DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES", TRAIN)
        self.assertLess(
            TRAIN.index("DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES"),
            TRAIN.index("export CUDA_VISIBLE_DEVICES=0,1"),
        )

    def test_queue_waits_for_source_stunet_precision_campaign(self):
        self.assertIn("source_stunet_precision_pipeline_20260827", QUEUE)
        self.assertIn('while [[ ! -s "${precision}" ]]', QUEUE)
        self.assertIn(
            'p.get("decision") not in {"CANDIDATE_SAVED","KEEP_EPOCH160_P70"}',
            QUEUE,
        )

    def test_queue_cannot_upload(self):
        self.assertIn('"uploaded":False', QUEUE)
        for forbidden in ("codabench.org", "requests.post", "curl ", "selenium"):
            self.assertNotIn(forbidden, QUEUE)


if __name__ == "__main__":
    unittest.main()
