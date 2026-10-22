import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
SCRIPT = (PANCANCER / "save_public_qualified_ensemble_20260825.sh").read_text(encoding="utf-8")


class SaveEnsembleSchedulingTest(unittest.TestCase):
    def test_stunet_hidden_exports_use_bounded_host_memory(self):
        self.assertIn("for part_id in $(seq 0 99)", SCRIPT)
        self.assertIn("--num-parts 100", SCRIPT)
        self.assertIn('--workers 1 --save-probabilities --sequential "${compact_args[@]}" --dataset-role hidden', SCRIPT.replace("\\\n    ", ""))
        self.assertNotIn("stunet_pids", SCRIPT)
        self.assertIn('gpu=$((part_id % 2))', SCRIPT)
        self.assertNotIn("--workers 4 --dataset-role hidden", SCRIPT)
        self.assertIn('nvidia-smi --id="${stunet_hidden_gpu}"', SCRIPT)

    def test_freetumor_hidden_exports_are_case_isolated(self):
        self.assertIn("for shard in $(seq 0 99)", SCRIPT)
        self.assertIn('gpu=$((shard % 2))', SCRIPT)
        self.assertIn("--num-shards 100 --workers 0", SCRIPT.replace("\\\n        ", ""))
        self.assertIn("--sw-batch-size 1", SCRIPT)
        self.assertIn('FREE_CASE_${shard4}.json', SCRIPT)
        self.assertIn('x.get("num_shards")!=100', SCRIPT)
        self.assertNotIn("free_pids", SCRIPT)
        self.assertNotIn("--num-shards 2 --workers 2", SCRIPT)

    def test_public_gate_precedes_hidden_transition(self):
        self.assertLess(
            SCRIPT.index('if [[ "${public_decision}" != QUALIFY_SUBMISSION_PREPARATION ]]'),
            SCRIPT.index('"${hidden_started}" <<\'PY\''),
        )
        self.assertIn("0.6154316379500749", SCRIPT)
        self.assertIn("0.5104225107769019", SCRIPT)

    def test_exact_selected_configuration_is_replayed(self):
        self.assertIn("STUNET_PUBLIC_OUTPUT", SCRIPT)
        self.assertIn("STUNET_ALLOWED_EPOCHS", SCRIPT)
        self.assertIn("gf_union_stu_ftu_s(01|10)_mm(500|1000|3000)", SCRIPT)
        self.assertIn('standalone=configuration=="stunet_standalone"', SCRIPT)
        self.assertIn("stunet_ftu_s(01|10)_mm(500|1000|3000)", SCRIPT)
        self.assertIn("stunet_p(30|40|60|70)_mm1000", SCRIPT)
        self.assertIn("gf_union_stu_p(30|40|60|70)_mm3000", SCRIPT)
        self.assertIn("gf_union_stu_p(30|40)_ftu_s10_mm3000", SCRIPT)
        self.assertIn("gf_anchor_confirm_(stu|ftu|either|both)_any", SCRIPT)
        self.assertIn("gf_anchor_confirm_either_p(70)", SCRIPT)
        self.assertIn("materialize_stunet_gf_ensemble.py", SCRIPT)
        self.assertIn('--configuration "${configuration}"', SCRIPT)
        self.assertIn("gf_touchcomp_0825.zip", SCRIPT)
        self.assertIn("checkpoint_inference_epoch_{epoch:04d}.pth", SCRIPT)

    def test_alternating_gpu_serial_inference_and_exact_archive(self):
        self.assertIn("for part_id in $(seq 0 99)", SCRIPT)
        self.assertIn('STUNET_CASE_${part4}.json', SCRIPT)
        self.assertIn('x.get("num_parts")!=100', SCRIPT)
        self.assertIn("--dataset-role hidden", SCRIPT)
        self.assertIn("--expected-count 100", SCRIPT)
        self.assertIn("len(members)!=100", SCRIPT)
        self.assertIn('"uploaded":False', SCRIPT)

    def test_no_upload_or_hidden_labels(self):
        for forbidden in ("codabench.org", "requests.post", "curl ", "selenium", "Validation-Hidden-Labels"):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
