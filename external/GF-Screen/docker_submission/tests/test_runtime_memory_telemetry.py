from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4]
SOURCES = (
    ROOT / "external/BiomedParse/miccai_dual_prompt_inference.py",
    ROOT / "external/GF-Screen/docker_submission/predict_dual_glance_shared_focus.py",
    ROOT / "external/STU-Net/pancancer/predict_stunet_proxy_sparse.py",
)


class RuntimeMemoryTelemetryTests(unittest.TestCase):
    def test_each_deployment_stage_resets_and_records_cuda_peak_memory(self):
        for source in SOURCES:
            with self.subTest(source=source.name):
                script = source.read_text(encoding="utf-8")
                self.assertIn("torch.cuda.reset_peak_memory_stats", script)
                self.assertIn("torch.cuda.max_memory_allocated", script)
                self.assertIn("torch.cuda.max_memory_reserved", script)
                self.assertIn('"peak_gpu_allocated_mb"', script)
                self.assertIn('"peak_gpu_reserved_mb"', script)


if __name__ == "__main__":
    unittest.main()
