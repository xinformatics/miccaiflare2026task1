import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
import sys

sys.path.insert(0, str(ROOT / "external/GF-Screen/docker_submission"))
from summarize_runtime_budget import canonical_case, summarize  # noqa: E402


def write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeBudgetTests(unittest.TestCase):
    def test_canonical_case(self):
        self.assertEqual(canonical_case("FLARE23Ts_0001_0000.nii.gz"), "FLARE23Ts_0001")
        self.assertEqual(canonical_case("FLARE23Ts_0001.npz"), "FLARE23Ts_0001")

    def test_joins_exact_50_case_stage_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [f"case_{index:04d}" for index in range(50)]
            biomed_prediction = root / "biomed_predict.json"
            gf_prediction = root / "gf_predict.json"
            write_json(
                biomed_prediction,
                {
                    "initialization_seconds": 2.0,
                    "cases": [
                        {"case": f"{case}.npz", "wall_seconds": 10.0}
                        for case in cases
                    ],
                },
            )
            write_json(
                gf_prediction,
                {
                    "model_initialization_seconds": 3.0,
                    "cases": [
                        {"case": f"{case}.nii.gz", "wall_seconds": 20.0}
                        for case in cases
                    ],
                },
            )
            base = {
                "status": "COMPLETE",
                "public_labels_read": False,
                "hidden_read": False,
                "submission_created": False,
                "uploaded": False,
            }
            biomed = root / "biomed.json"
            gf = root / "gf.json"
            stu = root / "stu.json"
            write_json(
                biomed,
                {
                    **base,
                    "exact_prompt_outputs": True,
                    "full_prediction": {
                        "path": str(biomed_prediction),
                        "sha256": hashlib.sha256(biomed_prediction.read_bytes()).hexdigest(),
                    },
                    "peak_gpu_memory_mb": {"reserved": 3000.0},
                },
            )
            write_json(
                gf,
                {
                    **base,
                    "exact_raw_route_equivalence": True,
                    "full_prediction": {
                        "path": str(gf_prediction),
                        "sha256": hashlib.sha256(gf_prediction.read_bytes()).hexdigest(),
                    },
                    "peak_gpu_memory_mb": {"reserved": 3500.0},
                },
            )
            write_json(
                stu,
                {
                    **base,
                    "exact_full_volume_proxy_equivalence": True,
                    "model_initialization_seconds": 1.0,
                    "peak_gpu_memory_mb": {"reserved": 4000.0},
                    "cases": [
                        {
                            "case": f"{case}.nii.gz",
                            "stage_seconds": 15.0,
                            "stunet_skipped": False,
                        }
                        for case in reversed(cases)
                    ],
                },
            )
            payload = summarize(
                argparse.Namespace(
                    biomedparse=biomed,
                    gf=gf,
                    stunet=stu,
                    hard_seconds=60.0,
                    target_seconds=50.0,
                    hard_memory_mb=16384.0,
                    efficiency_memory_mb=4096.0,
                )
            )
            self.assertEqual(payload["decision"], "QUALIFY_STEADY_STATE_RUNTIME")
            self.assertEqual(payload["case_count"], 50)
            self.assertEqual(payload["runtime_seconds"]["steady_state"]["max"], 45.0)
            self.assertEqual(payload["runtime_seconds"]["cold_start_conservative_max"], 51.0)
            self.assertTrue(payload["memory"]["sequential_stage_memory_eligible_4gb"])


if __name__ == "__main__":
    unittest.main()
