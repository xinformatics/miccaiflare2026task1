import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "export_selected_candidate_refiner_grid.py"


class ExportSelectedGridTests(unittest.TestCase):
    def test_selected_component_configuration_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration.json"
            output = root / "grid.json"
            calibration.write_text(
                json.dumps(
                    {
                        "classification_threshold": 0.59,
                        "segmentation_threshold": 0.95,
                        "deployment_mode": "refined",
                        "selected": {
                            "classification_threshold": 0.617,
                            "segmentation_threshold": 0.9,
                            "deployment_mode": "seed_union",
                            "source_classification_thresholds": {"currentgf": 0.63},
                            "source_segmentation_thresholds": {"lung_cell0": 0.85},
                            "source_deployment_modes": {"currentgf": "refined"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--calibration",
                    str(calibration),
                    "--tag",
                    "selected",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            row = json.loads(output.read_text(encoding="utf-8"))["configurations"][0]
            self.assertEqual(row["classification_threshold"], 0.617)
            self.assertEqual(row["segmentation_threshold"], 0.9)
            self.assertEqual(row["mode"], "seed_union")
            self.assertEqual(row["source_classification_thresholds"], {"currentgf": 0.63})


if __name__ == "__main__":
    unittest.main()
