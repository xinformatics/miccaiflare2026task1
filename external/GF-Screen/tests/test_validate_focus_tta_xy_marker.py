import json
import tempfile
import unittest
from pathlib import Path

from apply_candidate_refiner_focus_tta_xy import write_tta_marker
from validate_focus_tta_xy_marker import validate


class FocusTTAXYMarkerValidationTests(unittest.TestCase):
    def make_work(self, root):
        wrapper = Path(__file__).resolve().parents[1] / "apply_candidate_refiner_focus_tta_xy.py"
        import hashlib

        wrapper_hash = hashlib.sha256(wrapper.read_bytes()).hexdigest()
        plan = {
            "inference_fingerprint": "fingerprint",
            "identity": {
                "implementation": [
                    {"path": str(wrapper.resolve()), "sha256": wrapper_hash}
                ]
            },
        }
        complete = {"inference_fingerprint": "fingerprint"}
        (root / "INFERENCE_PLAN.json").write_text(json.dumps(plan))
        (root / "INFERENCE_COMPLETE.json").write_text(json.dumps(complete))
        write_tta_marker(root, ["focus_weak_sources_tta_xy"])

    def test_accepts_exact_route_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_work(root)
            checks = validate(root, {"focus_weak_sources_tta_xy"})
        self.assertTrue(all(checks.values()))

    def test_rejects_wrong_expected_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_work(root)
            checks = validate(root, {"focus_lung_cells_tta_xy"})
        self.assertFalse(checks["routes"])

    def test_rejects_modified_inference_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_work(root)
            (root / "INFERENCE_COMPLETE.json").write_text(
                json.dumps({"inference_fingerprint": "changed"})
            )
            checks = validate(root, {"focus_weak_sources_tta_xy"})
        self.assertFalse(checks["inference_hash"])
        self.assertFalse(checks["fingerprint"])


if __name__ == "__main__":
    unittest.main()
