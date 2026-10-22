import hashlib
import tempfile
import unittest
from pathlib import Path

from focus_composite_route_selection import records


class FocusEncoderNoTtaRouteTests(unittest.TestCase):
    def fixture(self, root: Path):
        routes = {}
        folds = []
        for fold in range(5):
            checkpoint = root / f"fold{fold}.pt"
            checkpoint.write_bytes(f"fold-{fold}".encode())
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            folds.append(
                {
                    "fold": fold,
                    "routes": {
                        "focus_weak_sources_notta": {
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": digest,
                            "tta_views": 1,
                        }
                    },
                }
            )
        return {
            "decision": "QUALIFY_ENCODER_FOCUS_B32_NOTTA_FOR_FIXED_PUBLIC",
            "focus_input_mode": "soft_lung_seed",
            "source_to_route": {
                "epoch105": "b32",
                "autopet_weak": "focus_weak_sources_notta",
            },
            "fold_records": folds,
        }

    def test_emits_five_fold_no_tta_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.fixture(Path(directory))
            result = records(payload, {"epoch105", "autopet_weak"})
            self.assertEqual(sum(row.startswith("FOCUS\t") for row in result), 5)
            self.assertIn("SOURCE\t-\tepoch105\tb32", result)

    def test_rejects_tta_route(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.fixture(Path(directory))
            payload["fold_records"][2]["routes"]["focus_weak_sources_notta"]["tta_views"] = 3
            with self.assertRaisesRegex(ValueError, "TTA is forbidden"):
                records(payload, {"epoch105", "autopet_weak"})

    def test_rejects_incomplete_source_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "source coverage"):
                records(payload, {"epoch105"})


if __name__ == "__main__":
    unittest.main()
