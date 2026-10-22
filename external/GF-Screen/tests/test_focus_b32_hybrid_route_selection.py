import unittest

from focus_b32_hybrid_route_selection import records


class FocusB32HybridRouteSelectionTests(unittest.TestCase):
    def test_emits_five_focus_checkpoints_and_complete_source_map(self):
        sources = {"epoch105", "currentgf", "lung_cell0"}
        summary = {
            "index_sha256": "same",
            "input_mode": "soft_lung_seed",
            "fold_records": [
                {"fold": fold, "checkpoint": f"focus{fold}.pt"}
                for fold in range(5)
            ],
        }
        decision = {
            "decision": "QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME",
            "index_sha256": "same",
            "focus_input_mode": "soft_lung_seed",
            "source_to_route": {
                "epoch105": "b32",
                "currentgf": "b32",
                "lung_cell0": "focus",
            },
        }
        result = records(summary, decision, sources)
        self.assertEqual(result[0], "INPUT\t-\t-\tsoft_lung_seed")
        self.assertEqual(sum(row.startswith("FOCUS\t") for row in result), 5)
        self.assertEqual(sum(row.startswith("SOURCE\t") for row in result), 3)

    def test_rejects_single_family_or_incomplete_source_mapping(self):
        summary = {
            "index_sha256": "same",
            "input_mode": "soft",
            "fold_records": [
                {"fold": fold, "checkpoint": f"focus{fold}.pt"}
                for fold in range(5)
            ],
        }
        decision = {
            "decision": "QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME",
            "index_sha256": "same",
            "focus_input_mode": "soft",
            "source_to_route": {"epoch105": "focus"},
        }
        with self.assertRaisesRegex(ValueError, "coverage"):
            records(summary, decision, {"epoch105", "currentgf"})
        with self.assertRaisesRegex(ValueError, "both"):
            records(summary, decision, {"epoch105"})


if __name__ == "__main__":
    unittest.main()
