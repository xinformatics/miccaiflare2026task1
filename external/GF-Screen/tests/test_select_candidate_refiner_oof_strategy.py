import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from select_candidate_refiner_oof_strategy import choose, eligibility, file_sha256, main


def summary(ap, prevalence, precision, recall, dice):
    return {
        "classification": {
            "average_precision": ap,
            "prevalence": prevalence,
            "precision": precision,
            "recall": recall,
        },
        "segmentation": {"dice": dice},
    }


class CandidateRefinerOOFStrategyTests(unittest.TestCase):
    def test_preserves_historical_three_strategy_contract(self):
        candidates = {
            "composite": summary(0.80, 0.20, 0.70, 0.50, 0.55),
            "segmentation": summary(0.72, 0.20, 0.65, 0.45, 0.62),
            "classification": summary(0.88, 0.20, 0.80, 0.55, 0.48),
        }
        self.assertEqual(choose(candidates), "segmentation")

    def test_prefers_segmentation_once_classification_is_adequate(self):
        candidates = {
            "composite": summary(0.80, 0.20, 0.70, 0.50, 0.55),
            "segmentation": summary(0.72, 0.20, 0.65, 0.45, 0.62),
            "cohort_segmentation": summary(0.74, 0.20, 0.66, 0.46, 0.60),
            "classification": summary(0.88, 0.20, 0.80, 0.55, 0.48),
        }
        self.assertEqual(choose(candidates), "segmentation")

    def test_rejects_high_dice_with_inadequate_classifier(self):
        candidates = {
            "composite": summary(0.70, 0.20, 0.65, 0.45, 0.55),
            "segmentation": summary(0.30, 0.20, 0.40, 0.10, 0.80),
            "cohort_segmentation": summary(0.68, 0.20, 0.62, 0.42, 0.54),
            "classification": summary(0.85, 0.20, 0.80, 0.55, 0.50),
        }
        self.assertFalse(eligibility(candidates["segmentation"]))
        self.assertEqual(choose(candidates), "composite")

    def test_can_select_cohort_balanced_checkpoint_on_global_oof_evidence(self):
        candidates = {
            "composite": summary(0.80, 0.20, 0.70, 0.50, 0.55),
            "segmentation": summary(0.72, 0.20, 0.65, 0.45, 0.60),
            "cohort_segmentation": summary(0.75, 0.20, 0.68, 0.48, 0.64),
            "classification": summary(0.88, 0.20, 0.80, 0.55, 0.48),
        }
        self.assertEqual(choose(candidates), "cohort_segmentation")

    def test_rejects_incomplete_strategy_sets(self):
        with self.assertRaises(ValueError):
            choose({"composite": summary(0.8, 0.2, 0.7, 0.5, 0.5)})

    def test_main_hash_binds_inputs_and_writes_selected_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = {
                "composite": summary(0.80, 0.20, 0.70, 0.50, 0.55),
                "segmentation": summary(0.72, 0.20, 0.65, 0.45, 0.60),
                "cohort_segmentation": summary(0.75, 0.20, 0.68, 0.48, 0.64),
                "classification": summary(0.88, 0.20, 0.80, 0.55, 0.48),
            }
            paths = {}
            for name, payload in candidates.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = path
            output = root / "selection.json"
            selected = root / "selected.json"
            argv = ["selector"]
            for name, path in paths.items():
                argv.extend(("--candidate", f"{name}={path}"))
            argv.extend(("--output", str(output), "--selected-summary", str(selected)))
            with patch.object(sys, "argv", argv):
                main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["selected_strategy"], "cohort_segmentation")
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(
                report["strategies"]["cohort_segmentation"]["summary_sha256"],
                file_sha256(paths["cohort_segmentation"]),
            )
            self.assertEqual(
                json.loads(selected.read_text(encoding="utf-8"))[
                    "checkpoint_selection"
                ]["implementation_sha256"],
                report["implementation_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
