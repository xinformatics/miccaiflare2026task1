import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from select_materialization import select


def robust(dsc=0.0, nsd=0.0, wins=0.5, catastrophic=0.0, decision="KEEP_ANCHOR"):
    return {
        "decision": decision,
        "paired_gain": {"Lesion_DSC": dsc, "Lesion_NSD": nsd},
        "win_rate": wins,
        "catastrophic_regression_rate": catastrophic,
    }


def healthy(false_positive_cases=0, mean_false_positive_mm3=0.0, cases=32):
    return {
        "cases": cases,
        "false_positive_cases": false_positive_cases,
        "mean_false_positive_mm3": mean_false_positive_mm3,
    }


class MaterializationSelectionTests(unittest.TestCase):
    def test_strict_promotion_always_materializes(self):
        result = select(robust(decision="PROMOTE"))
        self.assertEqual(result["decision"], "PROMOTE")
        self.assertTrue(result["materialize_hidden"])

    def test_predeclared_exploratory_gate(self):
        result = select(robust(dsc=0.011, nsd=0.002, wins=0.56, catastrophic=0.02))
        self.assertEqual(result["decision"], "EXPLORATORY")
        self.assertTrue(result["materialize_hidden"])

    def test_nsd_regression_rejects_exploration(self):
        result = select(robust(dsc=0.02, nsd=-0.0001, wins=0.70))
        self.assertEqual(result["decision"], "KEEP")
        self.assertFalse(result["materialize_hidden"])

    def test_catastrophic_tail_rejects_exploration(self):
        result = select(robust(dsc=0.02, nsd=0.01, wins=0.70, catastrophic=0.06))
        self.assertEqual(result["decision"], "KEEP")

    def test_exact_thresholds_pass(self):
        result = select(robust(dsc=0.01, nsd=0.0, wins=0.52, catastrophic=0.05))
        self.assertEqual(result["decision"], "EXPLORATORY")

    def test_healthy_noninferiority_passes(self):
        result = select(
            robust(dsc=0.02, nsd=0.01, wins=0.60),
            healthy_anchor=healthy(2, 100.0),
            healthy_candidate=healthy(1, 80.0),
        )
        self.assertEqual(result["decision"], "EXPLORATORY")
        self.assertTrue(result["materialize_hidden"])

    def test_new_healthy_false_positive_rejects(self):
        result = select(
            robust(dsc=0.02, nsd=0.01, wins=0.60, decision="PROMOTE"),
            healthy_anchor=healthy(0, 0.0),
            healthy_candidate=healthy(1, 10.0),
        )
        self.assertEqual(result["decision"], "KEEP")
        self.assertFalse(result["materialize_hidden"])
        self.assertTrue(result["public_materialize_hidden"])

    def test_healthy_volume_regression_rejects(self):
        result = select(
            robust(dsc=0.02, nsd=0.01, wins=0.60),
            healthy_anchor=healthy(2, 100.0),
            healthy_candidate=healthy(2, 100.1),
        )
        self.assertEqual(result["decision"], "KEEP")

    def test_one_sided_healthy_input_rejected(self):
        with self.assertRaises(ValueError):
            select(
                robust(dsc=0.02, nsd=0.01, wins=0.60),
                healthy_anchor=healthy(),
            )


if __name__ == "__main__":
    unittest.main()
