import unittest
from pathlib import Path

from select_complete_hybrid_public import select_candidate


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / "run_complete_hybrid_e40_e80_public_20260826.sh").read_text()
ARMER = (ROOT / "arm_complete_hybrid_qualified_hidden_20260826.sh").read_text()


class CompleteHybridPublicSchedulingTests(unittest.TestCase):
    def test_selector_ranks_only_within_qualified_candidates(self):
        rows=[]
        for epoch in (40,80):
            for config in ("stunet_standalone","gf_anchor_confirm_either_p70"):
                rows.append({"epoch":epoch,"configuration":config,"Lesion_DSC":0.61,"Lesion_NSD":0.55})
        rows[0].update(Lesion_DSC=0.65,Lesion_NSD=0.54)
        rows[1].update(Lesion_DSC=0.70,Lesion_NSD=0.50)
        selected,qualified=select_candidate(rows)
        self.assertTrue(qualified)
        self.assertEqual(selected["configuration"],"stunet_standalone")
        self.assertEqual(selected["epoch"],40)

    def test_public_is_serial_and_has_no_hidden_or_packaging_path(self):
        self.assertIn("epochs=(40 80)",PUBLIC)
        self.assertIn("--closed-only",PUBLIC)
        self.assertIn("--sequential",PUBLIC)
        self.assertIn("--num-parts 50",PUBLIC)
        self.assertIn("stunet_standalone",PUBLIC)
        self.assertIn("gf_anchor_confirm_either_p70",PUBLIC)
        for forbidden in ("Validation-Hidden","package_codabench","codabench.org","curl "):
            self.assertNotIn(forbidden,PUBLIC)

    def test_armer_requires_both_metric_gates_and_never_uploads(self):
        self.assertIn("0.6366639257651584",ARMER)
        self.assertIn("0.5347520263176905",ARMER)
        self.assertIn("save_public_qualified_ensemble_20260825.sh",ARMER)
        self.assertIn("STUNET_COMPONENT_MODEL_DIR",ARMER)
        for forbidden in ("codabench.org","requests.post","curl ","selenium"):
            self.assertNotIn(forbidden,ARMER)


if __name__ == "__main__":
    unittest.main()
