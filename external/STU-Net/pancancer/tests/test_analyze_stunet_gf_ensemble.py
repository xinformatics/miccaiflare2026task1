import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


PANCANCER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANCANCER))
from analyze_stunet_gf_ensemble import (
    confirmed_anchor_components,
    configurations,
    evaluate,
    high_value_diagnostic,
    load_compact_probability,
    select_deployment,
    selected_components,
)


class AnalyzeSTUNetGFEnsembleTest(unittest.TestCase):
    def test_components_use_physical_volume_and_six_connectivity(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[0, 0, 0] = True
        mask[1, 1, 1] = True
        mask[3:5, 3:5, 3:5] = True
        selected = selected_components(mask, voxel_mm3=100.0, minimum_mm3=500.0)
        self.assertFalse(selected[0, 0, 0])
        self.assertFalse(selected[1, 1, 1])
        self.assertTrue(selected[3:5, 3:5, 3:5].all())

    def test_support_fraction_keeps_whole_stunet_component(self):
        mask = np.zeros((5, 5, 5), dtype=bool);mask[1:3, 1:3, 1:3] = True
        support = np.zeros_like(mask);support[1, 1, 1] = True
        selected = selected_components(
            mask, voxel_mm3=100.0, minimum_mm3=500.0,
            support=support, minimum_support_fraction=0.10,
        )
        self.assertTrue(selected[1:3, 1:3, 1:3].all())

    def test_anchor_confirmation_is_whole_component_and_26_connected(self):
        anchor = np.zeros((5, 5, 5), dtype=bool)
        anchor[0, 0, 0] = True
        anchor[1, 1, 1] = True
        anchor[4, 4, 4] = True
        stunet = np.zeros_like(anchor); stunet[0, 0, 0] = True
        freetumor = np.zeros_like(anchor); freetumor[4, 4, 4] = True
        either = confirmed_anchor_components(anchor, stunet, freetumor, "either")
        both = confirmed_anchor_components(anchor, stunet, freetumor, "both")
        self.assertTrue(either[0, 0, 0] and either[1, 1, 1] and either[4, 4, 4])
        self.assertFalse(both.any())

    @staticmethod
    def save(path: Path, array: np.ndarray) -> None:
        affine = np.diag([10.0, 10.0, 10.0, 1.0])
        nib.save(nib.Nifti1Image(array.astype(np.uint8), affine), path)

    def test_independent_supported_recovery_can_clear_public_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dirs = {name: root / name for name in ("labels", "anchor", "stunet", "freetumor")}
            for directory in dirs.values(): directory.mkdir()
            for index in range(2):
                name=f"case_{index}.nii.gz";gt=np.zeros((8,8,8),dtype=np.uint8);gt[2:4,2:4,2:4]=1
                anchor=gt.copy() if index==0 else np.zeros_like(gt)
                stunet=gt.copy() if index==1 else np.zeros_like(gt)
                freetumor=stunet.copy()
                for role,array in (("labels",gt),("anchor",anchor),("stunet",stunet),("freetumor",freetumor)):
                    self.save(dirs[role]/name,array)
            result=evaluate(dirs["labels"],dirs["anchor"],dirs["stunet"],dirs["freetumor"])
        self.assertEqual(result["best_deployment"]["Lesion_DSC"],1.0)
        self.assertEqual(result["best_deployment"]["Lesion_NSD"],1.0)
        self.assertTrue(result["public_gate"])
        self.assertEqual(result["decision"],"QUALIFY_SUBMISSION_PREPARATION")

    def test_probability_calibrated_union_is_scored_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dirs = {name: root / name for name in ("labels", "anchor", "stunet", "freetumor")}
            for directory in dirs.values(): directory.mkdir()
            for index in range(2):
                name=f"case_{index}.nii.gz";gt=np.zeros((8,8,8),dtype=np.uint8);gt[2:4,2:4,2:4]=1
                anchor=gt.copy() if index==0 else np.zeros_like(gt)
                stunet=np.zeros_like(gt);freetumor=np.zeros_like(gt)
                for role,array in (("labels",gt),("anchor",anchor),("stunet",stunet),("freetumor",freetumor)):
                    self.save(dirs[role]/name,array)
                probability=np.zeros_like(gt,dtype=np.float32)
                if index==1: probability[2:4,2:4,2:4]=0.65
                np.savez_compressed(
                    dirs["stunet"]/f"case_{index}.prob16.npz",
                    lesion_probability_uint16=np.rint(probability*65535).astype(np.uint16),
                )
            result=evaluate(
                dirs["labels"],dirs["anchor"],dirs["stunet"],dirs["freetumor"],
                stunet_probabilities_dir=dirs["stunet"],
            )
        self.assertEqual(result["best_deployment"]["Lesion_DSC"],1.0)
        self.assertTrue(result["best_deployment"]["configuration"].startswith("gf_union_stu_p"))
        self.assertEqual(result["probability_thresholds"],[0.3,0.4,0.6,0.7])

    def test_case_metric_cache_resumes_and_invalidates_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dirs = {
                name: root / name
                for name in ("labels", "anchor", "stunet", "freetumor")
            }
            for directory in dirs.values():
                directory.mkdir()
            name = "case_0.nii.gz"
            gt = np.zeros((8, 8, 8), dtype=np.uint8)
            gt[2:4, 2:4, 2:4] = 1
            for role, array in (
                ("labels", gt),
                ("anchor", np.zeros_like(gt)),
                ("stunet", gt),
                ("freetumor", gt),
            ):
                self.save(dirs[role] / name, array)
            cache = root / "case_metrics"
            first = evaluate(
                dirs["labels"], dirs["anchor"], dirs["stunet"],
                dirs["freetumor"], case_cache_dir=cache,
            )
            second = evaluate(
                dirs["labels"], dirs["anchor"], dirs["stunet"],
                dirs["freetumor"], case_cache_dir=cache,
            )
            self.save(dirs["stunet"] / name, np.zeros_like(gt))
            third = evaluate(
                dirs["labels"], dirs["anchor"], dirs["stunet"],
                dirs["freetumor"], case_cache_dir=cache,
            )
            cached_payload = json.loads(
                next(cache.glob("*.metrics.json")).read_text(encoding="utf-8")
            )
        self.assertEqual(first["case_metric_cache"]["written"], 1)
        self.assertEqual(first["case_metric_cache"]["reused"], 0)
        self.assertEqual(second["case_metric_cache"]["written"], 0)
        self.assertEqual(second["case_metric_cache"]["reused"], 1)
        self.assertEqual(third["case_metric_cache"]["written"], 1)
        self.assertEqual(third["case_metric_cache"]["reused"], 0)
        self.assertLess(
            cached_payload["diagnostic"]["unique_metric_masks"],
            cached_payload["diagnostic"]["metric_configurations"],
        )

    def test_standalone_and_filtered_standalone_are_deployments(self):
        anchor = np.zeros((5, 5, 5), dtype=bool)
        stunet = np.zeros_like(anchor); stunet[1:3, 1:3, 1:3] = True
        freetumor = stunet.copy()
        masks = configurations(anchor, stunet, freetumor, voxel_mm3=100.0)
        self.assertTrue(np.array_equal(masks["stunet_standalone"], stunet))
        self.assertTrue(np.array_equal(masks["stunet_mm500"], stunet))
        self.assertTrue(np.array_equal(masks["stunet_ftu_s10_mm500"], stunet))
        self.assertIn("gf_anchor_confirm_either_any", masks)

    def test_probability_threshold_family_is_closed_and_dsc_oriented(self):
        anchor = np.zeros((5, 5, 5), dtype=bool)
        stunet = np.zeros_like(anchor)
        freetumor = np.zeros_like(anchor)
        probability = np.zeros(anchor.shape, dtype=np.float32)
        probability[1:4, 1:4, 1:4] = 0.65
        masks = configurations(
            anchor, stunet, freetumor, voxel_mm3=100.0,
            lesion_probability=probability,
        )
        self.assertTrue(masks["stunet_p60_mm1000"].any())
        self.assertFalse(masks["stunet_p70_mm1000"].any())
        self.assertIn("gf_union_stu_p30_mm3000", masks)
        self.assertIn("gf_union_stu_p30_ftu_s10_mm3000", masks)
        self.assertNotIn("gf_union_stu_p60_ftu_s10_mm3000", masks)
        self.assertIn("gf_anchor_confirm_either_p70", masks)
        self.assertNotIn("gf_anchor_confirm_either_p60", masks)

    def test_closed_only_scores_exactly_the_decision_candidates(self):
        anchor = np.zeros((5, 5, 5), dtype=bool)
        anchor[1:3, 1:3, 1:3] = True
        stunet = anchor.copy()
        freetumor = np.zeros_like(anchor)
        probability = np.zeros(anchor.shape, dtype=np.float32)
        probability[1:3, 1:3, 1:3] = 0.75
        masks = configurations(
            anchor,
            stunet,
            freetumor,
            voxel_mm3=100.0,
            lesion_probability=probability,
            closed_only=True,
        )
        self.assertEqual(
            set(masks),
            {"anchor", "stunet_standalone", "gf_anchor_confirm_either_p70"},
        )

    def test_compact_probability_loader_rejects_non_uint16(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case.prob16.npz"
            np.savez_compressed(path, lesion_probability_uint16=np.zeros((2, 3, 4), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "invalid compact"):
                load_compact_probability(path, (2, 3, 4))

    def test_high_value_diagnostic_separates_recovery_and_component_quality(self):
        gt=np.zeros((10,10,10),dtype=bool);gt[1:3,1:3,1:3]=True
        anchor=np.zeros_like(gt)
        good=gt.copy()
        bad=np.zeros_like(gt);bad[6:8,6:8,6:8]=True
        result=high_value_diagnostic(
            gt,anchor,{"good":good,"bad":bad},voxel_mm3=1000.0
        )
        self.assertEqual(len(result["missed"]),1)
        self.assertTrue(result["missed"][0]["good"])
        self.assertFalse(result["missed"][0]["bad"])
        self.assertEqual(result["candidate_components"]["good"],{"total":1,"gt_supported":1})
        self.assertEqual(result["candidate_components"]["bad"],{"total":1,"gt_supported":0})

    def test_selection_uses_best_dsc_among_gate_qualified_rows(self):
        rows = [
            {"configuration":"high_dsc_low_nsd","Lesion_DSC":0.650,"Lesion_NSD":0.49},
            {"configuration":"qualified","Lesion_DSC":0.620,"Lesion_NSD":0.52},
            {"configuration":"anchorish","Lesion_DSC":0.610,"Lesion_NSD":0.60},
        ]
        selected, unconstrained, gate = select_deployment(rows)
        self.assertTrue(gate)
        self.assertEqual(selected["configuration"],"qualified")
        self.assertEqual(unconstrained["configuration"],"high_dsc_low_nsd")

    def test_grid_is_predeclared_and_no_submission_authority(self):
        source=(PANCANCER/"analyze_stunet_gf_ensemble.py").read_text(encoding="utf-8")
        self.assertIn("MINIMUM_COMPONENT_MM3 = (500, 1000, 3000)",source)
        self.assertIn("SUPPORT_FRACTIONS = (0.01, 0.10)",source)
        self.assertIn("PROBABILITY_THRESHOLDS = (0.30, 0.40, 0.60, 0.70)", source)
        self.assertIn("ANCHOR_CONFIRMATION_PROBABILITY_THRESHOLDS = (0.70,)", source)
        self.assertIn("ProcessPoolExecutor",source)
        self.assertIn('parser.add_argument("--workers", default=8',source)
        self.assertIn('if key == "anchor"', source)
        self.assertIn('"submission_created": False',source)
        self.assertNotIn("Validation-Hidden",source)


if __name__ == "__main__":
    unittest.main()
