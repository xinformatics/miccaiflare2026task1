import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


GF_ROOT = Path(__file__).resolve().parents[1]
DOCKER_ROOT = GF_ROOT / "docker_submission"
sys.path.insert(0, str(DOCKER_ROOT))

from runtime_winner_mask_recipe import (  # noqa: E402
    compose_runtime_anchor,
    confirm_with_stunet_p70_or_existing_support_proxy,
    existing_support_proxy_mask,
)
from materialize_runtime_proxy_reference import validate_public_only_path  # noqa: E402


class RuntimeWinnerMaskRecipeTests(unittest.TestCase):
    def test_runtime_anchor_preserves_complete_components_at_each_frozen_stage(self):
        anchor_source = np.zeros((1, 1, 40), dtype=bool)
        anchor_source[0, 0, 0:10] = True
        anchor_source[0, 0, 12:22] = True
        whole = np.zeros_like(anchor_source)
        whole[0, 0, 0] = True
        tumor = np.zeros_like(anchor_source)
        guided = np.zeros_like(anchor_source)
        guided[0, 0, 24:29] = True
        whole[0, 0, 24:29] = True
        cbg = np.zeros_like(anchor_source)
        cbg[0, 0, 0:3] = True
        cbg[0, 0, 8:13] = True
        result, stats = compose_runtime_anchor(
            bbox_raw=anchor_source,
            prompt_whole_body_lesion=whole,
            prompt_tumor=tumor,
            guided_gf=guided,
            cbg_gf=cbg,
            spacing=(10.0, 10.0, 1.0),
        )
        expected = np.zeros_like(result)
        expected[0, 0, 0:13] = True
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(stats.bbox_filtered_voxels, 20)
        self.assertEqual(stats.prompt_confirmed_voxels, 10)
        self.assertEqual(stats.guided_voxels, 15)
        self.assertEqual(stats.final_anchor_voxels, 13)

    def test_all_three_frozen_proxy_branches_and_stu_keep_whole_components(self):
        shape = (1, 1, 59)
        anchor = np.zeros(shape, dtype=bool)
        starts = (0, 12, 24, 36, 48)
        for start in starts:
            anchor[0, 0, start : start + 10] = True
        whole = np.zeros(shape, dtype=bool)
        tumor = np.zeros(shape, dtype=bool)
        bbox = np.zeros(shape, dtype=bool)
        probability = np.zeros(shape, dtype=np.uint16)
        probability[0, 0, 0] = 65535
        tumor[0, 0, 12:16] = True
        whole[0, 0, 24:28] = True
        bbox[0, 0, 24:28] = True
        whole[0, 0, 36:44] = True
        result, stats = confirm_with_stunet_p70_or_existing_support_proxy(
            anchor=anchor,
            prompt_whole_body_lesion=whole,
            prompt_tumor=tumor,
            bbox_raw=bbox,
            stunet_probability_uint16=probability,
            spacing=(5.0, 2.0, 10.0),
        )
        expected = anchor.copy()
        expected[0, 0, 48:58] = False
        self.assertTrue(np.array_equal(result, expected))
        self.assertEqual(stats.anchor_components, 5)
        self.assertEqual(stats.stunet_confirmed_components, 1)
        self.assertEqual(stats.proxy_confirmed_components, 3)
        self.assertEqual(stats.rejected_components, 1)
        proxy = existing_support_proxy_mask(
            anchor=anchor,
            prompt_whole_body_lesion=whole,
            prompt_tumor=tumor,
            bbox_raw=bbox,
            spacing=(5.0, 2.0, 10.0),
        )
        self.assertFalse(proxy[0, 0, 0:10].any())
        self.assertTrue(proxy[0, 0, 12:46].sum() == 30)
        self.assertFalse(proxy[0, 0, 48:58].any())

    def test_strict_stu_threshold_and_input_contract(self):
        anchor = np.ones((1, 1, 1), dtype=bool)
        threshold = np.asarray([[[round(0.70 * 65535)]]], dtype=np.uint16)
        result, _ = confirm_with_stunet_p70_or_existing_support_proxy(
            anchor=anchor,
            prompt_whole_body_lesion=np.zeros_like(anchor),
            prompt_tumor=np.zeros_like(anchor),
            bbox_raw=anchor,
            stunet_probability_uint16=threshold,
            spacing=(1.0, 1.0, 1.0),
        )
        self.assertFalse(result.any())
        with self.assertRaises(ValueError):
            confirm_with_stunet_p70_or_existing_support_proxy(
                anchor=anchor,
                prompt_whole_body_lesion=np.zeros_like(anchor),
                prompt_tumor=np.zeros_like(anchor),
                bbox_raw=anchor,
                stunet_probability_uint16=threshold.astype(np.float32),
                spacing=(1.0, 1.0, 1.0),
            )

    def test_frozen_public_proxy_is_near_equivalent_but_not_quality_promoted(self):
        evidence = json.loads(
            (
                DOCKER_ROOT
                / "analysis/free_teacher_proxy_public_metrics_20260827.json"
            ).read_text(encoding="utf-8")
        )
        proxy = evidence["aggregates"]["high_precision_proxy_v1"]
        self.assertEqual(evidence["decision"], "KEEP_AS_RUNTIME_ABLATION_ONLY")
        self.assertFalse(proxy["quality_gate_pass"])
        self.assertLess(abs(proxy["delta_DSC_vs_historical_or"]), 0.0002)
        self.assertLess(abs(proxy["delta_NSD_vs_historical_or"]), 0.0003)
        self.assertEqual(
            evidence["teacher_counts"]["high_precision_proxy_v1"],
            {
                "teacher_unique_components_recovered": 3,
                "teacher_rejected_components_recovered": 0,
            },
        )

    def test_sparse_runtime_predictor_has_no_freetumor_input(self):
        source = (
            GF_ROOT.parent
            / "STU-Net/pancancer/predict_stunet_proxy_sparse.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('parser.add_argument("--freetumor"', source)
        self.assertIn("existing_support_proxy_mask", source)
        self.assertIn("dependency_path", source)
        self.assertIn("unresolved = anchor & ~proxy", source)
        probe = (
            GF_ROOT.parent
            / "STU-Net/pancancer/run_stunet_proxy_sparse_probe_20260827.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("part_id=28", probe)
        self.assertIn("runtime_eligible_20s", probe)
        self.assertIn("biomedparse_dual_prompt_public_equivalence_20260827", probe)
        self.assertNotIn("Validation-Hidden", probe)

    def test_proxy_reference_materializer_rejects_hidden_and_label_paths(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            validate_public_only_path(root / "public_sweeps" / "reference", "output")
            with self.assertRaises(ValueError):
                validate_public_only_path(
                    root / "Validation-Hidden-Images" / "case.nii.gz", "anchor"
                )
            with self.assertRaises(ValueError):
                validate_public_only_path(
                    root / "Validation-Public-Labels" / "case.nii.gz", "anchor"
                )

    def test_all50_sparse_controller_is_fail_closed_and_label_free(self):
        source = (
            GF_ROOT.parent
            / "STU-Net/pancancer/run_stunet_proxy_sparse_all50_20260827.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("exact_full_volume_proxy_equivalence", source)
        self.assertIn("max(times)<=20.0", source)
        self.assertIn("flock -n 9", source)
        self.assertIn("waiting for safe host", source)
        self.assertIn("max(used0,used1)<=1024", source)
        self.assertIn("available>=30*1024*1024", source)
        self.assertNotIn("Validation-Public-Labels", source)
        self.assertNotIn("Validation-Hidden", source)

    def test_runtime_anchor_replay_is_public_only_and_exact(self):
        materializer = (DOCKER_ROOT / "materialize_runtime_anchor.py").read_text(
            encoding="utf-8"
        )
        controller = (
            DOCKER_ROOT / "run_runtime_anchor_equivalence_20260827.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("compose_runtime_anchor", materializer)
        self.assertIn("validate_public_only_path", materializer)
        self.assertIn("exact_anchor_equivalence", controller)
        self.assertIn("three_route_shared_focus_public_equivalence_20260827", controller)
        self.assertNotIn("Validation-Public-Labels", controller)
        self.assertNotIn("Validation-Hidden", controller)


if __name__ == "__main__":
    unittest.main()
