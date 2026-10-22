from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from nnunet.select_resenc_s6_public import load_policy
from nnunet.validate_resenc_s6_global_selection import (
    DEPLOYMENTS,
    INCUMBENT,
    QUALIFICATION,
    FamilySpec,
    family_specs,
    select_global_candidate,
    sha256_file,
    validate_family_selection,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_POLICY = ROOT / "nnunet/RESENC_S6_PUBLIC_SELECTION_POLICY_20260827.json"
BASE_PREDICTOR = ROOT / "nnunet/predict_resenc_s6_safe.py"


class GlobalResEncSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_spec(self, name: str, dsc: float = 0.65, nsd: float = 0.56) -> FamilySpec:
        model = self.root / name / "model"
        fold = model / "fold_all"
        fold.mkdir(parents=True)
        checkpoints = {}
        for epoch in (100, 200, 300, 400, 500):
            filename = (
                "checkpoint_final.pth"
                if epoch == 500
                else f"checkpoint_epoch_{epoch:04d}.pth"
            )
            checkpoint = fold / filename
            checkpoint.write_bytes(f"{name}-checkpoint-{epoch}".encode())
            checkpoints[epoch] = checkpoint
        results = []
        for epoch in (100, 200, 300, 400, 500):
            for deployment in DEPLOYMENTS:
                is_selected = epoch == 100 and deployment == "resenc_standalone_p50"
                row_dsc = dsc if is_selected else 0.60
                row_nsd = nsd if is_selected else 0.50
                results.append(
                    {
                        "epoch": epoch,
                        "deployment": deployment,
                        "analysis_configuration": deployment,
                        "Lesion_DSC": row_dsc,
                        "Lesion_NSD": row_nsd,
                        "DSC_gain": row_dsc - INCUMBENT["Lesion_DSC"],
                        "NSD_gain": row_nsd - INCUMBENT["Lesion_NSD"],
                        "qualified": is_selected,
                        "checkpoint": {
                            "path": str(checkpoints[epoch].resolve()),
                            "sha256": sha256_file(checkpoints[epoch]),
                        },
                    }
                )
        selected = results[0]
        terminal = self.root / name / "TERMINAL.json"
        terminal.parent.mkdir(parents=True, exist_ok=True)
        terminal.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "decision": "REVIEW_TRAINED_CANDIDATE",
                    "policy": {
                        "path": str(BASE_POLICY.resolve()),
                        "sha256": sha256_file(BASE_POLICY),
                    },
                    "predeclared_epochs": [100, 200, 300, 400, 500],
                    "predeclared_deployments": list(DEPLOYMENTS),
                    "results": results,
                    "selected": selected,
                    "public_read": True,
                    "hidden_read": False,
                    "submission_created": False,
                    "uploaded": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return FamilySpec(
            name=name,
            trainer=f"trainer-{name}",
            epochs=(100, 200, 300, 400, 500),
            final_epoch=500,
            terminal=terminal,
            policy=BASE_POLICY,
            model_dir=model,
            predictor=BASE_PREDICTOR,
            policy_loader=load_policy,
        )

    def test_accepts_exact_qualified_family_candidate(self):
        candidate = validate_family_selection(self.make_spec("base"))
        self.assertEqual(candidate.family, "base")
        self.assertEqual(candidate.epoch, 100)
        self.assertEqual(candidate.deployment, "resenc_standalone_p50")

    def test_no_archive_family_contributes_no_candidate(self):
        spec = self.make_spec("base")
        terminal = json.loads(spec.terminal.read_text(encoding="utf-8"))
        terminal["results"][0]["Lesion_DSC"] = 0.60
        terminal["results"][0]["Lesion_NSD"] = 0.50
        terminal["results"][0]["qualified"] = False
        terminal["decision"] = "NO_HIDDEN_ARCHIVE"
        terminal["selected"] = None
        spec.terminal.write_text(json.dumps(terminal), encoding="utf-8")
        self.assertIsNone(validate_family_selection(spec))

    def test_missing_or_duplicate_result_coverage_fails_closed(self):
        spec = self.make_spec("base")
        terminal = json.loads(spec.terminal.read_text(encoding="utf-8"))
        terminal["results"].pop()
        spec.terminal.write_text(json.dumps(terminal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_family_selection(spec)

        spec = self.make_spec("duplicate")
        terminal = json.loads(spec.terminal.read_text(encoding="utf-8"))
        terminal["results"][-1] = terminal["results"][0]
        spec.terminal.write_text(json.dumps(terminal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid .* result row"):
            validate_family_selection(spec)

    def test_forged_suboptimal_selected_row_fails_closed(self):
        spec = self.make_spec("base", dsc=0.65, nsd=0.56)
        terminal = json.loads(spec.terminal.read_text(encoding="utf-8"))
        better = terminal["results"][1]
        better["Lesion_DSC"] = 0.67
        better["Lesion_NSD"] = 0.57
        better["qualified"] = True
        spec.terminal.write_text(json.dumps(terminal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "terminal winner mismatch"):
            validate_family_selection(spec)

    def test_forged_qualification_flag_fails_closed(self):
        spec = self.make_spec("base")
        terminal = json.loads(spec.terminal.read_text(encoding="utf-8"))
        terminal["results"][1]["qualified"] = True
        spec.terminal.write_text(json.dumps(terminal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "qualification flag mismatch"):
            validate_family_selection(spec)

    def test_global_ranking_preserves_better_upstream_family(self):
        base = self.make_spec("base", dsc=0.68, nsd=0.57)
        dual = self.make_spec("dualwindow", dsc=0.65, nsd=0.58)
        selected = select_global_candidate((base, dual))
        self.assertEqual(selected.family, "base")
        self.assertEqual(selected.fixed_public_DSC, 0.68)

    def test_nsd_breaks_equal_dsc_tie(self):
        positive = self.make_spec("positive", dsc=0.67, nsd=0.55)
        organ = self.make_spec("organ", dsc=0.67, nsd=0.58)
        self.assertEqual(select_global_candidate((positive, organ)).family, "organ")

    def test_hash_or_hidden_side_effect_mismatch_fails_closed(self):
        spec = self.make_spec("base")
        terminal = json.loads(spec.terminal.read_text(encoding="utf-8"))
        terminal["selected"]["checkpoint"]["sha256"] = "0" * 64
        terminal["results"][0]["checkpoint"]["sha256"] = "0" * 64
        spec.terminal.write_text(json.dumps(terminal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            validate_family_selection(spec)
        terminal["selected"]["checkpoint"]["sha256"] = sha256_file(
            spec.model_dir / "fold_all/checkpoint_epoch_0100.pth"
        )
        terminal["hidden_read"] = True
        spec.terminal.write_text(json.dumps(terminal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "public selection terminal"):
            validate_family_selection(spec)

    def test_production_specs_cover_all_four_trained_families(self):
        specs = family_specs(ROOT)
        self.assertEqual(
            [spec.name for spec in specs],
            ["base", "positive", "organ", "dualwindow"],
        )
        self.assertEqual(len({spec.trainer for spec in specs}), 4)
        self.assertTrue(all(spec.policy.is_file() for spec in specs))
        self.assertTrue(all(spec.predictor.is_file() for spec in specs))

    def test_all_family_policies_share_the_exact_global_gate(self):
        for spec in family_specs(ROOT):
            policy = spec.policy_loader(spec.policy)
            self.assertEqual(policy["incumbent_fixed_public"], INCUMBENT)
            self.assertEqual(policy["selection"]["qualification"], QUALIFICATION)

    def test_noncomparable_family_gate_fails_closed(self):
        spec = self.make_spec("base")

        def drifted_loader(path: Path) -> dict:
            policy = load_policy(path)
            policy["selection"]["qualification"][
                "minimum_fixed_public_DSC"
            ] = 0.62
            return policy

        with self.assertRaisesRegex(ValueError, "noncomparable gate"):
            validate_family_selection(replace(spec, policy_loader=drifted_loader))

    def test_nonfinite_public_metric_fails_closed(self):
        spec = self.make_spec("base")
        terminal = json.loads(spec.terminal.read_text(encoding="utf-8"))
        terminal["selected"]["Lesion_DSC"] = float("nan")
        terminal["results"][0]["Lesion_DSC"] = float("nan")
        spec.terminal.write_text(json.dumps(terminal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid .* result row"):
            validate_family_selection(spec)


if __name__ == "__main__":
    unittest.main()
