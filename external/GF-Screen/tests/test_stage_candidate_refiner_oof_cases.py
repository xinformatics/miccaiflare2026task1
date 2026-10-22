import tempfile
import unittest
from pathlib import Path

from stage_candidate_refiner_oof_cases import (
    build_partitions,
    parse_proposal_sources,
    stage_partitions,
)


class CandidateRefinerOOFStagingTests(unittest.TestCase):
    def test_exact_case_coverage_and_zero_patch_assignment(self):
        items = [
            {"name": f"case{i}", "cohort": "cohort", "complete_label": True}
            for i in range(5)
        ]
        rows = [
            {"case": f"case{i}", "cohort": "cohort", "patch_id": f"case{i}__c0001"}
            for i in range(4)
        ]
        partitions, zero = build_partitions(rows, items, folds=2, seed=7)
        observed = [item["name"] for values in partitions.values() for item in values]
        self.assertEqual(sorted(observed), [f"case{i}" for i in range(5)])
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(zero, ["case4"])

    def test_rejects_case_outside_manifest(self):
        items = [{"name": "case0", "cohort": "cohort", "complete_label": True}]
        rows = [{"case": "other", "cohort": "cohort", "patch_id": "other__c0001"}]
        with self.assertRaises(ValueError):
            build_partitions(rows, items, folds=2, seed=7)

    def test_named_sources_are_staged_separately_with_exact_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {name: root / name for name in ("epoch105", "weak")}
            for directory in sources.values():
                directory.mkdir()
            items = []
            rows = []
            for index in range(4):
                case = f"case{index}"
                image = root / f"{case}_0000.nii.gz"
                label = root / f"{case}_label.nii.gz"
                image.write_bytes(b"image")
                label.write_bytes(b"label")
                for source in sources.values():
                    (source / f"{case}.nii.gz").write_bytes(source.name.encode())
                items.append(
                    {
                        "name": case,
                        "cohort": "cohort",
                        "complete_label": True,
                        "prediction_name": f"{case}.nii.gz",
                        "image": str(image),
                        "label": str(label),
                    }
                )
                rows.append(
                    {"case": case, "cohort": "cohort", "patch_id": f"{case}__c1"}
                )
            output = root / "staging"
            report = stage_partitions(
                rows, items, sources, output, folds=2, seed=7, legacy_layout=False
            )
            self.assertEqual(report["proposal_layout"], "named_sources")
            self.assertTrue(report["exact_proposal_coverage"])
            for name, source in sources.items():
                staged = output / "all" / "proposals" / name
                self.assertEqual(len(list(staged.glob("*.nii.gz"))), 4)
                for path in staged.glob("*.nii.gz"):
                    self.assertEqual(path.resolve().parent, source.resolve())

            legacy_output = root / "legacy_staging"
            legacy_report = stage_partitions(
                rows,
                items,
                {"proposal": sources["epoch105"]},
                legacy_output,
                folds=2,
                seed=7,
                legacy_layout=True,
            )
            self.assertEqual(legacy_report["proposal_layout"], "legacy_single")
            self.assertEqual(
                len(list((legacy_output / "all" / "proposals").glob("*.nii.gz"))),
                4,
            )
            self.assertFalse(
                (legacy_output / "all" / "proposals" / "proposal").exists()
            )

    def test_proposal_parser_rejects_mixed_and_invalid_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            parsed, legacy = parse_proposal_sources(source, [])
            self.assertTrue(legacy)
            self.assertEqual(parsed, {"proposal": source.resolve()})
            with self.assertRaisesRegex(ValueError, "either"):
                parse_proposal_sources(source, [f"named={source}"])
            with self.assertRaisesRegex(ValueError, "NAME=DIRECTORY"):
                parse_proposal_sources(None, [str(source)])
            with self.assertRaisesRegex(ValueError, "invalid"):
                parse_proposal_sources(None, [f"../bad={source}"])


if __name__ == "__main__":
    unittest.main()
