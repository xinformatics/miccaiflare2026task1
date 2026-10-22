import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from evaluate_internal_checkpoint_grid import (
    atomic_json,
    build_curve_analysis,
    build_fixed_public_shortlist,
    compare_source_parity,
    import_inline_results,
    load_complete_result,
    parse_inline_validations,
    result_path,
    sha256_file,
    validate_cli,
    write_curve_svg,
    write_summary,
)


class EvaluateInternalCheckpointGridTest(unittest.TestCase):
    def build_fixture(self, root, epochs=(284, 285)):
        run = root / "run"
        output = root / "output"
        run.mkdir()
        output.mkdir()
        inventory = []
        for epoch in epochs:
            checkpoint = run / f"checkpoint_epoch_{epoch:04d}.pt"
            checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
            inventory.append((epoch, checkpoint))
        args = Namespace(
            run_dir=run,
            output=output,
            start_epoch=min(epochs),
            end_epoch=max(epochs),
            gpus=[0, 1],
            workers=2,
            idle_memory_mib=4096,
            parity_epoch=min(epochs),
            parity_tolerance=1e-6,
        )
        return args, inventory

    @staticmethod
    def write_result(output, epoch, checkpoint, dice, annotated_recall=None):
        payload = {
            "status": "COMPLETE",
            "evaluation_source": "inline_distributed_training_validation",
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": sha256_file(checkpoint),
            },
            "internal_validation": {
                "dataset_cases": 90,
                "dice": dice,
                "elapsed_seconds": 12.5,
            },
            "public_read": False,
            "hidden_read": False,
        }
        if annotated_recall is not None:
            payload["internal_validation"]["annotated_recall"] = annotated_recall
        result_path(output, epoch).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    @staticmethod
    def write_standalone_result(output, epoch, checkpoint, dice):
        from evaluate_internal_checkpoint_grid import EVALUATOR

        payload = {
            "status": "COMPLETE",
            "evaluation_source": "standalone_single_gpu_validation",
            "evaluator": {
                "path": str(EVALUATOR.resolve()),
                "sha256": sha256_file(EVALUATOR),
            },
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": sha256_file(checkpoint),
            },
            "internal_validation": {
                "dataset_cases": 90,
                "dice": dice,
                "elapsed_seconds": 12.5,
            },
            "public_read": False,
            "hidden_read": False,
        }
        result_path(output, epoch).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def test_validates_complete_contiguous_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            args, expected = self.build_fixture(Path(directory))
            self.assertEqual(validate_cli(args), expected)

    def test_rejects_missing_checkpoint_in_range(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.build_fixture(Path(directory), epochs=(284, 286))
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint_epoch_0285"):
                validate_cli(args)

    def test_rejects_duplicate_gpu_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.build_fixture(Path(directory))
            args.gpus = [0, 0]
            with self.assertRaisesRegex(ValueError, "unique"):
                validate_cli(args)

    def test_accepts_exact_complete_result(self):
        with tempfile.TemporaryDirectory() as directory:
            args, inventory = self.build_fixture(Path(directory))
            epoch, checkpoint = inventory[0]
            expected = self.write_result(args.output, epoch, checkpoint, 0.31)
            self.assertEqual(
                load_complete_result(result_path(args.output, epoch), checkpoint),
                expected,
            )

    def test_rejects_checkpoint_mutation_after_result(self):
        with tempfile.TemporaryDirectory() as directory:
            args, inventory = self.build_fixture(Path(directory))
            epoch, checkpoint = inventory[0]
            self.write_result(args.output, epoch, checkpoint, 0.31)
            checkpoint.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_complete_result(result_path(args.output, epoch), checkpoint)

    def test_standalone_result_is_bound_to_evaluator_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            args, inventory = self.build_fixture(Path(directory))
            epoch, checkpoint = inventory[0]
            payload = self.write_standalone_result(
                args.output, epoch, checkpoint, 0.31
            )
            self.assertEqual(
                load_complete_result(result_path(args.output, epoch), checkpoint),
                payload,
            )
            payload["evaluator"]["sha256"] = "0" * 64
            result_path(args.output, epoch).write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "evaluator identity mismatch"):
                load_complete_result(result_path(args.output, epoch), checkpoint)

    def test_rejects_result_that_read_challenge_data(self):
        with tempfile.TemporaryDirectory() as directory:
            args, inventory = self.build_fixture(Path(directory))
            epoch, checkpoint = inventory[0]
            payload = self.write_result(args.output, epoch, checkpoint, 0.31)
            payload["public_read"] = True
            result_path(args.output, epoch).write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "challenge data"):
                load_complete_result(result_path(args.output, epoch), checkpoint)

    def test_summary_selectable_fields_cover_every_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            args, inventory = self.build_fixture(Path(directory))
            self.write_result(args.output, 284, inventory[0][1], 0.31)
            self.write_result(args.output, 285, inventory[1][1], 0.34)
            rows = write_summary(args.output, inventory)
            self.assertEqual([row["epoch"] for row in rows], [284, 285])
            self.assertEqual([row["cases"] for row in rows], [90, 90])
            self.assertEqual(max(rows, key=lambda row: row["dice"])["epoch"], 285)
            self.assertIsNone(rows[0]["annotated_component_recall"])
            self.assertEqual(
                rows[0]["recall_evaluation_source"],
                "unavailable_pre_instrumentation",
            )
            self.assertTrue((args.output / "summary.tsv").is_file())

    def test_summary_uses_standalone_parity_recall_without_replacing_dice(self):
        with tempfile.TemporaryDirectory() as directory:
            args, inventory = self.build_fixture(Path(directory))
            self.write_result(args.output, 284, inventory[0][1], 0.31)
            self.write_result(args.output, 285, inventory[1][1], 0.34)
            recall = {
                "annotated_voxel_recall": 0.72,
                "annotated_component_recall": 0.43,
                "annotated_component_recall_lt100": 0.10,
                "annotated_component_recall_100to500": 0.20,
                "annotated_component_recall_500to1000": 0.30,
                "annotated_component_recall_1000to3000": 0.60,
                "annotated_component_recall_ge3000": 0.90,
            }
            rows = write_summary(
                args.output, inventory, recall_overrides={284: recall}
            )
            self.assertEqual(rows[0]["dice"], 0.31)
            self.assertEqual(rows[0]["annotated_component_recall"], 0.43)
            self.assertEqual(
                rows[0]["recall_evaluation_source"],
                "standalone_parity_override",
            )
            self.assertIsNone(rows[1]["annotated_component_recall"])

    def test_atomic_json_replaces_existing_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            terminal = Path(directory) / "TERMINAL.json"
            atomic_json(terminal, {"status": "OLD"})
            atomic_json(terminal, {"status": "COMPLETE"})
            self.assertEqual(
                json.loads(terminal.read_text(encoding="utf-8")),
                {"status": "COMPLETE"},
            )
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_learning_curve_analysis_and_svg_cover_all_points(self):
        rows = [
            {
                "epoch": 284,
                "dice": 0.31,
                "evaluation_source": "standalone_single_gpu_validation",
                "annotated_component_recall": 0.41,
                "annotated_voxel_recall": 0.61,
                "annotated_component_recall_ge3000": 0.71,
            },
            {
                "epoch": 285,
                "dice": 0.34,
                "evaluation_source": "inline_distributed_training_validation",
                "annotated_component_recall": None,
                "annotated_voxel_recall": None,
                "annotated_component_recall_ge3000": None,
            },
            {
                "epoch": 286,
                "dice": 0.32,
                "evaluation_source": "standalone_single_gpu_validation",
                "annotated_component_recall": 0.47,
                "annotated_voxel_recall": 0.66,
                "annotated_component_recall_ge3000": 0.76,
            },
        ]
        analysis = build_curve_analysis(rows)
        self.assertEqual(analysis["checkpoint_count"], 3)
        self.assertEqual(analysis["best"]["epoch"], 285)
        self.assertAlmostEqual(
            analysis["largest_adjacent_gain"]["dice_change"], 0.03
        )
        self.assertAlmostEqual(
            analysis["largest_adjacent_drop"]["dice_change"], -0.02
        )
        self.assertEqual(analysis["recall_available_checkpoint_count"], 2)
        self.assertEqual(
            analysis["best_annotated_component_recall"]["epoch"], 286
        )
        self.assertEqual(
            analysis["best_annotated_large_component_recall_ge3000"]["epoch"],
            286,
        )
        self.assertEqual(analysis["best_annotated_voxel_recall"]["epoch"], 286)
        with tempfile.TemporaryDirectory() as directory:
            path = write_curve_svg(Path(directory), analysis)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Locked internal Dice by checkpoint", content)
            self.assertIn("best e285: 0.3400", content)
            self.assertEqual(content.count("<circle"), 5)

    def test_recall_shortlist_is_dice_guarded_complementary_and_deduplicated(self):
        def point(epoch, dice, component=None, voxel=None, large=None):
            values = {
                "epoch": epoch,
                "dice": dice,
                "evaluation_source": "standalone_single_gpu_validation",
                "checkpoint_sha256": str(epoch) * 64,
                "audit": f"/audit/{epoch}.json",
                "annotated_component_recall": component,
                "annotated_voxel_recall": voxel,
                "annotated_component_recall_ge3000": large,
            }
            for field in (
                "annotated_component_recall_lt100",
                "annotated_component_recall_100to500",
                "annotated_component_recall_500to1000",
            ):
                # Tiny-lesion recall is deliberately irrelevant to nomination.
                values[field] = 0.99
            return values

        analysis = {
            "points": [
                point(284, 0.400),
                point(285, 0.395, component=0.70, voxel=0.60, large=0.40),
                point(286, 0.394, component=0.60, voxel=0.65, large=0.75),
                point(287, 0.392, component=0.55, voxel=0.72, large=0.65),
                # Better recall is rejected because Dice is outside the guard.
                point(288, 0.380, component=0.99, voxel=0.99, large=0.99),
            ]
        }
        result = build_fixed_public_shortlist(analysis)
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual([row["epoch"] for row in result["candidates"]], [284, 286, 287])
        by_epoch = {row["epoch"]: row for row in result["candidates"]}
        self.assertEqual(
            by_epoch[286]["selection_roles"],
            ["best_large_component_recall_ge3000_within_dice_guard"],
        )
        self.assertEqual(
            by_epoch[287]["selection_roles"],
            ["best_voxel_recall_within_dice_guard"],
        )
        self.assertNotIn("best_component_recall_within_dice_guard", str(result))
        self.assertNotIn("best_small_component_recall_within_dice_guard", str(result))
        self.assertNotIn(288, by_epoch)
        self.assertEqual(
            result["selection_objective"],
            "case_balanced_dice_plus_ge3000mm3_and_voxel_recall",
        )
        self.assertEqual(result["large_component_minimum_mm3"], 3000)
        self.assertEqual(result["promotion_authority"], "fixed_public_50_only")
        self.assertFalse(result["public_read"])

    def test_recall_shortlist_falls_back_to_best_dice_when_recall_unavailable(self):
        analysis = {
            "points": [
                {"epoch": 284, "dice": 0.31},
                {"epoch": 285, "dice": 0.34},
            ]
        }
        result = build_fixed_public_shortlist(analysis)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["epoch"], 285)

    def test_recall_shortlist_rejects_invalid_policy(self):
        with self.assertRaisesRegex(ValueError, "Dice guard"):
            build_fixed_public_shortlist({"points": [{"epoch": 1, "dice": 0.3}]}, -0.1)
        with self.assertRaisesRegex(ValueError, "maximum"):
            build_fixed_public_shortlist(
                {"points": [{"epoch": 1, "dice": 0.3}]}, maximum_candidates=0
            )

    def test_inline_and_standalone_source_parity_is_strict(self):
        inline = {
            "evaluation_source": "inline_distributed_training_validation",
            "internal_validation": {"dice": 0.33906215},
        }
        standalone = {
            "internal_validation": {"dice": 0.33906214},
        }
        result = compare_source_parity(inline, standalone, 295, 1e-6)
        self.assertEqual(result["status"], "PASS")
        self.assertLess(result["absolute_difference"], 1e-6)
        standalone["internal_validation"]["dice"] = 0.3390
        with self.assertRaisesRegex(ValueError, "Dice mismatch"):
            compare_source_parity(inline, standalone, 295, 1e-6)

    def test_source_parity_rejects_wrong_provenance(self):
        wrong = {
            "evaluation_source": "standalone_single_gpu_validation",
            "internal_validation": {"dice": 0.3},
        }
        with self.assertRaisesRegex(ValueError, "not an imported inline"):
            compare_source_parity(wrong, wrong, 295, 1e-6)

    def test_parses_inline_validation_as_checkpoint_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(
                "Final validation  284/312 dice 0.3402788 time 375.64s\n"
                "Final validation  289/312 dice 0.31757718 time 358.24s\n",
                encoding="utf-8",
            )
            parsed = parse_inline_validations(log)
            self.assertEqual(sorted(parsed), [285, 290])
            self.assertEqual(parsed[285]["dice"], 0.3402788)
            self.assertEqual(parsed[290]["elapsed_seconds"], 358.24)

    def test_rejects_conflicting_duplicate_inline_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(
                "Final validation  284/312 dice 0.34 time 10s\n"
                "Final validation  284/312 dice 0.35 time 10s\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflicting"):
                parse_inline_validations(log)

    def test_imports_only_matching_inline_checkpoint_results(self):
        with tempfile.TemporaryDirectory() as directory:
            args, inventory = self.build_fixture(Path(directory))
            log = args.run_dir / "train.log"
            log.write_text(
                "Final validation  284/312 dice 0.34 time 10.5s\n",
                encoding="utf-8",
            )
            imported = import_inline_results(args.output, inventory, log)
            self.assertEqual(imported, [285])
            self.assertFalse(result_path(args.output, 284).exists())
            payload = json.loads(
                result_path(args.output, 285).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["internal_validation"]["dataset_cases"], 90)
            self.assertEqual(payload["internal_validation"]["distributed_steps"], 45)
            self.assertEqual(payload["evaluation_source"], "inline_distributed_training_validation")
            self.assertFalse(payload["public_read"])


if __name__ == "__main__":
    unittest.main()
