import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "validate_post_e100_branch.py"
SPEC = importlib.util.spec_from_file_location("validate_post_e100_branch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
validate_branch = MODULE.validate_branch


def checkpoint(path: Path, epoch: int, value: float = 1.0) -> None:
    torch.save(
        {
            "current_epoch": epoch,
            "network_weights": {"weight": torch.tensor([value])},
            "optimizer_state": {"state": {0: {"momentum": torch.tensor([value])}}},
        },
        path,
    )


class PostE100BranchValidationTest(unittest.TestCase):
    def test_base_accepts_later_resumable_optimizer_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, latest = root / "e100.pth", root / "latest.pth"
            checkpoint(source, 100)
            checkpoint(latest, 130, 2.0)
            result = validate_branch("base", source, latest)
            self.assertEqual(result["latest_epoch"], 130)

    def test_component_manifest_binds_exact_seed_then_accepts_later_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, latest = root / "e100.pth", root / "latest.pth"
            trainer, manifest = root / "trainer.py", root / "SOURCE.json"
            checkpoint(source, 100)
            latest.write_bytes(source.read_bytes())
            trainer.write_text("component trainer\n", encoding="utf-8")
            first = validate_branch("component", source, latest, trainer, manifest)
            self.assertTrue(first["manifest_created"])
            checkpoint(latest, 130, 2.0)
            resumed = validate_branch("component", source, latest, trainer, manifest)
            self.assertFalse(resumed["manifest_created"])
            self.assertEqual(resumed["latest_epoch"], 130)

    def test_component_rejects_nonidentical_initial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, latest = root / "e100.pth", root / "latest.pth"
            trainer, manifest = root / "trainer.py", root / "SOURCE.json"
            checkpoint(source, 100)
            checkpoint(latest, 100, 2.0)
            trainer.write_text("component trainer\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact epoch-100 bytes"):
                validate_branch("component", source, latest, trainer, manifest)

    def test_rejects_terminal_epoch_as_resumable_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, latest = root / "e100.pth", root / "latest.pth"
            checkpoint(source, 100)
            checkpoint(latest, 300)
            with self.assertRaisesRegex(ValueError, "epoch 300"):
                validate_branch("base", source, latest)


if __name__ == "__main__":
    unittest.main()
