import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from GF_RL_trainer import atomic_copyfile, atomic_torch_save, save_checkpoint


class AtomicCheckpointSaveTest(unittest.TestCase):
    def test_checkpoint_is_loadable_and_no_temporary_is_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(3, 2)
            args = SimpleNamespace(distributed=False, logdir=str(root))
            save_checkpoint(model, 7, args, filename="checkpoint_epoch_0008.pt")
            checkpoint = root / "checkpoint_epoch_0008.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["epoch"], 7)
            self.assertEqual(set(payload["state_dict"]), set(model.state_dict()))
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_serialization_failure_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "checkpoint.pt"
            destination.write_bytes(b"previous-complete-checkpoint")

            def fail_after_partial(payload, path):
                del payload
                Path(path).write_bytes(b"partial")
                raise OSError("simulated interrupted save")

            with mock.patch("GF_RL_trainer.torch.save", side_effect=fail_after_partial):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    atomic_torch_save({"epoch": 8}, destination)
            self.assertEqual(destination.read_bytes(), b"previous-complete-checkpoint")

    def test_best_model_copy_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model_final.pt"
            destination = root / "model.pt"
            source.write_bytes(os.urandom(4096))
            atomic_copyfile(source, destination)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertEqual(list(root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
