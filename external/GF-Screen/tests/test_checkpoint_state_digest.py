import tempfile
import unittest
from pathlib import Path

import torch

from checkpoint_state_digest import state_dict_digest


class CheckpointStateDigestTests(unittest.TestCase):
    def test_metadata_does_not_change_digest_but_weights_do(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.pt"
            metadata_changed = root / "metadata_changed.pt"
            weights_changed = root / "weights_changed.pt"
            state = {
                "weight": torch.arange(12).reshape(3, 4).float(),
                "num_batches_tracked": torch.tensor(7),
            }
            torch.save({"state_dict": state, "epoch": 1}, first)
            torch.save({"state_dict": state, "epoch": 99, "optimizer": "different"}, metadata_changed)
            changed_state = dict(state)
            changed_state["weight"] = state["weight"] + 1
            torch.save({"state_dict": changed_state, "epoch": 1}, weights_changed)

            self.assertEqual(state_dict_digest(first), state_dict_digest(metadata_changed))
            self.assertNotEqual(state_dict_digest(first), state_dict_digest(weights_changed))


if __name__ == "__main__":
    unittest.main()
