import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from evaluate_internal_checkpoint import evaluation_args


class EvaluateInternalCheckpointTest(unittest.TestCase):
    def test_reconstructs_locked_long_focus_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = Namespace(
                checkpoint=root / "checkpoint.pt",
                cache_dir=root / "cache",
                abdomen_json=root / "abdomen.json",
                chest_json=root / "chest.json",
                healthy_json=root / "healthy.json",
                workers=2,
                gpu=1,
                harmonize_cached_chest_window=False,
            )
            args = evaluation_args(cli)
            self.assertEqual(args.seed, 20260825)
            self.assertEqual(args.batch_size, 4)
            self.assertEqual(args.sw_batch_size, 2)
            self.assertEqual(args.workers, 2)
            self.assertEqual(args.gpu, 1)
            self.assertEqual(args.rank, 1)
            self.assertFalse(args.distributed)
            self.assertTrue(args.amp)
            self.assertFalse(args.harmonize_cached_chest_window)


if __name__ == "__main__":
    unittest.main()
