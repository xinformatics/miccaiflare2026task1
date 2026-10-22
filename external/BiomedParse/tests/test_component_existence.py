import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_component_existence import auc, named_directories  # noqa: E402


class ComponentExistenceTests(unittest.TestCase):
    def test_auc_handles_perfect_inverse_and_tied_scores(self):
        labels = np.asarray([False, False, True, True])
        self.assertEqual(auc(labels, np.asarray([0.0, 0.1, 0.8, 0.9])), 1.0)
        self.assertEqual(auc(labels, np.asarray([0.8, 0.9, 0.0, 0.1])), 0.0)
        self.assertEqual(auc(labels, np.ones(4)), 0.5)

    def test_auc_requires_both_classes(self):
        self.assertIsNone(auc(np.asarray([True, True]), np.asarray([0.1, 0.2])))

    def test_named_directories_rejects_duplicates(self):
        self.assertEqual(named_directories(["high=/tmp/high"]), {"high": Path("/tmp/high")})
        with self.assertRaises(ValueError):
            named_directories(["high=/tmp/a", "high=/tmp/b"])


if __name__ == "__main__":
    unittest.main()
