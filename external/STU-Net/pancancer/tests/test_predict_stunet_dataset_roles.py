import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "predict_stunet_safe.py"
SPEC = importlib.util.spec_from_file_location("predict_stunet_dataset_roles", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PredictSTUNetDatasetRoleTests(unittest.TestCase):
    def test_each_validation_role_requires_its_exact_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "validation" / "Validation-Public-Images"
            hidden = root / "validation" / "Validation-Hidden-Images"
            training = root / "train" / "verified_complete"
            for path in (public, hidden, training):
                path.mkdir(parents=True)
            MODULE.validate_dataset_role_path(public, "public")
            MODULE.validate_dataset_role_path(hidden, "hidden")
            with self.assertRaises(ValueError):
                MODULE.validate_dataset_role_path(hidden, "public")
            with self.assertRaises(ValueError):
                MODULE.validate_dataset_role_path(training, "hidden")

    def test_training_complete_rejects_all_validation_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "train" / "verified_complete"
            validation = root / "validation" / "custom"
            training.mkdir(parents=True)
            validation.mkdir(parents=True)
            MODULE.validate_dataset_role_path(training, "training_complete")
            with self.assertRaises(ValueError):
                MODULE.validate_dataset_role_path(validation, "training_complete")


if __name__ == "__main__":
    unittest.main()
