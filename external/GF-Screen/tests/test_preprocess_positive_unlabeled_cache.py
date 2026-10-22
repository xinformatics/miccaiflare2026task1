import hashlib
import tempfile
import unittest
from pathlib import Path

from preprocess_positive_unlabeled_cache import cache_inventory, dataset_args


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    ROOT / "run_preprocess_positive_unlabeled_cache_after_pseudo_20260825.sh"
).read_text(encoding="utf-8")


class FakeDataset:
    def __init__(self, root: Path):
        self.cache_dir = root
        self.transform_hash = ""
        self.data = [{"key": "a"}, {"key": "b"}]

    @staticmethod
    def hash_func(item):
        return item["key"].encode("utf-8")


class PositiveUnlabeledCacheTests(unittest.TestCase):
    def test_base_datalist_is_independent_of_calling_directory(self):
        args = dataset_args(Path("/tmp/cache"), Path("/tmp/pseudo.json"))
        self.assertTrue(Path(args.abdomen_json).is_absolute())
        self.assertTrue(Path(args.abdomen_json).is_file())
        self.assertEqual(Path(args.data_dir), ROOT)

    def test_completed_cache_is_verified_before_contract_can_be_rewritten(self):
        completed_branch = WRAPPER.index('if [[ -s "${complete}" ]]')
        contract_writer = WRAPPER.index("validate_positive_unlabeled_manifest.py")
        self.assertLess(completed_branch, contract_writer)
        self.assertIn("--verify-only", WRAPPER[completed_branch:contract_writer])

    def test_wrapper_forces_same_filesystem_atomic_cache_publication(self):
        self.assertIn('export TMPDIR="${cache_root}/tmp/pseudo_positive_cache"', WRAPPER)
        source = (
            ROOT / "preprocess_positive_unlabeled_cache.py"
        ).read_text(encoding="utf-8")
        self.assertIn("temporary_dir.stat().st_dev != abdomen_dir.stat().st_dev", source)
        self.assertIn("cross-filesystem shutil.move is not atomic", source)

    def test_inventory_is_exact_and_content_size_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.pt").write_bytes(b"abc")
            (root / "b.pt").write_bytes(b"defgh")
            result = cache_inventory(FakeDataset(root))
            rendered = "a.pt\t3\nb.pt\t5\n"
            self.assertEqual(result["members"], 2)
            self.assertEqual(result["bytes"], 8)
            self.assertEqual(
                result["name_size_sha256"],
                hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            )

    def test_inventory_rejects_missing_or_empty_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.pt").write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "missing pseudo cache entries: 1"):
                cache_inventory(FakeDataset(root))
            (root / "b.pt").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "missing pseudo cache entries: 1"):
                cache_inventory(FakeDataset(root))


if __name__ == "__main__":
    unittest.main()
