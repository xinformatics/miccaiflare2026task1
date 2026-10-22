import csv
import tempfile
import unittest
from pathlib import Path

from audit_public_zero_case_bank import audit


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class AuditPublicZeroCaseBankTest(unittest.TestCase):
    def test_only_complete_fixed_public_candidates_are_ranked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchor = root / "anchor.csv"
            write_csv(
                anchor,
                ["case", "dice"],
                [
                    {"case": "case_a.nii.gz", "dice": 0.0},
                    {"case": "case_b.nii.gz", "dice": 0.5},
                ],
            )
            fields = ["Name", "Lesion_DSC", "Lesion_NSD"]
            write_csv(
                root / "candidate_low.csv",
                fields,
                [
                    {"Name": "case_a", "Lesion_DSC": 0.2, "Lesion_NSD": 0.1},
                    {"Name": "case_b", "Lesion_DSC": 0.8, "Lesion_NSD": 0.7},
                ],
            )
            write_csv(
                root / "candidate_best.csv",
                fields,
                [
                    {"Name": "case_a.nii.gz", "Lesion_DSC": 0.4, "Lesion_NSD": 0.3},
                    {"Name": "case_b.nii.gz", "Lesion_DSC": 0.6, "Lesion_NSD": 0.5},
                ],
            )
            write_csv(
                root / "incomplete.csv",
                fields,
                [{"Name": "case_a", "Lesion_DSC": 0.9, "Lesion_NSD": 0.9}],
            )
            write_csv(root / "unrelated.csv", ["case", "score"], [{"case": "x", "score": 1}])

            summary, rows = audit(root, anchor, top_k=2)

            self.assertEqual(summary["csv_files_scanned"], 5)
            self.assertEqual(summary["metric_csv_files"], 3)
            self.assertEqual(summary["complete_fixed_public_candidates"], 2)
            self.assertEqual(summary["zero_dice_cases"], ["case_a"])
            best = summary["best_by_case"]["case_a"]
            self.assertEqual(best["case_dsc"], 0.4)
            self.assertTrue(str(best["metrics_path"]).endswith("candidate_best.csv"))
            self.assertEqual([row["case_dsc"] for row in rows], [0.4, 0.2])


if __name__ == "__main__":
    unittest.main()
