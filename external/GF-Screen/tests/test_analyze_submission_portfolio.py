import tempfile
import unittest
from pathlib import Path

from analyze_submission_portfolio import analyze, load_metrics


def rows(dsc, nsd=None):
    nsd = dsc if nsd is None else nsd
    return {
        f"case{index}": {"Lesion_DSC": value, "Lesion_NSD": nsd[index]}
        for index, value in enumerate(dsc)
    }


class SubmissionPortfolioTests(unittest.TestCase):
    def test_casewise_oracle_prefers_complementary_candidates(self):
        anchor = rows([0.5, 0.5, 0.5, 0.5])
        candidates = {
            "a": rows([0.9, 0.9, 0.1, 0.1]),
            "b": rows([0.1, 0.1, 0.9, 0.9]),
            "c": rows([0.8, 0.8, 0.1, 0.1]),
        }
        report = analyze("anchor", anchor, candidates, slots=2)
        self.assertEqual(report["portfolios"][0]["candidates"], ["a", "b"])
        self.assertAlmostEqual(
            report["portfolios"][0]["casewise_max_Lesion_DSC"], 0.9
        )

    def test_rejects_case_mismatch(self):
        with self.assertRaisesRegex(ValueError, "case mismatch"):
            analyze("anchor", rows([0.5, 0.5]), {"a": rows([0.5])}, slots=1)

    def test_csv_loader_rejects_duplicate_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(
                "Name,Lesion_DSC,Lesion_NSD\ncase,0.5,0.4\ncase,0.6,0.5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_metrics(path)


if __name__ == "__main__":
    unittest.main()
