import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_direct_focus_public_terminal import validate


DSC_GATE = 0.6154316379500749
NSD_FLOOR = 0.5104225107769019


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DirectFocusPublicTerminalValidationTests(unittest.TestCase):
    def fixture(self, root: Path, dsc: float = 0.6200, nsd: float = 0.5200):
        route = root / "ROUTE.json"
        calibration = root / "PREDECLARED_CALIBRATION.json"
        summary = root / "summary_by_dsc.tsv"
        terminal = root / "TERMINAL.json"
        route.write_text('{"decision":"qualified"}\n', encoding="utf-8")
        calibration.write_text('{"deployment":"frozen"}\n', encoding="utf-8")
        with summary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("variant", "DSC", "NSD"), delimiter="\t")
            writer.writeheader()
            writer.writerow({"variant": "fen25", "DSC": dsc, "NSD": nsd})
        qualified = dsc >= DSC_GATE and nsd >= NSD_FLOOR
        payload = {
            "status": "COMPLETE",
            "decision": "QUALIFY_SUBMISSION_PREPARATION" if qualified else "KEEP_TOUCHCOMP",
            "fixed_public": {
                "DSC": dsc,
                "NSD": nsd,
                "DSC_gate": DSC_GATE,
                "NSD_floor": NSD_FLOOR,
            },
            "route": {"path": str(route.resolve()), "sha256": sha256(route)},
            "calibration": {
                "path": str(calibration.resolve()),
                "sha256": sha256(calibration),
            },
            "summary": {"path": str(summary.resolve()), "sha256": sha256(summary)},
            "public_read": True,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
        }
        terminal.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return terminal, route, calibration, summary

    def test_accepts_qualified_and_keep_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            self.assertEqual(validate(*paths), "QUALIFY_SUBMISSION_PREPARATION")
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory), dsc=0.6100, nsd=0.5200)
            self.assertEqual(validate(*paths), "KEEP_TOUCHCOMP")

    def test_rejects_changed_summary_content(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            paths[3].write_text("variant\tDSC\tNSD\nfen25\t0.9\t0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary identity mismatch"):
                validate(*paths)

    def test_accepts_exact_terminal_means_with_rounded_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root, dsc=0.4101, nsd=0.2614)
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            payload["fixed_public"]["DSC"] = 0.41008214595460957
            payload["fixed_public"]["NSD"] = 0.2614227170521056
            paths[0].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            self.assertEqual(validate(*paths), "KEEP_TOUCHCOMP")

    def test_rejects_gate_decision_and_provenance_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            payload["decision"] = "KEEP_TOUCHCOMP"
            paths[0].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decision/gate mismatch"):
                validate(*paths)
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            payload["hidden_read"] = True
            paths[0].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "public-only provenance mismatch"):
                validate(*paths)


if __name__ == "__main__":
    unittest.main()
