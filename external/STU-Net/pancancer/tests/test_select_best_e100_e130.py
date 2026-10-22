import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from select_best_e100_e130_20260826 import select_best


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SelectBestE100E130Test(unittest.TestCase):
    def test_epoch100_fallback_is_selected_when_epoch130_misses_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            screen = root / "screen.json"
            screen.write_text(json.dumps({
                "status":"COMPLETE","case_count":50,"anchor_DSC":0.6004316379500749,
                "configurations":[{"configuration":"either_p70_any","Lesion_DSC":0.62,"passes_dsc_gate":True}],
                "hidden_read":False,"submission_created":False,"uploaded":False,
            }))
            metrics = root / "metrics.csv"
            with metrics.open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=["Name","Lesion_DSC","Lesion_NSD"]);writer.writeheader()
                for index in range(50):writer.writerow({"Name":f"case{index}","Lesion_DSC":0.62,"Lesion_NSD":0.53})
            checkpoint=root/"e100.pth";checkpoint.write_bytes(b"checkpoint")
            summary=root/"e130.tsv";summary.write_text("branch\tcheckpoint_epoch\tconfiguration\tLesion_DSC\tLesion_NSD\tDSC_gain\tNSD_gain\nbase_global_box\t130\tstunet_standalone\t0.4\t0.3\t-0.2\t-0.2\n")
            terminal=root/"e130.json";terminal.write_text(json.dumps({
                "status":"COMPLETE","decision":"KEEP_TOUCHCOMP","predeclared_epochs":[130],
                "summary":{"path":str(summary.resolve()),"sha256":sha(summary)},
                "hidden_read":False,"submission_created":False,"uploaded":False,
            }))
            result=select_best(e100_screen=screen,e100_metrics=metrics,e100_checkpoint=checkpoint,e130_terminal=terminal,output_dir=root/"out")
            saved=json.loads((root/"out"/"TERMINAL.json").read_text())
        self.assertEqual(result["selected_epoch"],100)
        self.assertEqual(saved["selected_configuration"],"gf_anchor_confirm_either_p70")
        self.assertEqual(saved["predeclared_epochs"],[100,130])


if __name__ == "__main__":
    unittest.main()
