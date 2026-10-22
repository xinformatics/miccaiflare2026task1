import unittest

from select_focus_tta_evidence_floor import select_evidence_floor


def equivalence(**updates):
    payload = {
        "cases": 50,
        "max_case_dsc_delta": 0.0008,
        "max_case_nsd_delta": 0.0007,
        "max_case_voxel_delta": 86,
        "total_abs_voxel_delta": 193,
        "total_abs_voxel_fraction": 3.7e-5,
        "mean_dsc_delta": 2e-5,
        "mean_nsd_delta": 0.0,
    }
    payload.update(updates)
    return payload


def robust(anchor, dsc=0.009, nsd=0.011, ci=0.002, win=0.58, catastrophic=0.0):
    return {
        "anchor": anchor,
        "candidate": "tta_xy",
        "cases": 50,
        "paired_gain": {"Lesion_DSC": dsc, "Lesion_NSD": nsd, "combined": (dsc + nsd) / 2},
        "bootstrap_95ci": {
            "Lesion_DSC": [ci, 0.02],
            "Lesion_NSD": [ci, 0.02],
            "combined": [ci, 0.02],
        },
        "win_rate": win,
        "catastrophic_regression_rate": catastrophic,
    }


def runtime(maximum="36.2", eligible="yes"):
    return [
        {"variant": "notta", "max_seconds": "20", "runtime_eligible": "yes"},
        {"variant": "tta_xy", "max_seconds": maximum, "runtime_eligible": eligible},
    ]


class FocusTTAEvidenceFloorTests(unittest.TestCase):
    def test_selects_observed_safe_evidence_floor(self):
        result = select_evidence_floor(
            equivalence(), robust("notta"), robust("gf_champ_new2k_s25"), runtime()
        )
        self.assertEqual(result["decision"], "SAVE_EXPLORATORY_TTA_FALLBACK")
        self.assertEqual(result["selected_variant"], "tta_xy")
        self.assertFalse(result["automatic_upload_authorized"])

    def test_rejects_subthreshold_dsc_gain(self):
        result = select_evidence_floor(
            equivalence(), robust("notta", dsc=0.004), robust("gf_champ_new2k_s25"), runtime()
        )
        self.assertEqual(result["decision"], "KEEP_CHAMPION")

    def test_rejects_nonpositive_confidence_bound(self):
        result = select_evidence_floor(
            equivalence(), robust("notta", ci=0.0), robust("gf_champ_new2k_s25"), runtime()
        )
        self.assertEqual(result["decision"], "KEEP_CHAMPION")

    def test_rejects_excess_numerical_drift(self):
        result = select_evidence_floor(
            equivalence(total_abs_voxel_delta=257),
            robust("notta"),
            robust("gf_champ_new2k_s25"),
            runtime(),
        )
        self.assertEqual(result["decision"], "KEEP_CHAMPION")

    def test_rejects_runtime_over_challenge_limit(self):
        result = select_evidence_floor(
            equivalence(), robust("notta"), robust("gf_champ_new2k_s25"), runtime("55.1")
        )
        self.assertEqual(result["decision"], "KEEP_CHAMPION")

    def test_rejects_wrong_candidate_identity(self):
        report = robust("notta")
        report["candidate"] = "tta_x"
        with self.assertRaisesRegex(ValueError, "identity"):
            select_evidence_floor(
                equivalence(), report, robust("gf_champ_new2k_s25"), runtime()
            )


if __name__ == "__main__":
    unittest.main()
