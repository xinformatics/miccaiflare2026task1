import unittest

from select_focus_tta_candidate import select_candidate


def row(variant, dsc, nsd, runtime="yes", notta="PROMOTE", champion="PROMOTE", match="yes"):
    return {
        "variant": variant,
        "DSC": str(dsc),
        "NSD": str(nsd),
        "combined": str((dsc + nsd) / 2),
        "runtime_eligible": runtime,
        "vs_notta": "ANCHOR" if variant == "notta" else notta,
        "vs_champion": "ANCHOR" if variant == "notta" else champion,
        "champion_anchor_match": match,
    }


class FocusTTASelectionTests(unittest.TestCase):
    def test_selects_highest_dsc_among_fully_qualified_variants(self):
        result = select_candidate(
            [row("notta", 0.59, 0.50), row("tta_x", 0.61, 0.52), row("tta_xy", 0.62, 0.51)]
        )
        self.assertEqual(result["decision"], "PROMOTE_TTA_HIDDEN")
        self.assertEqual(result["selected"]["variant"], "tta_xy")

    def test_rejects_runtime_ineligible_variant(self):
        result = select_candidate(
            [row("notta", 0.59, 0.50), row("tta_x", 0.61, 0.52, runtime="no"), row("tta_xy", 0.60, 0.51, champion="KEEP_ANCHOR")]
        )
        self.assertEqual(result["decision"], "KEEP_CHAMPION")

    def test_requires_both_paired_promotions(self):
        result = select_candidate(
            [row("notta", 0.59, 0.50), row("tta_x", 0.61, 0.52, notta="KEEP_ANCHOR"), row("tta_xy", 0.62, 0.53, champion="KEEP_ANCHOR")]
        )
        self.assertEqual(result["decision"], "KEEP_CHAMPION")

    def test_requires_metric_nonregression(self):
        result = select_candidate(
            [row("notta", 0.59, 0.50), row("tta_x", 0.61, 0.49), row("tta_xy", 0.58, 0.53)]
        )
        self.assertEqual(result["decision"], "KEEP_CHAMPION")

    def test_rejects_incomplete_inventory(self):
        with self.assertRaisesRegex(ValueError, "each expected variant"):
            select_candidate([row("notta", 0.59, 0.50), row("tta_x", 0.61, 0.52)])

    def test_rejects_pipeline_that_does_not_reproduce_champion(self):
        with self.assertRaisesRegex(ValueError, "locked bounds"):
            select_candidate(
                [row("notta", 0.59, 0.50, match="no"), row("tta_x", 0.61, 0.52), row("tta_xy", 0.62, 0.53)]
            )


if __name__ == "__main__":
    unittest.main()
