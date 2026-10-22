import unittest

from rank_prompted_manifest import rank_key, ranking_tier, safe_tag


class RankPromptedManifestTests(unittest.TestCase):
    def test_tiers_preserve_nsd_and_prioritize_robust_promotion(self):
        self.assertEqual(ranking_tier("PROMOTE", 0.4, 0.3, 0.6, 0.5), "PROMOTE")
        self.assertEqual(ranking_tier("KEEP_ANCHOR", 0.61, 0.51, 0.6, 0.5), "NONREGRESS")
        self.assertEqual(ranking_tier("KEEP_ANCHOR", 0.70, 0.49, 0.6, 0.5), "FALLBACK")

    def test_rank_is_tier_first_then_deployed_score(self):
        nonregress = {
            "deployment_tier": "NONREGRESS",
            "prompt_combined": "0.55",
            "prompt_DSC": "0.60",
            "prompt_NSD": "0.50",
        }
        fallback = {
            "deployment_tier": "FALLBACK",
            "prompt_combined": "0.80",
            "prompt_DSC": "0.90",
            "prompt_NSD": "0.70",
        }
        self.assertGreater(rank_key(nonregress), rank_key(fallback))

    def test_tag_sanitization_is_stable(self):
        self.assertEqual(safe_tag("model/l0.50 mm500"), "model_l0.50_mm500")


if __name__ == "__main__":
    unittest.main()
