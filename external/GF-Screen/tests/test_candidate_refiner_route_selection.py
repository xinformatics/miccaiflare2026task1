import unittest

from candidate_refiner_route_selection import EXPECTED_SOURCES, deployment_specification


def payload():
    sources = {name: "discovery" for name in EXPECTED_SOURCES}
    sources["currentgf"] = "currentgf"
    sources["epoch105"] = "epoch105"
    return {
        "selected_strategy": "source_routed",
        "classification_checkpoint_filename": "checkpoint_best_classification.pt",
        "contour_checkpoint_filenames": {
            "currentgf": "checkpoint_best_currentgf_segmentation.pt",
            "epoch105": "checkpoint_best_epoch105_segmentation.pt",
            "discovery": "checkpoint_best_cohort_segmentation.pt",
        },
        "contour_checkpoint_run_families": {
            "currentgf": "primary",
            "epoch105": "primary",
            "discovery": "discovery",
        },
        "source_contour_routes": sources,
    }


class RouteSelectionTests(unittest.TestCase):
    def test_accepts_complete_three_route_policy(self):
        result = deployment_specification(payload())
        self.assertEqual(set(result["routes"]), {"currentgf", "epoch105", "discovery"})
        self.assertEqual(result["routes"]["discovery"]["family"], "discovery")

    def test_accepts_domain_specific_five_route_policy(self):
        value = payload()
        value["contour_checkpoint_filenames"] = {
            "currentgf": "checkpoint_best_currentgf_segmentation.pt",
            "epoch105": "checkpoint_best_epoch105_segmentation.pt",
            "autopet": "checkpoint_best_autopet_segmentation.pt",
            "lndb": "checkpoint_best_lndb_segmentation.pt",
            "lung": "checkpoint_best_lung_segmentation.pt",
        }
        value["contour_checkpoint_run_families"] = {
            name: "primary" for name in value["contour_checkpoint_filenames"]
        }
        value["source_contour_routes"] = {
            "currentgf": "currentgf",
            "epoch105": "epoch105",
            "autopet_weak": "autopet",
            "lndb_weak": "lndb",
            **{f"lung_cell{index}": "lung" for index in range(8)},
        }
        result = deployment_specification(value)
        self.assertEqual(
            set(result["routes"]),
            {"currentgf", "epoch105", "autopet", "lndb", "lung"},
        )
        self.assertEqual(result["sources"]["autopet_weak"], "autopet")
        self.assertEqual(result["sources"]["lung_cell7"], "lung")

    def test_accepts_mixed_source_policy_with_global_fallback(self):
        value = payload()
        value["contour_checkpoint_filenames"] = {
            "global": "checkpoint_best_segmentation.pt",
            "autopet": "checkpoint_best_autopet_segmentation.pt",
        }
        value["contour_checkpoint_run_families"] = {
            "global": "primary",
            "autopet": "primary",
        }
        value["source_contour_routes"] = {
            name: ("autopet" if name == "autopet_weak" else "global")
            for name in EXPECTED_SOURCES
        }
        result = deployment_specification(value)
        self.assertEqual(set(result["routes"]), {"global", "autopet"})
        self.assertEqual(result["sources"]["autopet_weak"], "autopet")
        self.assertEqual(result["sources"]["lndb_weak"], "global")

    def test_rejects_missing_source(self):
        value = payload()
        value["source_contour_routes"].pop("lung_cell7")
        with self.assertRaisesRegex(ValueError, "incomplete source inventory"):
            deployment_specification(value)

    def test_rejects_unused_route(self):
        value = payload()
        value["contour_checkpoint_filenames"]["unused"] = "unused.pt"
        value["contour_checkpoint_run_families"]["unused"] = "primary"
        with self.assertRaisesRegex(ValueError, "every and only"):
            deployment_specification(value)

    def test_rejects_unknown_checkpoint_family(self):
        value = payload()
        value["contour_checkpoint_run_families"]["discovery"] = "other"
        with self.assertRaisesRegex(ValueError, "invalid route checkpoint family"):
            deployment_specification(value)

    def test_rejects_checkpoint_path_instead_of_basename(self):
        value = payload()
        value["contour_checkpoint_filenames"]["currentgf"] = "/tmp/model.pt"
        with self.assertRaisesRegex(ValueError, "safe .pt basename"):
            deployment_specification(value)

    def test_rejects_empty_checkpoint_filename(self):
        value = payload()
        value["contour_checkpoint_filenames"]["currentgf"] = ""
        with self.assertRaisesRegex(ValueError, "safe .pt basename"):
            deployment_specification(value)

    def test_rejects_route_name_that_breaks_tsv_protocol(self):
        value = payload()
        value["contour_checkpoint_filenames"]["bad\tname"] = value[
            "contour_checkpoint_filenames"
        ].pop("discovery")
        value["contour_checkpoint_run_families"]["bad\tname"] = value[
            "contour_checkpoint_run_families"
        ].pop("discovery")
        for source, route in value["source_contour_routes"].items():
            if route == "discovery":
                value["source_contour_routes"][source] = "bad\tname"
        with self.assertRaisesRegex(ValueError, "safe token"):
            deployment_specification(value)


if __name__ == "__main__":
    unittest.main()
