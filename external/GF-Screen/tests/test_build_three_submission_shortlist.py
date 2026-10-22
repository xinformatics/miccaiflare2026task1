import csv
import tempfile
import unittest
from pathlib import Path

from build_three_submission_shortlist import discover, resolve_generated, threshold_tag


class BuildThreeSubmissionShortlistTests(unittest.TestCase):
    def test_threshold_tag_matches_packaging_scripts(self):
        self.assertEqual(threshold_tag("0.5"), "0p50")
        self.assertEqual(threshold_tag("0.60"), "0p60")

    def test_generated_paths_are_resolved_against_stage_checkout(self):
        self.assertEqual(resolve_generated("scores/a.csv", Path("/stage")), Path("/stage/scores/a.csv"))
        self.assertEqual(resolve_generated("/fixed/a.csv", Path("/stage")), Path("/fixed/a.csv"))

    def test_discover_matches_all_three_packaging_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            gf = project / "external" / "GF-Screen"
            biomed = project / "external" / "BiomedParse"
            submissions = project / "submissions"
            submissions.mkdir(parents=True)
            stages = [
                (
                    gf / "public_sweeps/long_focus_dual_20260820/prompted_ranking",
                    {
                        "variant": "context24",
                        "checkpoint_tag": "checkpoint_epoch_0300",
                        "lesion_threshold": "0.60",
                        "min_mm3": "500",
                        "prompt_metrics": "long.metrics.csv",
                        "prompt_robust": "long.robust.json",
                        "prompt_decision": "PROMOTE",
                    },
                ),
                (
                    gf / "public_sweeps/long_focus_wise_dual_20260820/prompted_ranking",
                    {
                        "variant": "component",
                        "checkpoint_tag": "wise_focus_alpha0p500",
                        "lesion_threshold": "0.50",
                        "min_mm3": "1000",
                        "prompt_metrics": "wise.metrics.csv",
                        "prompt_robust": "wise.robust.json",
                        "prompt_decision": "PROMOTE",
                    },
                ),
                (
                    biomed / "public_sweeps/automatic_text_proposals_20260820/prompted_ranking",
                    {
                        "variant": "guided",
                        "score_tag": "guided_prompt_or_mm100",
                        "selected_mode": "guided",
                        "prompt_metrics": str(biomed / "guided.metrics.csv"),
                        "prompt_robust": str(biomed / "guided.robust.json"),
                    },
                ),
                (
                    gf / "public_sweeps/nnunet_guided_focus_20260820/prompted_ranking",
                    {
                        "checkpoint_tag": "epoch_0099",
                        "lesion_threshold": "0.60",
                        "min_mm3": "500",
                        "proposal_min_voxels": "1",
                        "prompt_metrics": "nnunet.metrics.csv",
                        "prompt_robust": "nnunet.robust.json",
                    },
                ),
            ]
            for stage, row in stages:
                stage.mkdir(parents=True)
                with (stage / "selected.tsv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=row, delimiter="\t")
                    writer.writeheader()
                    writer.writerow(row)
            sidecar_root = gf / "public_sweeps/optional"
            sidecar_root.mkdir(parents=True)
            sidecar_row = {
                "name": "optional_tta",
                "family": "focus_tta",
                "metrics": "optional.metrics.csv",
                "submission": "optional.zip",
                "gate": "optional.robust.json",
            }
            with (sidecar_root / "submission_candidate.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sidecar_row, delimiter="\t")
                writer.writeheader()
                writer.writerow(sidecar_row)
            for path in (
                gf / "long.metrics.csv", gf / "long.robust.json",
                gf / "wise.metrics.csv", gf / "wise.robust.json",
                gf / "nnunet.metrics.csv", gf / "nnunet.robust.json",
                biomed / "guided.metrics.csv", biomed / "guided.robust.json",
                submissions / "gf_long20_context24_checkpoint_epoch_0300_l0p60_mm500_prompt_or_e050.zip",
                submissions / "gf_longwise_component_wise_focus_alpha0p500_l0p50_mm1000_prompt_or_e050.zip",
                submissions / "gf_biomedparse_guided_prompt_or_mm100_prompt_or_e050.zip",
                submissions / "gf_nnunet_guided_epoch_0099_v1_l0.60_mm500_prompt_or_e050.zip",
                sidecar_root / "optional.metrics.csv",
                sidecar_root / "optional.zip",
                sidecar_root / "optional.robust.json",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            specs = discover(project)
            self.assertEqual(
                [spec.archive.name for spec in specs],
                [
                    "gf_long20_context24_checkpoint_epoch_0300_l0p60_mm500_prompt_or_e050.zip",
                    "gf_longwise_component_wise_focus_alpha0p500_l0p50_mm1000_prompt_or_e050.zip",
                    "gf_biomedparse_guided_prompt_or_mm100_prompt_or_e050.zip",
                    "gf_nnunet_guided_epoch_0099_v1_l0.60_mm500_prompt_or_e050.zip",
                    "optional.zip",
                ],
            )


if __name__ == "__main__":
    unittest.main()
