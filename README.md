# MICCAI FLARE 2026 — Task 1

Code for automatic pan-cancer CT lesion segmentation, including Glance–Focus training,
whole-volume experts, evaluation, and Docker inference packaging.

This is a code-only snapshot of the research workspace. Model weights will be distributed
separately on **Zenodo**; the record/DOI has not yet been published. Challenge datasets,
checkpoints, predictions, experiment outputs, and Docker image archives are excluded.

## Code layout

| Path | Contents |
| --- | --- |
| `external/GF-Screen/` | Glance–Focus models, training, inference, selection, and tests |
| `external/GF-Screen/docker_submission/` | Deployment Dockerfiles, entry points, and validation |
| `external/FLARE25-Task1-winner/` | Whole-volume nnU-Net expert and project adaptations |
| `external/STU-Net/` | STU-Net source and partial-label training/inference |
| `nnunet/` | Dataset preparation, trainers, public evaluation, and packaging utilities |
| `handoff_bundle/code/` | Organizer-compatible evaluation utilities |
| Other `external/` and `third_party/` directories | Auxiliary research dependencies and experiments |

Start with the README and dependency/build files in the component you intend to use.
These components use different environments; there is no single environment file for all
experiments. The historical deployment environment and entry point are defined in
`external/GF-Screen/docker_submission/Dockerfile` and `predict.sh`.

## Reproducing experiments

1. Obtain the FLARE datasets from the organizers under their data-use terms.
2. Install the requirements for the selected component. Review its scripts for local
   data, output, and checkpoint paths before running them.
3. Obtain the required weights from the forthcoming Zenodo release and place them at
   the paths expected by that component. Docker builds requiring weights will remain
   incomplete until those external artifacts are supplied.
4. Run the component's training, inference, or evaluation entry point.

Historical orchestration scripts contain machine-specific paths and service assumptions.
This snapshot has not been validated as a portable, end-to-end installation. Task 2
RECIST segmentation is maintained separately and is not included as a complete pipeline.

## Model distribution

Keep model artifacts outside Git. The Zenodo release should provide checkpoint filenames,
SHA-256 checksums, model/configuration descriptions, and the corresponding Git commit.
Once published, add the Zenodo DOI and download instructions here. No Zenodo upload is
performed by this repository setup.

`.gitignore` excludes common artifact formats. Run `python3 scripts/check_code_only.py`
before pushing to check tracked files for forbidden artifact formats and files larger than 2 MB.

## Attribution and licensing

Upstream license and notice files are retained alongside their source. Those licenses
apply to their respective components; no new blanket license is assigned to project
modifications. Review component licenses before reuse or redistribution. See
`UPSTREAM_SOURCES.md` for the available upstream origins.
