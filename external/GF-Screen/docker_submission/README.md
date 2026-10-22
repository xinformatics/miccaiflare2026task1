# GF-Screen Task 1 container

> **Current status (2026-08-27): historical container only.** This image
> reproduces the older single-GF deployment, not the verified
> GF/BiomedParse/CBG/STU runtime candidate. Do not use it as the final test image
> until the gates in `WINNER_DEPLOYMENT_GAP_20260827.md` pass.

The historical image follows the organizer command exactly: it reads NIfTI CT images from
`/workspace/inputs`, writes flat binary masks to `/workspace/outputs`, and starts
with `sh predict.sh`. It uses the full 75% GF-Screen grid for ordinary scans and
the 50% grid above 30 million pre-crop resampled voxels to protect the 60-second
per-case limit.

Build and test from the GF-Screen repository root:

```bash
docker build -f docker_submission/Dockerfile -t pancancerseg:latest .
docker run --gpus 'device=0' -m 28G --rm \
  -v "$PWD/docker_submission/smoke_inputs:/workspace/inputs:ro" \
  -v "$PWD/docker_submission/smoke_outputs:/workspace/outputs" \
  pancancerseg:latest /bin/bash -c "sh predict.sh"
python docker_submission/validate_outputs.py \
  --input docker_submission/smoke_inputs \
  --output docker_submission/smoke_outputs
```

The confirmed team tag is `mayo_clinic:latest`:

```bash
docker tag pancancerseg:latest mayo_clinic:latest
docker save mayo_clinic:latest -o mayo_clinic.tar
gzip -1 mayo_clinic.tar
```

The replacement image is intentionally not built yet. It must first pass the
serialized public-label-free gates recorded in
`winner_deployment_manifest_20260827.json`: persistent dual-prompt BiomedParse,
three-route shared Focus, exact pre-STU anchor composition, and sparse-STU
runtime-candidate equivalence. Passing source-level unit tests alone does not
authorize packaging.
