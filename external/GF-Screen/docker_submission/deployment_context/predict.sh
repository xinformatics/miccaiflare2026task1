#!/usr/bin/env sh
set -eu

cd /workspace
exec python -u deployment_inference.py \
  --input /workspace/inputs \
  --output /workspace/outputs \
  --mode "${MAYO_ABLATION:-gf}"
