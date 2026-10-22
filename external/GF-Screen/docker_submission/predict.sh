#!/usr/bin/env sh
set -eu

cd /workspace
exec python -u inference.py \
  --input /workspace/inputs \
  --output /workspace/outputs \
  --checkpoint /workspace/model.pt \
  --glance-threshold 0.5 \
  --lesion-threshold 0.5 \
  --min-component-mm3 1000 \
  --focus-batch-size 4 \
  --overlap 0.75 \
  --overlap-z 0.75 \
  --large-volume-mvox 30 \
  --large-overlap 0.5
