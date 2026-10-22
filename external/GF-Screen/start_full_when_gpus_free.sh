#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

log="runs/start_full_when_gpus_free.log"
exec > >(tee -a "${log}") 2>&1

echo "Waiting for both GPUs to have <2 GiB allocated"
while true; do
  mapfile -t memory_used < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true
  )
  if [[ ${#memory_used[@]} -eq 2 ]] \
      && (( memory_used[0] < 2048 )) \
      && (( memory_used[1] < 2048 )); then
    break
  fi
  sleep 30
done

echo "Both GPUs are free; launching full-model GF-Screen fine-tuning"
exec bash run_full_finetune.sh
