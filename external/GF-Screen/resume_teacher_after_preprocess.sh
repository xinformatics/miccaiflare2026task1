#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
marker="${cache_root}/PREPROCESSING_COMPLETE.json"
handoff_log="runs/resume_teacher_after_preprocess.log"

exec > >(tee -a "${handoff_log}") 2>&1

echo "Waiting for ${marker}"
until [[ -f "${marker}" ]]; do
  sleep 30
done

echo "Preprocessing complete; running read-only cache audit"
MPLCONFIGDIR=/tmp/matplotlib-gf-screen-audit \
  /home/m348911/miniconda3/envs/radoncAI/bin/python audit_preprocessed_cache.py \
    --cache-dir "${cache_root}" --per-region 2

# Release only stale training processes owned by this user and launched from
# this MICCAI workspace. Never interfere with another workspace or user.
workspace_root="/mnt/nvme2-data/shashank/MICCAI26"
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' | sort -u
)
for pid in "${gpu_pids[@]}"; do
  [[ "${pid}" =~ ^[0-9]+$ ]] || continue
  [[ -r "/proc/${pid}/cmdline" ]] || continue
  process_user="$(ps -o user= -p "${pid}" 2>/dev/null | xargs || true)"
  process_cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
  process_cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "${process_user}" == "$(id -un)" \
        && "${process_cwd}" == "${workspace_root}"* \
        && "${process_cmd}" =~ (GF_RL_train\.py|nnUNetv2_train) ]]; then
    echo "Stopping stale MICCAI GPU process pid=${pid}: ${process_cmd}"
    kill -TERM "${pid}"
  fi
done

echo "Cache audit passed; waiting for both GPUs to have <2 GiB allocated"
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

echo "Both GPUs are free; launching teacher-guided fine-tuning"
exec bash run_teacher_pu_finetune.sh
