#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
gf="${workspace}/external/GF-Screen"
old_pgid="${OLD_PGID:-1096807}"
uniform="${gf}/runs/gf_component_box_uniform_p000_ep280_lr3e5_5ep_20260820"
sqrt="${gf}/runs/gf_component_box_uniform_p050_ep280_lr3e5_5ep_20260820"
summary="${gf}/public_sweeps/gf_component_box_uniform_dual_20260820/summary_by_combined.tsv"

training_complete() {
  local run
  for run in "${uniform}" "${sqrt}"; do
    [[ -f "${run}/model_final.pt" ]] || return 1
    [[ $(find "${run}" -maxdepth 1 -name 'checkpoint_epoch_028[1-5].pt' | wc -l) -eq 5 ]] \
      || return 1
    grep -q 'wandb: Synced' "${run}/train.log" || return 1
  done
}

while kill -0 -- "-${old_pgid}" 2>/dev/null; do
  if training_complete; then
    echo "[$(date --iso-8601=seconds)] factorial training complete; reloading grid evaluator"
    kill -TERM -- "-${old_pgid}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 -- "-${old_pgid}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${old_pgid}" 2>/dev/null || true
    cd "${workspace}"
    setsid bash external/GF-Screen/run_component_box_uniform_factorial_20260820.sh \
      > external/GF-Screen/component_box_uniform_factorial_20260820.wrapper.log 2>&1 &
    new_pid=$!
    echo "${new_pid}" > "${workspace}/component_box_uniform_factorial_20260820.pid"
    echo "[$(date --iso-8601=seconds)] restarted factorial evaluator as PID ${new_pid}"
    while kill -0 "${new_pid}" 2>/dev/null; do sleep 15; done
    [[ -s "${summary}" ]] || {
      echo "factorial evaluator exited without summary" >&2
      exit 1
    }
    echo "[$(date --iso-8601=seconds)] factorial evaluation complete"
    exit 0
  fi
  sleep 15
done

if [[ -s "${summary}" ]]; then
  echo "[$(date --iso-8601=seconds)] factorial completed without reload"
  exit 0
fi
echo "factorial exited before training completion" >&2
exit 1
