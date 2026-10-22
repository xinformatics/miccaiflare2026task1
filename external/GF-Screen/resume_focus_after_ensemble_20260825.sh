#!/usr/bin/env bash
set -euo pipefail

# Resume the preserved checkpoint-303 Focus optimizer state only after the
# independent STU-Net ensemble route has made its public-gated save decision.
# This controller never reads predictions or creates/uploads a submission.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_focus_pu_long30_e283_to313_20260825/checkpoint_epoch_0303.pt"
deferral="${root}/public_sweeps/gf_focus_pu_e303_interim_saved_20260825/TRAINING_DEFERRED_FOR_ENSEMBLE.json"
ensemble_terminal="${project}/external/STU-Net/pancancer/public_sweeps/stunetb_ensemble_saved_20260825/TERMINAL.json"
resume_unit="miccai-long-focus-pu-resume-after-ensemble-20260825"
lock="${root}/runs/resume_focus_after_ensemble_20260825.lock"

mkdir -p "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "post-ensemble Focus resume controller already active" >&2; exit 1; }

while :; do
  if [[ ! -s "${ensemble_terminal}" ]]; then
    printf '%s waiting for STU-Net ensemble saver terminal\n' "$(date -Ins)"
    sleep 30
    continue
  fi
  ensemble_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${ensemble_terminal}")
  case "${ensemble_status}" in
    COMPLETE|SKIPPED) break ;;
    FAILED)
      printf '%s waiting for replacement of failed STU-Net ensemble terminal\n' "$(date -Ins)"
      sleep 30
      ;;
    *) echo "unexpected STU-Net ensemble status: ${ensemble_status}" >&2; exit 2 ;;
  esac
done

for file in "${checkpoint}" "${deferral}"; do
  [[ -s "${file}" ]] || { echo "missing preserved Focus resume asset: ${file}" >&2; exit 2; }
done
"${python_bin}" - "${checkpoint}" "${deferral}" "${ensemble_terminal}" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
checkpoint,deferral_path,terminal_path=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
deferred=json.load(open(deferral_path,encoding="utf-8"))
if deferred.get("status")!="DEFERRED": raise SystemExit("invalid Focus deferral status")
item=deferred.get("resume_checkpoint",{})
if Path(item.get("path","")).resolve()!=checkpoint.resolve() or item.get("sha256")!=sha(checkpoint):
    raise SystemExit("preserved checkpoint identity mismatch")
if Path(deferred.get("resume_after_terminal","")).resolve()!=terminal_path.resolve():
    raise SystemExit("ensemble terminal identity mismatch")
state=torch.load(checkpoint,map_location="cpu",weights_only=True)
if int(state.get("epoch",-1))!=302 or not state.get("optimizer_seg") or not state.get("scheduler_seg"):
    raise SystemExit("checkpoint-303 resume state mismatch")
PY

while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 1024 ]]; then break; fi
  printf '%s waiting for post-ensemble Focus GPUs: gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

if systemctl --user is-active --quiet "${resume_unit}.service"; then exit 0; fi
if [[ "$(systemctl --user show -p LoadState --value "${resume_unit}.service" 2>/dev/null || true)" == loaded ]]; then
  systemctl --user reset-failed "${resume_unit}.service" 2>/dev/null || true
  systemctl --user start "${resume_unit}.service"
else
  systemd-run --user --unit="${resume_unit}" --collect \
    --property=Nice=0 --property=CPUWeight=100 --property=IOWeight=100 \
    /usr/bin/bash "${root}/run_long_focus_pu_after_direct_public_20260825.sh"
fi

