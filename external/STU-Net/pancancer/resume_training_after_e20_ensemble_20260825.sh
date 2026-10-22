#!/usr/bin/env bash
set -euo pipefail

# Resume the preserved full STU-Net run after a configurable interim saver has either
# created its qualified archive or stopped before hidden inference.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
stunet_root="${project_root}/external/STU-Net"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
interim_epoch="${STUNET_INTERIM_EPOCH:-20}"
[[ "${interim_epoch}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid interim epoch" >&2; exit 2; }
epoch4=$(printf '%04d' "${interim_epoch}")
public="${STUNET_PUBLIC_OUTPUT:-${stunet_root}/pancancer/public_sweeps/stunetb_e${interim_epoch}_ensemble_20260825}"
save_output="${STUNET_SAVE_OUTPUT:-${stunet_root}/pancancer/public_sweeps/stunetb_e${interim_epoch}_ensemble_saved_20260825}"
saver="${save_output}/TERMINAL.json"
checkpoint="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_epoch_${epoch4}.pth"
resume_checkpoint="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_latest.pth"
training_script="${stunet_root}/pancancer/run_training_after_independent_audit_20260825.sh"
post_gate_script="${STUNET_POST_GATE_SCRIPT:-${training_script}}"
read -r -a active_units <<< "${STUNET_ACTIVE_UNITS:-miccai-stunetb-training-after-independent-20260825.service miccai-stunetb-training-resume-e20-20260825.service}"
resume_unit="${STUNET_RESUME_UNIT:-miccai-stunetb-training-post-e${interim_epoch}-20260825}"
lock="${STUNET_RESUME_LOCK:-${stunet_root}/pancancer/runs/resume_training_after_e${interim_epoch}_ensemble_20260825.lock}"

mkdir -p "$(dirname "${lock}")"
exec 9>"${lock}"
flock -n 9 || { echo "STU-Net post-e${interim_epoch} resume controller already active" >&2;exit 1; }

while :; do
  if [[ ! -s "${saver}" ]]; then printf '%s waiting for epoch-%s ensemble saver terminal\n' "$(date -Ins)" "${interim_epoch}";sleep 30;continue;fi
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${saver}")
  case "${status}" in
    COMPLETE|SKIPPED) break ;;
    FAILED) printf '%s waiting for replacement of failed epoch-%s saver terminal\n' "$(date -Ins)" "${interim_epoch}";sleep 30 ;;
    *) echo "unexpected epoch-${interim_epoch} saver status: ${status}" >&2;exit 2 ;;
  esac
done

"${python_bin}" - "${public}/TERMINAL.json" "${public}/TRAINING_PAUSED.json" \
  "${checkpoint}" "${resume_checkpoint}" "${interim_epoch}" <<'PY'
import hashlib,json,sys,numpy as np,torch
from pathlib import Path
terminal_path,pause_path,checkpoint,resume=map(Path,sys.argv[1:5]);epoch=int(sys.argv[5]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
terminal=json.load(open(terminal_path,encoding="utf-8"));pause=json.load(open(pause_path,encoding="utf-8"))
if terminal.get("status")!="COMPLETE" or terminal.get("predeclared_epochs")!=[epoch]:
    raise SystemExit(f"invalid epoch-{epoch} public handoff")
for payload,key,path in ((pause,"checkpoint",checkpoint),(pause,"resume_checkpoint",resume)):
    item=payload.get(key,{})
    if Path(item.get("path","")).resolve()!=path.resolve() or item.get("sha256")!=sha(path):
        raise SystemExit(f"epoch-{epoch} {key} identity mismatch")
allowed=[np._core.multiarray.scalar,np.dtype,type(np.dtype(np.float64)),type(np.dtype(np.float32)),type(np.dtype(np.int64)),type(np.dtype(np.bool_))]
with torch.serialization.safe_globals(allowed):
    for path in (checkpoint,resume):
        state=torch.load(path,map_location="cpu",weights_only=True)
        if int(state.get("current_epoch",-1))!=epoch or not state.get("optimizer_state"):
            raise SystemExit(f"invalid epoch-{epoch} resume state: {path}")
PY

for unit in "${active_units[@]}"; do
  if systemctl --user is-active --quiet "${unit}"; then exit 0;fi
done
systemd-run --user --unit="${resume_unit}" --collect \
  --property=Restart=on-failure --property=RestartSec=60 \
  --property=Nice=0 --property=CPUWeight=100 --property=IOWeight=100 \
  /usr/bin/bash "${post_gate_script}"
