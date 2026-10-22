#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
runner="${root}/run_pancancer_stunet.py"
validator="${root}/validate_post_e100_branch.py"
run_root="${root}/runs/stunetb_partialbox_300ep_20260825"
output="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all"
source="${output}/checkpoint_epoch_0100.pth"
latest="${output}/checkpoint_latest.pth"
final="${output}/checkpoint_final.pth"
terminal="${run_root}/TRAINING_TERMINAL.json"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"

exec 9>"${run_root}/post_e100_gpu0.lock"
flock -n 9 || { echo "base GPU-0 continuation already active" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=0
export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export MPLCONFIGDIR="${run_root}/.matplotlib_gpu0"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export nnUNet_n_proc_DA=8

if [[ ! -s "${final}" ]]; then
  "${python_bin}" "${validator}" --mode base --source "${source}" --latest "${latest}"
  while :; do
    used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
    printf '%s waiting for idle physical GPU 0: %s MiB\n' "$(date -Ins)" "${used}"
    sleep 30
  done
  "${python_bin}" -u "${runner}" \
    Dataset501_PanCancer 3d_fullres_STUNetB_PartialBox all \
    -tr STUNetTrainerPanCancer -p nnUNetPlans -num_gpus 1 --c \
    >> "${run_root}/train.log" 2>&1
fi
[[ -s "${final}" ]]

"${python_bin}" - "${source}" "${final}" "${runner}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source,final,runner,output=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","model":"STU-Net-B global-box continuation",
 "recipe":{"source_epoch":100,"target_epoch":300,"physical_gpu":0,"num_gpus":1,
   "global_batch_size":2,"trusted_region":"single expanded foreground bbox"},
 "source_checkpoint":{"path":str(source.resolve()),"sha256":sha(source)},
 "final_checkpoint":{"path":str(final.resolve()),"sha256":sha(final)},
 "runner":{"path":str(runner.resolve()),"sha256":sha(runner)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
