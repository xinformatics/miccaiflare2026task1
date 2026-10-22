#!/usr/bin/env bash
set -euo pipefail

# Two-GPU STU-Net-B smoke gate followed by full 300-epoch partial-label-safe
# fine-tuning. It starts only after the serialized GF/FreeTumor public audit has
# released both GPUs. It never reads validation or hidden images.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
stunet_root="${project_root}/external/STU-Net"
runner="${stunet_root}/pancancer/run_pancancer_stunet.py"
checkpoint="${stunet_root}/pretrained/base_ep4k.model"
upstream="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_saved_20260825/TERMINAL.json"
run_root="${stunet_root}/pancancer/runs/stunetb_partialbox_300ep_20260825"
terminal="${run_root}/TRAINING_TERMINAL.json"
lock="${run_root}/controller.lock"
results_root="${project_root}/nnUNet_results/Dataset501_PanCancer"
smoke_output="${results_root}/STUNetTrainerPanCancerSmoke__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all"
full_output="${results_root}/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all"
expected_checkpoint_sha="7232a6296d320da1d60debe873a1d3fcad74e9404272db8f7440e30d62c5997a"
expected_repo_commit="0652eaf5cafda6bc269942238c3695994f2dc4f0"
idle_memory_mib=4096
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"

mkdir -p "${run_root}"
exec 9>"${lock}"
flock -n 9 || { echo "STU-Net training controller already active" >&2; exit 1; }

write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${checkpoint}" "${runner}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
reason=sys.argv[1];checkpoint,runner,output=map(Path,sys.argv[2:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)} if checkpoint.is_file() else None,
 "runner":{"path":str(runner.resolve()),"sha256":sha(runner)} if runner.is_file() else None,
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}

failure_line=unknown
failure() {
  local status=$? line="${failure_line}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_failure \
    "STU-Net training controller failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap failure EXIT

if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED)
      preserved="${run_root}/TRAINING_TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      mv --no-clobber "${terminal}" "${preserved}"
      ;;
    *) echo "unexpected STU-Net terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

[[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${expected_checkpoint_sha}" ]]
[[ "$(git -C "${stunet_root}" rev-parse HEAD)" == "${expected_repo_commit}" ]]

"${python_bin}" - "${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans.json" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]);plans=json.load(open(path,encoding="utf-8"))
cfg=plans["configurations"].get("3d_fullres_STUNetB_PartialBox",{})
expected=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,1,2]]
if (cfg.get("data_identifier")!="nnUNetPlans_3d_fullres_S4D2W32"
    or cfg.get("patch_size")!=[128,160,160] or cfg.get("spacing")!=[2.4,1.6,1.6]
    or cfg.get("batch_size")!=2
    or cfg.get("architecture",{}).get("arch_kwargs",{}).get("strides")!=expected):
    raise SystemExit("STU-Net plan contract mismatch")
PY

while :; do
  if [[ ! -s "${upstream}" ]]; then
    printf '%s waiting for independent public-audit terminal\n' "$(date -Ins)"
    sleep 30
    continue
  fi
  upstream_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${upstream}")
  case "${upstream_status}" in
    # The save-only consumer has made its final GPU-allocation decision in
    # either state: COMPLETE means an archive was created, while SKIPPED means
    # the public gate rejected it before hidden inference. Both safely release
    # the independent ensemble stage.
    COMPLETE|SKIPPED) break ;;
    FAILED)
      printf '%s waiting for replacement of failed independent public audit\n' \
        "$(date -Ins)"
      sleep 30
      ;;
    *) echo "unexpected independent-audit status: ${upstream_status}" >&2; exit 2 ;;
  esac
done

wait_for_idle_pair() {
  while :; do
    used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
        && "${used0}" -le "${idle_memory_mib}" \
        && "${used1}" -le "${idle_memory_mib}" ]]; then
      break
    fi
    printf '%s waiting for idle STU-Net GPUs: gpu0=%s gpu1=%s MiB\n' \
      "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
}

export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export MPLCONFIGDIR="${run_root}/.matplotlib"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1
# nnU-Net applies this per rank. Eight augmenters on each of two ranks uses the
# requested 16 CPU workers without repeating the earlier 24-worker pressure.
export nnUNet_n_proc_DA=8

if [[ ! -s "${smoke_output}/checkpoint_final.pth" ]]; then
  wait_for_idle_pair
  "${python_bin}" -u "${runner}" \
    Dataset501_PanCancer 3d_fullres_STUNetB_PartialBox all \
    -tr STUNetTrainerPanCancerSmoke -p nnUNetPlans \
    -pretrained_weights "${checkpoint}" -num_gpus 2 \
    > "${run_root}/smoke.log" 2>&1
fi
[[ -s "${smoke_output}/checkpoint_final.pth" ]]

wait_for_idle_pair
if [[ -s "${full_output}/checkpoint_final.pth" ]]; then
  printf '%s full STU-Net final checkpoint already present\n' "$(date -Ins)"
elif [[ -s "${full_output}/checkpoint_latest.pth" ]]; then
  "${python_bin}" -u "${runner}" \
    Dataset501_PanCancer 3d_fullres_STUNetB_PartialBox all \
    -tr STUNetTrainerPanCancer -p nnUNetPlans -num_gpus 2 --c \
    >> "${run_root}/train.log" 2>&1
else
  "${python_bin}" -u "${runner}" \
    Dataset501_PanCancer 3d_fullres_STUNetB_PartialBox all \
    -tr STUNetTrainerPanCancer -p nnUNetPlans \
    -pretrained_weights "${checkpoint}" -num_gpus 2 \
    > "${run_root}/train.log" 2>&1
fi
[[ -s "${full_output}/checkpoint_final.pth" ]]

"${python_bin}" - "${upstream}" "${checkpoint}" "${runner}" \
  "${smoke_output}/checkpoint_final.pth" "${full_output}/checkpoint_final.pth" \
  "${run_root}/smoke.log" "${run_root}/train.log" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
upstream,pretrained,runner,smoke,final,smoke_log,train_log,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
epochs=sorted(final.parent.glob("checkpoint_epoch_*.pth"))
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","model":"STU-Net-B partial-box proposal model",
 "recipe":{"epochs":300,"initial_lr":3e-4,"optimizer":"SGD-Nesterov",
   "scheduler":"PolyLR","global_batch_size":2,"gpus":2,
   "augmentation_workers":16,
   "patch_size":[128,160,160],"spacing":[2.4,1.6,1.6],
   "bbox_context_mm":16,"oversample_foreground_percent":1.0,
   "loss":"trusted-box CE plus Tversky alpha0.3 beta0.7"},
 "upstream":{"path":str(upstream.resolve()),"sha256":sha(upstream)},
 "pretrained":{"path":str(pretrained.resolve()),"sha256":sha(pretrained),"embedded_epoch":4000},
 "runner":{"path":str(runner.resolve()),"sha256":sha(runner)},
 "smoke_checkpoint":{"path":str(smoke.resolve()),"sha256":sha(smoke)},
 "final_checkpoint":{"path":str(final.resolve()),"sha256":sha(final)},
 "trajectory_checkpoints":[{"path":str(p.resolve()),"sha256":sha(p)} for p in epochs],
 "smoke_log":{"path":str(smoke_log.resolve()),"sha256":sha(smoke_log)},
 "train_log":{"path":str(train_log.resolve()),"sha256":sha(train_log)},
 "wandb":{"project":"pancancer-ct","run_id":"stunetbpartial20260825"},
 "training_data":"Dataset501_PanCancer Train-Labeled partial masks only",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

trap - EXIT
cat "${terminal}"
