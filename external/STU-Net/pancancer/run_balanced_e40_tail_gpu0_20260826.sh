#!/usr/bin/env bash
set -euo pipefail

# Bounded acceleration for the user-requested epoch-40 interim evaluation.
# The main controller advances from part 0 on GPU 1. This helper owns only a
# bounded tail block (40..49 by default, or the explicitly assigned 35..39
# follow-up) on otherwise idle GPU 0, predicts one CT per process into an
# isolated directory, and publishes atomically after validating its terminal.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
predictor="${root}/predict_stunet_safe.py"
images="${project_root}/validation/Validation-Public-Images"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160Balanced__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="${model}/fold_all/checkpoint_inference_epoch_0040.pth"
checkpoint_sha="3015cdf2d9b267989fd2fde2939cdac8670c55c31b96fd50f59eadf9da2a29f6"
epoch_root="${root}/public_sweeps/stunetb_balanced_e40_e80_20260826/epoch_0040"
predictions="${epoch_root}/predictions"
part_start="${BALANCED_TAIL_PART_START:-40}"
part_end="${BALANCED_TAIL_PART_END:-49}"
[[ "${part_start}" =~ ^[0-9]+$ && "${part_end}" =~ ^[0-9]+$ ]]
(( part_start >= 35 && part_start <= part_end && part_end <= 49 ))
scratch="${epoch_root}/parallel_gpu0_${part_start}_${part_end}"
complete="${scratch}/TAIL_GPU0_COMPLETE.json"

mkdir -p "${scratch}/predictions" "${predictions}"
exec 9>"${scratch}/controller.lock"
flock -n 9 || { echo "balanced epoch-40 GPU0 tail helper already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
[[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${checkpoint_sha}" ]]

for part_id in $(seq "${part_start}" "${part_end}"); do
  part4=$(printf '%04d' "${part_id}")
  target_terminal="${epoch_root}/BALANCED_E0040_CASE_${part4}.json"
  scratch_terminal="${scratch}/BALANCED_E0040_CASE_${part4}.json"
  if [[ -s "${target_terminal}" ]]; then
    continue
  fi
  "${python_bin}" -u "${predictor}" --images "${images}" \
    --output "${scratch}/predictions" --model-dir "${model}" \
    --checkpoint "$(basename "${checkpoint}")" --gpu 0 \
    --part-id "${part_id}" --num-parts 50 --workers 1 \
    --save-probabilities --sequential --dataset-role public \
    --terminal "${scratch_terminal}" >"${scratch}/case_${part4}.log" 2>&1

  readarray -t names < <("${python_bin}" - "${scratch_terminal}" "${checkpoint_sha}" "${part_id}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); expected_sha=sys.argv[2]; part=int(sys.argv[3])
cases=p.get("cases") or []; probs=p.get("probability_files") or []
if (p.get("status")!="COMPLETE" or p.get("dataset_role")!="public"
    or p.get("part_id")!=part or p.get("num_parts")!=50
    or p.get("case_count")!=1 or len(cases)!=1 or len(probs)!=1
    or p.get("checkpoint",{}).get("sha256")!=expected_sha
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid GPU0 tail terminal")
print(cases[0]); print(probs[0])
PY
  )
  mask="${scratch}/predictions/${names[0]}"
  probability="${scratch}/predictions/${names[1]}"
  [[ -s "${mask}" && -s "${probability}" ]]
  [[ ! -e "${predictions}/${names[0]}" && ! -e "${predictions}/${names[1]}" ]]
  mv "${mask}" "${predictions}/${names[0]}"
  mv "${probability}" "${predictions}/${names[1]}"
  mv "${scratch_terminal}" "${target_terminal}"
done

"${python_bin}" - "${complete}" "${checkpoint_sha}" "${part_start}" "${part_end}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
output=Path(sys.argv[1]); sha=sys.argv[2]; start=int(sys.argv[3]); end=int(sys.argv[4])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","parts":list(range(start,end+1)),"checkpoint_sha256":sha,
 "dataset_role":"public","public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
