#!/usr/bin/env bash
set -euo pipefail

# Label-blind scale-out of the 210-case-selected weak proposal bank to all 101
# autoPET and 101 LNDb cases in the balanced 707-case training set. Selection
# is allowed to use complete 210-case training labels, but this controller does
# not read any label, public case, hidden case, or submission artifact.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${gf}/analysis/fpr_training_mining_707_20260822/manifest.json"
inputs="${root}/analysis/weak_domain_biomedparse_707_20260823"
initial="${root}/analysis/weak_domain_biomedparse_enrichment_20260823"
extension="${root}/analysis/weak_domain_biomedparse_recall_extension_20260823"
weak_transfer="${gf}/analysis/candidate_refiner_v2_weak_transfer_oof_20260823/TRANSFER_OOF_COMPLETE.txt"
epoch105="${gf}/analysis/candidate_refiner_v2_training_707_20260823/epoch105_abd_g090_l050"
epoch105_complete="${epoch105}/PROPOSALS_COMPLETE.json"
output="${root}/analysis/weak_domain_biomedparse_707_selected_20260823"
plan="${output}/SELECTION_PLAN.json"
complete="${output}/PROPOSALS_COMPLETE.json"
skipped="${output}/INSUFFICIENT_HEADROOM.json"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "707 weak-domain selected inference already active" >&2; exit 1; }
[[ -s "${complete}" || -s "${skipped}" ]] && exit 0
while [[ ! -s "${inputs}/INPUTS_COMPLETE.json" \
    || ! -s "${extension}/EXTENSION_COMPLETE.txt" ]]; do
  printf '%s waiting for 707 inputs and frozen 210 proposal selection\n' "$(date -Ins)"
  sleep 30
done
for required in "${manifest}" "${initial}/selection.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

# Freeze the exact operating point and prompt family before any 707 inference.
"${python_bin}" - "${initial}" "${extension}" "${plan}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path

initial,extension,output=map(Path,sys.argv[1:])
if (extension/"selection.json").is_file():
    stage="extension"
    selection_path=extension/"selection.json"
    selection=json.loads(selection_path.read_text(encoding="utf-8"))
    source_root=extension
    prompts={
      "autopet":[
        "whole body lesion","tumor","malignant tumor","PET-avid lesion",
        "lymph node metastasis","bone metastasis","liver metastasis",
        "lung metastasis",
      ],
      "lndb":[
        "pulmonary nodule","lung nodule","lung mass","lung tumor","lung cancer",
      ],
    }
    disable_nms=True
else:
    stage="initial"
    selection_path=initial/"selection.json"
    selection=json.loads(selection_path.read_text(encoding="utf-8"))
    source_root=initial
    prompts={
      "autopet":["malignant lesion","metastasis","cancer","abnormal mass"],
      "lndb":["lung tumor","lung nodule"],
    }
    disable_nms=False
threshold=float(selection["selected"]["existence_threshold"])
if threshold not in (0.10,0.20,0.30,0.50,0.70):
    raise SystemExit(f"unexpected selected threshold: {threshold}")
payload={
  "schema_version":1,
  "created_at":datetime.now().astimezone().isoformat(),
  "selection_stage":stage,
  "selection":str(selection_path.resolve()),
  "selection_decision":selection["decision"],
  "pilot_prediction_root":str(source_root.resolve()),
  "existence_threshold":threshold,
  "disable_nms":disable_nms,
  "prompts":prompts,
  "selection_used_complete_210_training_labels":True,
  "inference_reads_labels":False,
  "public_or_hidden_data_read":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY

threshold_tag=$("${python_bin}" -c \
  'import json,sys;print("{:03d}".format(round(100*float(json.load(open(sys.argv[1]))["existence_threshold"]))))' \
  "${plan}")
selection_stage=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["selection_stage"])' "${plan}")
disable_nms=$("${python_bin}" -c \
  'import json,sys;print(int(json.load(open(sys.argv[1]))["disable_nms"]))' "${plan}")
selection_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["selection_decision"])' "${plan}")
if [[ "${selection_decision}" != SUFFICIENT_PROPOSAL_HEADROOM \
    && "${selection_decision}" != USEFUL_PROPOSAL_GAIN ]]; then
  "${python_bin}" - "${plan}" "${skipped}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
plan_path,output=map(Path,sys.argv[1:])
plan=json.loads(plan_path.read_text(encoding="utf-8"))
payload={
  "created_at":datetime.now().astimezone().isoformat(),
  "decision":"SKIP_707_WEAK_INFERENCE",
  "reason":"selected 210-case weak proposal bank met neither the high-headroom nor useful-gain gate",
  "selection_plan":str(plan_path.resolve()),
  "selection_decision":plan["selection_decision"],
  "gpu_used":False,
  "labels_read":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY
  exit 0
fi
if [[ "${selection_stage}" == extension ]]; then
  pilot_root="${extension}"
else
  pilot_root="${initial}"
fi

reuse_pilot() {
  local tag="$1"
  local destination="${output}/${tag}_npz"
  mkdir -p "${destination}"
  "${python_bin}" - "${inputs}/${tag}/inputs" "${pilot_root}/${tag}_npz" \
    "${destination}" "${plan}" "${tag}" "${threshold_tag}" <<'PY'
import json,os,sys
from pathlib import Path
import numpy as np

inputs,pilot,destination,plan_path=map(Path,sys.argv[1:5])
tag,threshold_tag=sys.argv[5:]
plan=json.loads(plan_path.read_text(encoding="utf-8"))
expected_prompts=np.asarray(plan["prompts"][tag])
key=f"segs_e{threshold_tag}"
available={path.name for path in inputs.glob("*.npz")}
reused=0
for source in sorted(pilot.glob("*.npz")):
    if source.name not in available:
        continue
    with np.load(source,allow_pickle=False) as archive:
        if key not in archive or "prompts" not in archive:
            raise SystemExit(f"incomplete pilot prediction: {source}")
        if not np.array_equal(np.asarray(archive["prompts"]),expected_prompts):
            raise SystemExit(f"pilot prompt mismatch: {source}")
    target=destination/source.name
    if target.exists():
        if not os.path.samefile(source,target):
            raise SystemExit(f"pilot reuse collision: {target}")
    else:
        os.link(source,target)
    reused+=1
if reused!=30:
    raise SystemExit(f"expected 30 reusable pilot predictions for {tag}, found {reused}")
print(f"reused {reused} {tag} pilot predictions")
PY
}
reuse_pilot autopet
reuse_pilot lndb

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
infer_bank() {
  local gpu="$1" tag="$2"; shift 2
  local prompt_args=() nms_args=() prompt
  for prompt in "$@"; do prompt_args+=(--prompt "${prompt}"); done
  [[ "${disable_nms}" -eq 1 ]] && nms_args+=(--disable-nms)
  CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
    ../miccai_inference.py --input-dir "${inputs}/${tag}/inputs" \
    --output-dir "${output}/${tag}_npz" "${prompt_args[@]}" --device cuda \
    --slice-batch-size 8 "${nms_args[@]}" \
    --existence-threshold "0.${threshold_tag#0}" --continue-prediction \
    > "${output}/${tag}.inference.log" 2>&1
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  free0=0; free1=0
  [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 4096 ]] && free0=1
  [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]] && free1=1
  [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && break
  printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

infer_autopet() {
  local gpu="$1"
  if [[ "${selection_stage}" == extension ]]; then
    infer_bank "${gpu}" autopet \
      "whole body lesion" "tumor" "malignant tumor" "PET-avid lesion" \
      "lymph node metastasis" "bone metastasis" "liver metastasis" \
      "lung metastasis"
  else
    infer_bank "${gpu}" autopet \
      "malignant lesion" "metastasis" "cancer" "abnormal mass"
  fi
}
infer_lndb() {
  local gpu="$1"
  if [[ "${selection_stage}" == extension ]]; then
    infer_bank "${gpu}" lndb \
      "pulmonary nodule" "lung nodule" "lung mass" "lung tumor" "lung cancer"
  else
    infer_bank "${gpu}" lndb "lung tumor" "lung nodule"
  fi
}
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  infer_autopet 0 & p0=$!; infer_lndb 1 & p1=$!
  status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
elif [[ "${free0}" -eq 1 ]]; then
  infer_autopet 0
  infer_lndb 0
else
  infer_autopet 1
  infer_lndb 1
fi

for tag in autopet lndb; do
  [[ $(find "${output}/${tag}_npz" -maxdepth 1 -name '*.npz' | wc -l) -eq 101 ]]
  destination="${output}/${tag}_nifti"
  mkdir -p "${destination}"
  "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
    --images "${inputs}/${tag}/images" --predictions "${output}/${tag}_npz" \
    --output "${destination}" --segmentation-key "segs_e${threshold_tag}" \
    --continue-prediction > "${output}/${tag}.convert.log" 2>&1
  [[ $(find "${destination}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 101 ]]
done

# Epoch105 is used only as a complete geometry/reference stream when sparse
# weak-domain masks are expanded to all 707 cases. Do not leave an available
# GPU idle while that independent proposal producer is still queued.
while [[ ! -s "${epoch105_complete}" ]]; do
  printf '%s waiting for complete epoch105 reference proposals after weak inference\n' \
    "$(date -Ins)"
  sleep 30
done
[[ -s "${weak_transfer}" ]] || {
  echo "epoch105 marker appeared before its required weak-transfer gate" >&2
  exit 2
}

for tag in autopet lndb; do
  destination="${output}/${tag}_nifti"
  "${python_bin}" -u "${gf}/fuse_sparse_overlays.py" \
    --manifest "${manifest}" --base "${epoch105}" --overlay "${destination}" \
    --empty-base --output "${output}/${tag}_complete" \
    --marker "${output}/${tag}_complete.json" --workers 4 \
    > "${output}/${tag}_complete.log" 2>&1
done

"${python_bin}" - "${manifest}" "${inputs}" "${output}" "${plan}" \
  "${complete}" "${threshold_tag}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
import numpy as np

manifest,inputs,root,plan_path,output=map(Path,sys.argv[1:6])
threshold_tag=sys.argv[6]
plan=json.loads(plan_path.read_text(encoding="utf-8"))
names={str(item["prediction_name"]) for item in json.loads(manifest.read_text(encoding="utf-8"))["items"]}
if len(names)!=707:
    raise SystemExit(f"expected 707 manifest names, found {len(names)}")
bank_reports={}
for tag in ("autopet","lndb"):
    input_names={path.name for path in (inputs/tag/"inputs").glob("*.npz")}
    predictions=sorted((root/f"{tag}_npz").glob("*.npz"))
    if len(input_names)!=101 or {path.name for path in predictions}!=input_names:
        raise SystemExit(f"{tag} prediction coverage mismatch")
    expected_prompts=np.asarray(plan["prompts"][tag])
    for path in predictions:
        with np.load(path,allow_pickle=False) as archive:
            if f"segs_e{threshold_tag}" not in archive:
                raise SystemExit(f"missing selected mask: {path}")
            if not np.array_equal(np.asarray(archive["prompts"]),expected_prompts):
                raise SystemExit(f"prompt mismatch: {path}")
    complete_names={path.name for path in (root/f"{tag}_complete").glob("*.nii.gz")}
    if complete_names!=names:
        raise SystemExit(f"{tag} complete-source coverage mismatch")
    report=json.loads((root/f"{tag}_complete.json").read_text(encoding="utf-8"))
    if report.get("cases")!=707 or report.get("empty_base") is not True or report.get("labels_read") is not False:
        raise SystemExit(f"{tag} sparse completion audit failed")
    bank_reports[tag]={
      "inference_cases":101,
      "reused_pilot_cases":30,
      "generated_additional_cases":71,
      "complete_source":str((root/f"{tag}_complete").resolve()),
      "complete_source_audit":str((root/f"{tag}_complete.json").resolve()),
      "sha256_inventory":report["sha256_inventory"],
    }
payload={
  "schema_version":1,
  "created_at":datetime.now().astimezone().isoformat(),
  "cases":707,
  "selected_from_complete_210_training_labels":True,
  "selection_plan":str(plan_path.resolve()),
  "existence_threshold":plan["existence_threshold"],
  "disable_nms":plan["disable_nms"],
  "banks":bank_reports,
  "component_policy_for_training":"source_separate",
  "labels_read":False,
  "public_or_hidden_data_read":False,
  "submission_created":False,
}
temporary=output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload,indent=2))
PY
