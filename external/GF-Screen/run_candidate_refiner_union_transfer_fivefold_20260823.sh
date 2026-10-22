#!/usr/bin/env bash
set -euo pipefail

# Build an exact five-fold transfer anchor on the union-native patch index.
# Every prediction is produced by the matching fixed epoch105-trained fold.
# No optimization, public/hidden data, packaging, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
analysis="${root}/analysis/candidate_refiner_union_native_training_20260823"
prep_terminal="${analysis}/PREPARATION_TERMINAL.json"
prompt_terminal="${root}/analysis/focus_prompt_candidate_refiner_pipeline_20260823/PIPELINE_TERMINAL.json"
output="${analysis}/transfer_fivefold"
selection="${output}/checkpoint_selection.json"
selected_oof="${output}/five_fold_oof_summary.json"
terminal="${output}/UNION_TRANSFER_FIVEFOLD_TERMINAL.json"

fold_dir() { printf '%s/runs/candidate_refiner_v2_bbox24_b32_v2det_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_union_transfer_fivefold_20260823.lock"
flock -n 9 || { echo "union transfer five-fold controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  "${python_bin}" - "${terminal}" "${status}" "${line}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
output=Path(sys.argv[1]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","exit_status":int(sys.argv[2]),"failed_line":sys.argv[3],
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${prep_terminal}" || ! -s "${prompt_terminal}" ]]; do
  printf '%s waiting for union index and prompt-pipeline terminal\n' "$(date -Ins)"
  sleep 30
done
prep_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${prep_terminal}")
[[ "${prep_status}" == COMPLETE ]] || { echo "union-native preparation failed" >&2; exit 2; }
index=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["index"])' "${prep_terminal}")
[[ -s "${index}" ]]

gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  printf '%s waiting for both GPUs: gpu0=%s MiB gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

result_matches() {
  local result="$1" prediction="$2" checkpoint="$3"
  "${python_bin}" - "${result}" "${prediction}" "${checkpoint}" "${index}" <<'PY'
import hashlib,json,sys
from pathlib import Path
result,prediction,checkpoint,index=map(Path,sys.argv[1:])
if not all(path.is_file() for path in (result,prediction,checkpoint,index)): raise SystemExit(1)
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()
p=json.loads(result.read_text(encoding="utf-8"))
valid=(p.get("cross_index_transfer") is True
       and p.get("checkpoint")==str(checkpoint.resolve())
       and p.get("checkpoint_sha256")==sha(checkpoint)
       and p.get("evaluation_index")==str(index.resolve())
       and p.get("evaluation_index_sha256")==sha(index)
       and p.get("predictions")==str(prediction.resolve())
       and p.get("predictions_sha256")==sha(prediction)
       and p.get("public_read") is False and p.get("hidden_read") is False)
raise SystemExit(0 if valid else 1)
PY
}
evaluate_fold() {
  local gpu="$1" fold="$2" family checkpoint_name checkpoint fold_output result predictions
  fold_output="${output}/fold${fold}"; mkdir -p "${fold_output}"
  for spec in \
    'composite checkpoint_best.pt' \
    'segmentation checkpoint_best_segmentation.pt' \
    'classification checkpoint_best_classification.pt'; do
    read -r family checkpoint_name <<< "${spec}"
    checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
    result="${fold_output}/${family}.json"; predictions="${fold_output}/${family}.npz"
    [[ -s "${checkpoint}" ]] || return 2
    result_matches "${result}" "${predictions}" "${checkpoint}" && continue
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/evaluate_candidate_refiner_transfer.py" \
      --index "${index}" --checkpoint "${checkpoint}" --output "${result}" \
      --predictions "${predictions}" --batch-size 8 --workers 2 --allow-index-transfer \
      > "${fold_output}/${family}.log" 2>&1
  done
}
evaluate_sequence() { local gpu="$1"; shift; local fold; for fold in "$@"; do evaluate_fold "${gpu}" "${fold}"; done; }
evaluate_sequence 0 0 2 4 & pid0=$!
evaluate_sequence 1 1 3 & pid1=$!
status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

selection_args=()
for family in composite segmentation classification; do
  case "${family}" in
    composite) checkpoint_name=checkpoint_best.pt;;
    segmentation) checkpoint_name=checkpoint_best_segmentation.pt;;
    classification) checkpoint_name=checkpoint_best_classification.pt;;
  esac
  args=()
  for fold in 0 1 2 3 4; do
    args+=(--checkpoint "$(fold_dir "${fold}")/${checkpoint_name}")
    args+=(--prediction "${output}/fold${fold}/${family}.npz")
    args+=(--transfer-result "${output}/fold${fold}/${family}.json")
  done
  summary="${output}/five_fold_oof_summary_${family}.json"
  "${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" --index "${index}" \
    "${args[@]}" --output "${summary}" > "${output}/summarize_${family}.log" 2>&1
  selection_args+=(--candidate "${family}=${summary}")
done
"${python_bin}" -u "${root}/select_candidate_refiner_oof_strategy.py" \
  "${selection_args[@]}" --output "${selection}" --selected-summary "${selected_oof}" \
  > "${output}/select.log" 2>&1
"${python_bin}" - "${prep_terminal}" "${selection}" "${selected_oof}" "${index}" "${terminal}" <<'PY'
import csv,datetime,json,os,sys
from pathlib import Path
prep,selection,oof,index,output=map(Path,sys.argv[1:]); summary=json.loads(oof.read_text(encoding="utf-8"))
with index.open(newline="",encoding="utf-8") as stream:
    patch_cases=len({row["case"] for row in csv.DictReader(stream)})
full_volume_cases=int(json.loads(prep.read_text(encoding="utf-8"))["cases"])
coverage=summary.get("coverage",{})
if full_volume_cases!=210 or coverage.get("cases")!=patch_cases or not coverage.get("every_patch_exactly_once"):
    raise SystemExit("transfer OOF patch/full-volume coverage failed")
if not all(row.get("cross_index_transfer") is True for row in summary.get("folds",[])):
    raise SystemExit("transfer OOF provenance failed")
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","selection":str(selection.resolve()),"patch_oof":str(oof.resolve()),
 "selection_payload":json.loads(selection.read_text(encoding="utf-8")),
 "patch_oof_payload":summary,"preparation":str(prep.resolve()),
 "full_volume_cases":full_volume_cases,"patch_cases":patch_cases,
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
