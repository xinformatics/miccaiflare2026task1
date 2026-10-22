#!/usr/bin/env bash
set -euo pipefail

# Fixed public qualification for the OOF-selected small-classifier + Focus-
# contour hybrid. No public threshold/mode search and no hidden inference.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
focus_oof="${FOCUS_PUBLIC_OOF_ROOT:-${root}/analysis/focus_candidate_refiner_fullvolume_oof_20260823}"
oof_terminal="${FOCUS_PUBLIC_OOF_TERMINAL:-${focus_oof}/FOCUS_OOF_TERMINAL.json}"
calibration="${FOCUS_PUBLIC_CALIBRATION:-${focus_oof}/deployment_calibration.json}"
focus_patch_oof="${FOCUS_PUBLIC_PATCH_OOF:-${root}/analysis/focus_candidate_refiner_fivefold_20260823/focus_patch_oof.json}"
selection="${root}/analysis/candidate_refiner_v2_training_20260822/checkpoint_selection.json"
primary_public="${root}/public_sweeps/candidate_refiner_v2_bbox24_20260822"
primary_work="${FOCUS_PUBLIC_PROPOSAL_WORK:-${primary_public}/work_all_bbox24}"
proposal_terminal="${FOCUS_PUBLIC_PROPOSAL_TERMINAL:-}"
public_labels="${project}/validation/Validation-Public-Labels"
public_base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
output="${FOCUS_PUBLIC_OUTPUT:-${root}/public_sweeps/focus_candidate_refiner_20260823}"
work="${output}/work_all_bbox24"
terminal="${FOCUS_PUBLIC_TERMINAL:-${output}/FOCUS_PUBLIC_TERMINAL.json}"
controller_lock="${FOCUS_PUBLIC_LOCK:-${root}/runs/run_focus_candidate_refiner_public_20260823.lock}"
variant_prefix="${FOCUS_PUBLIC_VARIANT_PREFIX:-focus_stand}"

small_fold_dir() { printf '%s/runs/candidate_refiner_v2_bbox24_b32_v2det_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${output}" "${work}"
exec 9>"${controller_lock}"
flock -n 9 || { echo "Focus public controller already active" >&2; exit 1; }
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  if [[ ! -s "${terminal}" ]]; then
    "${python_bin}" - "${terminal}" "${status}" "${line}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
output=Path(sys.argv[1]); output.parent.mkdir(parents=True,exist_ok=True)
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"FAILED","decision":"KEEP_ANCHOR","exit_status":int(sys.argv[2]),
 "failed_line":sys.argv[3],"public_read":False,"hidden_read":False}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
  fi
  exit "${status}"
}
trap write_failure EXIT
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${oof_terminal}" ]]; do
  printf '%s waiting for Focus full-volume OOF decision\n' "$(date -Ins)"
  sleep 30
done
oof_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${oof_terminal}")
if [[ "${oof_decision}" != PROMOTE ]]; then
  "${python_bin}" - "${oof_terminal}" "${terminal}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
source,output=map(Path,sys.argv[1:])
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_PRIMARY","reason":"Focus hybrid did not pass exact-210 OOF gate",
 "source":str(source.resolve()),"public_read":False,"hidden_read":False}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
  exit 0
fi
if [[ -n "${proposal_terminal}" ]]; then
  while [[ ! -s "${proposal_terminal}" ]]; do
    printf '%s waiting for proposal-domain public preparation\n' "$(date -Ins)"
    sleep 30
  done
fi
for required in "${calibration}" "${focus_patch_oof}" "${selection}" "${primary_work}/manifest.json" \
    "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
[[ $(find "${public_labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${public_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
cp "${primary_work}/manifest.json" "${work}/manifest.json"
checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
mapfile -t selected_focus_paths < <("${python_bin}" - "${focus_patch_oof}" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))["fold_records"]
if sorted(int(row["fold"]) for row in rows)!=list(range(5)):
    raise SystemExit("invalid Focus fold records")
for row in sorted(rows,key=lambda value:int(value["fold"])):
    print(row["checkpoint"])
PY
)
[[ "${#selected_focus_paths[@]}" -eq 5 ]] || exit 2
focus_input_mode=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("input_mode","soft"))' "${focus_patch_oof}")
case "${focus_input_mode}" in soft|soft_seed|soft_lung_seed);; *) echo "invalid Focus input mode" >&2; exit 2;; esac
small_args=(); focus_args=(); small_paths=(); focus_paths=()
for fold in 0 1 2 3 4; do
  small="$(small_fold_dir "${fold}")/${checkpoint_name}"
  focus="${selected_focus_paths[$fold]}"
  for required in "${small}" "${focus}"; do [[ -s "${required}" ]] || exit 2; done
  small_args+=(--checkpoint "${small}"); focus_args+=(--contour-checkpoint "${focus}")
  small_paths+=("${small}"); focus_paths+=("${focus}")
done
gpu_memory_mib() { nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
while :; do
  used=$(gpu_memory_mib)
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] && break
  printf '%s waiting for GPU 0: %s MiB\n' "$(date -Ins)" "${used}"; sleep 30
done
inference_matches() {
  "${python_bin}" - "${work}/INFERENCE_COMPLETE.json" "${focus_input_mode}" "${#small_paths[@]}" "${small_paths[@]}" "${focus_paths[@]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
marker=Path(sys.argv[1]); input_mode=sys.argv[2]; count=int(sys.argv[3]); values=list(map(Path,sys.argv[4:]));
if not marker.is_file(): raise SystemExit(1)
payload=json.loads(marker.read_text()); small=values[:count]; focus=values[count:]
def sha(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): value.update(block)
    return value.hexdigest()
valid=(payload.get("checkpoints")==[str(p.resolve()) for p in small]
       and payload.get("checkpoint_sha256")==[sha(p) for p in small]
       and payload.get("contour_checkpoints")==[str(p.resolve()) for p in focus]
       and payload.get("contour_checkpoint_sha256")==[sha(p) for p in focus]
       and payload.get("contour_input_modes")==[input_mode]*len(focus))
raise SystemExit(0 if valid else 1)
PY
}
if ! inference_matches; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" "${small_args[@]}" "${focus_args[@]}" --batch-size 2 --workers 2 \
    > "${output}/infer_public.log" 2>&1
fi
inference_matches
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${work}" --output "${work}/AUDIT.json" --workers 4 \
  --require-predictions > "${output}/audit_inference.log" 2>&1
read -r classification segmentation mode < <("${python_bin}" - "${calibration}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(f'{float(p["classification_threshold"]):.2f} {float(p["segmentation_threshold"]):.2f} {p["deployment_mode"]}')
PY
)
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tdeployment\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' > "${output}/summary.tsv"

score_variant() {
  local deployment="$1" tag="$2" prediction="$3"
  local metrics="${output}/${tag}.metrics.csv" score_log="${output}/${tag}.score.log"
  local robust="${output}/${tag}.robust.json" dsc nsd combined decision
  [[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${prediction}" \
    --gt "${public_labels}" --csv "${metrics}" --workers 4 > "${score_log}" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" --anchor "${anchor_metrics}" \
    --candidate "${metrics}" --anchor-name gf_champ_new2k_s25 --candidate-name "${tag}" \
    --output "${robust}" > "${output}/${tag}.robust.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${classification}" "${segmentation}" "${mode}" "${deployment}" \
    "${dsc}" "${nsd}" "${combined}" "${metrics}" "${robust}" "${decision}" \
    >> "${output}/summary.tsv"
}

for deployment in anchor standalone; do
  tag="${variant_prefix}_${deployment}_${mode}_c${classification/./p}_s${segmentation/./p}"
  prediction="${output}/${tag}"
  empty_arg=()
  [[ "${deployment}" == standalone ]] && empty_arg+=(--empty-base)
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
    --work "${work}" --base "${public_base}" --output "${prediction}" \
    "${empty_arg[@]}" --classification-threshold "${classification}" \
    --segmentation-threshold "${segmentation}" --mode "${mode}" \
    --min-seed-coverage 0 --min-refined-overlap 0 \
    --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
    > "${output}/${tag}.materialize.log" 2>&1
  score_variant "${deployment}" "${tag}" "${prediction}"
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k6,6gr -k7,7gr
} > "${output}/summary_by_dsc.tsv"
selected=$("${python_bin}" "${root}/select_promoted_refiner_summary.py" --summary "${output}/summary.tsv")
decision=KEEP_ANCHOR
[[ -n "${selected}" ]] && decision=PROMOTE
"${python_bin}" - "${calibration}" "${output}/summary.tsv" "${decision}" "${terminal}" <<'PY'
import csv,json,sys
from datetime import datetime
from pathlib import Path
calibration,summary=map(Path,sys.argv[1:3]); decision=sys.argv[3]; output=Path(sys.argv[4])
with summary.open(newline="",encoding="utf-8") as stream:
    rows=list(csv.DictReader(stream,delimiter="\t"))
promoted=[row for row in rows if row.get("decision")=="PROMOTE"]
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision,"calibration":str(calibration.resolve()),
 "summary":str(summary.resolve()),"promoted_variants":[row["variant"] for row in promoted],
 "public_read":True,"hidden_read":False}
temporary=output.with_name(f".{output.name}.tmp"); temporary.write_text(json.dumps(payload,indent=2)+"\n"); temporary.replace(output)
PY
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"
