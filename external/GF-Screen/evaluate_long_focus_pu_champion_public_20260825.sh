#!/usr/bin/env bash
set -euo pipefail

# One fixed-public deployment screen for the long positive-unlabeled Focus
# continuation.  The checkpoint is evaluated with the exact champion image
# window, 1000 mm3 filter, prompt confirmation, and guided rescue.  A second,
# predeclared conservative deployment imports only candidate components that
# touch the immutable gf_conf anchor.  This script has no hidden-data,
# packaging, or upload path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${LONG_FOCUS_CHECKPOINT:-${root}/runs/gf_focus_pu_long30_e283_to313_20260825/checkpoint_epoch_0313.pt}"
output="${LONG_FOCUS_PUBLIC_OUTPUT:-${root}/public_sweeps/gf_focus_pu_long30_e313_20260825}"
gpu="${LONG_FOCUS_PUBLIC_GPU:-0}"
glance_threshold="${LONG_FOCUS_GLANCE_THRESHOLD:-0.20}"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
gfconf="${root}/public_sweeps/champion_cbg41_component_confirmation_20260822/predictions/cbg_confirm_distance_mm_0_rescue_min_mm3_inf"
touchcomp_metrics="${root}/public_sweeps/gfconf_support_growth_quick_20260824/touching_component.metrics.csv"
inference="${output}/inference"
filtered500="${output}/filtered_mm500"
prompted500="${output}/prompted_mm500"
deployed500="${output}/deployed_mm500_new2k_s25"
filtered1000="${output}/filtered_mm1000"
prompted1000="${output}/prompted_mm1000"
deployed1000="${output}/deployed_mm1000_new2k_s25"
touching="${output}/touching_gfconf_components"
summary="${output}/summary.tsv"
terminal="${output}/TERMINAL.json"
lock="${LONG_FOCUS_PUBLIC_LOCK:-${root}/runs/evaluate_long_focus_pu_champion_public_20260825.lock}"

[[ "${gpu}" =~ ^[01]$ ]] || { echo "LONG_FOCUS_PUBLIC_GPU must be 0 or 1" >&2; exit 2; }
"${python_bin}" - "${glance_threshold}" <<'PY'
import math, sys
value = float(sys.argv[1])
if not math.isfinite(value) or not 0.0 <= value <= 1.0:
    raise SystemExit("LONG_FOCUS_GLANCE_THRESHOLD must be finite and in [0, 1]")
PY

mkdir -p "${output}" "${inference}" "${filtered500}" "${prompted500}" \
  "${deployed500}" "${filtered1000}" "${prompted1000}" \
  "${deployed1000}" "${touching}" "${output}/gates" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "long Focus public evaluation is already active" >&2; exit 1; }

write_terminal() {
  local status="$1" decision="$2" reason="$3" selected="${4:-}"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${selected}" \
    "${glance_threshold}" "${checkpoint}" "${summary}" "${terminal}" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path
status, decision, reason, selected = sys.argv[1:5]
glance_threshold = float(sys.argv[5])
checkpoint, summary, output = map(Path, sys.argv[6:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path):
    return {"path": str(path.resolve()), "sha256": sha(path)} if path.is_file() else None
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "status": status,
    "decision": decision,
    "reason": reason,
    "selected_variant": selected or None,
    "glance_threshold": glance_threshold,
    "checkpoint": asset(checkpoint),
    "summary": asset(summary),
    "fixed_public_DSC_gate": 0.6154316379500749,
    "fixed_public_NSD_floor": 0.5104225107769019,
    "public_read": summary.is_file(),
    "hidden_read": False,
    "submission_created": False,
    "uploaded": False,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
}

if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE) exit 0 ;;
    FAILED)
      preserved_terminal="${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      mv --no-clobber "${terminal}" "${preserved_terminal}"
      printf '%s preserved failed public-evaluation terminal for exact resume: %s\n' \
        "$(date -Ins)" "${preserved_terminal}"
      ;;
    *) echo "unexpected existing public-evaluation terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_TOUCHCOMP \
    "public evaluation failed at line ${line}, status ${status}"
  exit "${status}"
}
trap failure EXIT

for file in "${checkpoint}" "${prompt_config}" "${touchcomp_metrics}"; do
  [[ -s "${file}" ]] || { echo "missing public-evaluation asset: ${file}" >&2; exit 2; }
done
for directory in "${images}" "${labels}" "${prompt_support}" \
    "${guided_candidate}" "${gfconf}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public assets: ${directory}" >&2; exit 2;
  }
done

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
  --test_data_path "${images}" --save_prediction_path "${inference}" \
  --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold "${glance_threshold}" --lesion_threshold 0.50 \
  --focus_batch_size 16 --workers 2 --skip_existing \
  > "${output}/inference.log" 2>&1
[[ $(find "${inference}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

deploy_at_minimum() {
  local minimum="$1" filtered="$2" prompted="$3" deployed="$4"
  if [[ $(find "${filtered}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
    "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
      --pred-a "${inference}" --fusion a --min-mm3 "${minimum}" --workers 4 \
      --output "${filtered}" > "${output}/filter_mm${minimum}.log" 2>&1
  fi
  if [[ $(find "${prompted}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
    "${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
      --base "${filtered}" --support "${prompt_support}" \
      --config "${prompt_config}" --output "${prompted}" \
      > "${output}/prompt_mm${minimum}.log" 2>&1
  fi
  if [[ $(find "${deployed}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
    "${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
      --base "${prompted}" --candidate "${guided_candidate}" \
      --support "${prompt_support}" --output "${deployed}" \
      --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
      --min-support-fraction 0.25 --min-support-mm3 100 \
      --support-dilation-mm inf --workers 4 > "${output}/guided_mm${minimum}.log" 2>&1
  fi
  for directory in "${filtered}" "${prompted}" "${deployed}"; do
    [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  done
}
# The two physical-minimum branches share only immutable inputs and write to
# disjoint directories. Running them serially cost about nine avoidable minutes
# in the checkpoint-293 interim screen. Preserve exact per-branch commands and
# outputs while executing both bounded four-worker pipelines concurrently.
deploy_at_minimum 500 "${filtered500}" "${prompted500}" "${deployed500}" &
pid500=$!
deploy_at_minimum 1000 "${filtered1000}" "${prompted1000}" "${deployed1000}" &
pid1000=$!
deploy_status=0
wait "${pid500}" || deploy_status=$?
wait "${pid1000}" || deploy_status=$?
[[ "${deploy_status}" -eq 0 ]]

touch_count=$(find "${touching}" -maxdepth 1 -name '*.nii.gz' | wc -l)
if [[ "${touch_count}" -eq 0 ]]; then
  "${python_bin}" -u "${root}/screen_anchor_support_contour_growth.py" materialize \
    --anchor "${gfconf}" --support "${deployed1000}" \
    --configuration touching_component --output "${touching}" --workers 4 \
    > "${output}/touching.log" 2>&1
elif [[ "${touch_count}" -ne 50 ]]; then
  echo "incomplete touching-component materialization: ${touch_count}/50" >&2
  exit 3
fi

printf 'variant\tDSC\tNSD\tcombined\tchanged_cases\tgate\tpredictions\tmetrics\trobust\n' > "${summary}"
for spec in deployed500:"${deployed500}" deployed1000:"${deployed1000}" touching:"${touching}"; do
  tag="${spec%%:*}"; prediction="${spec#*:}"
  metrics="${output}/${tag}.metrics.csv"
  score_log="${output}/${tag}.score.log"
  robust="${output}/gates/${tag}.json"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${prediction}" --gt "${labels}" --csv "${metrics}" --workers 4 \
    > "${score_log}" 2>&1
  "${python_bin}" -u "${root}/select_dsc_first_nonregression.py" \
    --anchor "${touchcomp_metrics}" --candidate "${metrics}" \
    --anchor-name gf_touchcomp_0825 --candidate-name "long_pu_${tag}" \
    --min-dsc-gain 0.015 --min-dsc-win-rate 0.55 --min-nsd-gain 0.0 \
    --min-nsd-uncertainty-bound -0.005 --bootstrap 20000 --seed 20260825 \
    --output "${robust}" > "${output}/gates/${tag}.log"
  read -r dsc nsd < <("${python_bin}" - "${metrics}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
print(
    sum(float(row["Lesion_DSC"]) for row in rows) / len(rows),
    sum(float(row["Lesion_NSD"]) for row in rows) / len(rows),
)
PY
  )
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.9f",(d+n)/2}')
  changed=$("${python_bin}" - "${touchcomp_metrics}" "${metrics}" <<'PY'
import csv, sys
left = {r["Name"]: (r["Lesion_DSC"], r["Lesion_NSD"]) for r in csv.DictReader(open(sys.argv[1]))}
right = {r["Name"]: (r["Lesion_DSC"], r["Lesion_NSD"]) for r in csv.DictReader(open(sys.argv[2]))}
print(sum(left[name] != right[name] for name in left))
PY
  )
  gate=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${dsc}" "${nsd}" "${combined}" "${changed}" "${gate}" \
    "${prediction}" "${metrics}" "${robust}" >> "${summary}"
done

selection=$("${python_bin}" - "${summary}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
eligible = [
    row for row in rows
    if row["gate"] == "PROMOTE"
    and float(row["DSC"]) >= 0.6154316379500749
    and float(row["NSD"]) >= 0.5104225107769019
]
eligible.sort(key=lambda row: (float(row["DSC"]), float(row["NSD"])), reverse=True)
print(eligible[0]["variant"] if eligible else "")
PY
)
if [[ -n "${selection}" ]]; then
  write_terminal COMPLETE QUALIFY_SUBMISSION_PREPARATION \
    "passed the fixed-public +0.015 DSC and NSD-nonregression gate" "${selection}"
else
  write_terminal COMPLETE KEEP_TOUCHCOMP \
    "neither predeclared deployment passed the fixed-public gate"
fi
trap - EXIT
cat "${summary}"
