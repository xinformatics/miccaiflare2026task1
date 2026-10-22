#!/usr/bin/env bash
set -euo pipefail

# Exact-N case-held-out full-volume OOF for the hybrid small-classifier +
# pretrained-Focus-contour model. Defaults preserve the original 210-case
# experiment; environment overrides support the audited exact-707 path.
# Public/hidden data are never read.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
focus_train="${FOCUS_TRAIN_ROOT:-${root}/analysis/focus_candidate_refiner_fivefold_20260823}"
focus_terminal="${FOCUS_TRAIN_TERMINAL:-${focus_train}/FOCUS_FIVEFOLD_TERMINAL.json}"
focus_patch_oof="${FOCUS_PATCH_OOF:-${focus_train}/focus_patch_oof.json}"
focus_route_summary="${FOCUS_OOF_ROUTE_SUMMARY:-}"
focus_route_decision="${FOCUS_OOF_ROUTE_DECISION:-}"
mixed_route_decision="${FOCUS_OOF_MIXED_ROUTE_DECISION:-}"
classifier_fusion="${FOCUS_OOF_CLASSIFIER_FUSION:-}"
primary_train="${FOCUS_OOF_PRIMARY_TRAIN:-${root}/analysis/candidate_refiner_v2_training_20260822}"
primary_root="${FOCUS_OOF_BASE_ROOT:-${root}/analysis/candidate_refiner_v2_fullvolume_oof_20260823}"
primary_complete="${FOCUS_OOF_BASE_COMPLETE:-${primary_root}/FULL_VOLUME_OOF_COMPLETE.txt}"
primary_calibration="${FOCUS_OOF_BASE_CALIBRATION:-${primary_root}/deployment_calibration.json}"
primary_patch_oof="${FOCUS_OOF_PRIMARY_PATCH_OOF:-${primary_train}/five_fold_oof_summary.json}"
selection="${FOCUS_OOF_SELECTION:-${primary_train}/checkpoint_selection.json}"
classifier_checkpoint_override="${FOCUS_OOF_CLASSIFIER_CHECKPOINT_NAME:-}"
manifest="${FOCUS_OOF_MANIFEST:-${root}/analysis/fpr_training_mining_20260822/manifest.json}"
output="${FOCUS_OOF_OUTPUT:-${root}/analysis/focus_candidate_refiner_fullvolume_oof_20260823}"
calibration="${output}/deployment_calibration.json"
robust="${output}/paired_vs_primary.robust.json"
terminal="${FOCUS_OOF_TERMINAL:-${output}/FOCUS_OOF_TERMINAL.json}"
controller_lock="${FOCUS_OOF_LOCK:-${root}/runs/run_focus_candidate_refiner_fullvolume_oof_20260823.lock}"
candidate_name="${FOCUS_OOF_CANDIDATE_NAME:-focus_hybrid_oof}"
small_run_prefix="${FOCUS_OOF_SMALL_RUN_PREFIX:-${root}/runs/candidate_refiner_v2_bbox24_b32_v2det_fold}"
base_work_name="${FOCUS_OOF_BASE_WORK_NAME:-work_bbox24}"
base_proposal_rel="${FOCUS_OOF_BASE_PROPOSAL_REL:-proposals}"
expected_cases="${FOCUS_OOF_EXPECTED_CASES:-210}"
expected_sources="${FOCUS_OOF_EXPECTED_SOURCES:-epoch105,autopet_weak,lndb_weak}"
storage_lock="${FOCUS_OOF_STORAGE_LOCK:-${root}/runs/focus_fullvolume_storage_reservation_20260824.lock}"
minimum_free_gib="${FOCUS_OOF_MINIMUM_FREE_GIB:-180}"
[[ "${expected_cases}" =~ ^[0-9]+$ && "${expected_cases}" -gt 0 ]] || { echo "invalid expected case count" >&2; exit 2; }
[[ "${minimum_free_gib}" =~ ^[0-9]+$ && "${minimum_free_gib}" -ge 100 \
    && "${minimum_free_gib}" -le 500 ]] \
  || { echo "invalid Focus full-volume free-space floor" >&2; exit 2; }

small_fold_dir() { printf '%s%s_20260823' "${small_run_prefix}" "$1"; }
primary_fold_work() { printf '%s/fold%s/%s' "${primary_root}" "$1" "${base_work_name}"; }
focus_fold_work() { printf '%s/fold%s/work_bbox24' "${output}" "$1"; }

mkdir -p "${output}"
exec 9>"${controller_lock}"
flock -n 9 || { echo "Focus full-volume OOF controller already active" >&2; exit 1; }
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
 "status":"FAILED","decision":"KEEP_PRIMARY","exit_status":int(sys.argv[2]),
 "failed_line":sys.argv[3],"public_read":False,"hidden_read":False}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
  fi
  exit "${status}"
}
trap write_failure EXIT
if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE|SKIPPED) exit 0 ;;
    FAILED)
      mv --no-clobber "${terminal}" \
        "${terminal%.json}.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      ;;
    *) echo "unexpected Focus full-volume terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi
if [[ -n "${mixed_route_decision}" && ( -n "${focus_route_summary}" || -n "${focus_route_decision}" ) ]]; then
  echo "mixed B32/Focus routing and Focus epoch routing are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${mixed_route_decision}" ]]; then
  while [[ ! -s "${mixed_route_decision}" ]]; do
    printf '%s waiting for mixed B32/Focus source-route decision\n' "$(date -Ins)"
    sleep 30
  done
  focus_status=COMPLETE
  focus_decision=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
    "${mixed_route_decision}")
  required_focus_decision=QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME
  focus_gate="${mixed_route_decision}"
elif [[ -n "${focus_route_summary}" || -n "${focus_route_decision}" ]]; then
  [[ -n "${focus_route_summary}" && -n "${focus_route_decision}" ]] || {
    echo "Focus route summary and decision must be supplied together" >&2; exit 2;
  }
  while [[ ! -s "${focus_route_decision}" ]]; do
    printf '%s waiting for Focus epoch-route decision\n' "$(date -Ins)"
    sleep 30
  done
  focus_status=COMPLETE
  focus_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' "${focus_route_decision}")
  required_focus_decision=QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME
  focus_gate="${focus_route_decision}"
else
  while [[ ! -s "${focus_terminal}" ]]; do
    printf '%s waiting for Focus five-fold decision\n' "$(date -Ins)"
    sleep 30
  done
  focus_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${focus_terminal}")
  focus_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' "${focus_terminal}")
  required_focus_decision=PROMOTE_TO_FULL_VOLUME_OOF
  focus_gate="${focus_terminal}"
fi
if [[ "${focus_status}" != COMPLETE || "${focus_decision}" != "${required_focus_decision}" ]]; then
  "${python_bin}" - "${focus_gate}" "${terminal}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
source,output=map(Path,sys.argv[1:])
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_PRIMARY","reason":"Focus five-fold gate did not promote",
 "source":str(source.resolve()),"public_read":False,"hidden_read":False}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
  exit 0
fi
if [[ -n "${classifier_fusion}" ]]; then
  [[ -z "${focus_route_summary}" && -z "${mixed_route_decision}" ]] || {
    echo "classifier fusion requires global, not routed, Focus contours" >&2; exit 2;
  }
  while [[ ! -s "${classifier_fusion}" ]]; do
    printf '%s waiting for exact707 Focus classifier-fusion decision\n' "$(date -Ins)"
    sleep 30
  done
  classifier_fusion_decision=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
    "${classifier_fusion}")
  if [[ "${classifier_fusion_decision}" != QUALIFY_FOCUS_CLASSIFIER_FUSION ]]; then
    "${python_bin}" - "${classifier_fusion}" "${classifier_fusion_decision}" "${terminal}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
source=Path(sys.argv[1]); observed=sys.argv[2]; output=Path(sys.argv[3])
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_PRIMARY",
 "reason":f"Focus classifier-fusion decision was {observed}",
 "source":str(source.resolve()),"public_read":False,"hidden_read":False}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
    exit 0
  fi
fi
while [[ ! -s "${primary_complete}" ]]; do
  printf '%s waiting for primary full-volume OOF\n' "$(date -Ins)"
  sleep 30
done
for required in "${focus_patch_oof}" "${primary_calibration}" "${primary_patch_oof}" \
    "${selection}" "${manifest}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
read -r min_seed_coverage min_refined_overlap min_volume_ratio max_volume_ratio <<< "$("${python_bin}" - "${primary_calibration}" <<'PY'
import json,math,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); g=p.get("geometry_gates",{})
values=[float(g.get(name,default)) for name,default in (
 ("min_seed_coverage",0.0),("min_refined_overlap",0.0),
 ("min_volume_ratio",0.0),("max_volume_ratio",1e12))]
if not all(math.isfinite(value) and value>=0 for value in values):
    raise SystemExit("invalid primary geometry gates")
if values[0]>1 or values[1]>1 or values[2]>values[3]:
    raise SystemExit("invalid primary geometry gate bounds")
print(*(f"{value:.12g}" for value in values))
PY
)"
primary_deployment=$("${python_bin}" - "${primary_calibration}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8")).get("calibration_deployment","standalone")
if value not in {"standalone","anchor_currentgf"}:
    raise SystemExit(f"invalid primary calibration deployment: {value}")
print(value)
PY
)
geometry_gate_json=$("${python_bin}" - "${min_seed_coverage}" \
  "${min_refined_overlap}" "${min_volume_ratio}" "${max_volume_ratio}" <<'PY'
import json,sys
names=("min_seed_coverage","min_refined_overlap","min_volume_ratio","max_volume_ratio")
values=list(map(float,sys.argv[1:]))
gates={name:value for name,value in zip(names,values)}
print(json.dumps(gates,sort_keys=True,separators=(",",":")))
PY
)
[[ -z "${focus_route_summary}" || -s "${focus_route_summary}" ]] || {
  echo "missing Focus route summary: ${focus_route_summary}" >&2; exit 2;
}
[[ -z "${mixed_route_decision}" || -s "${mixed_route_decision}" ]] || {
  echo "missing mixed B32/Focus route decision: ${mixed_route_decision}" >&2; exit 2;
}
checkpoint_name="${classifier_checkpoint_override}"
if [[ -z "${checkpoint_name}" ]]; then
  checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
fi
case "${checkpoint_name}" in
  checkpoint_best.pt|checkpoint_best_classification.pt|checkpoint_best_segmentation.pt|checkpoint_best_cohort_segmentation.pt);;
  *) echo "invalid classifier checkpoint override: ${checkpoint_name}" >&2; exit 2 ;;
esac
focus_routed=0
focus_route_specs=()
focus_checkpoints=()
if [[ -n "${mixed_route_decision}" ]]; then
  focus_routed=2
  mapfile -t focus_route_specs < <("${python_bin}" - "${mixed_route_decision}" "${focus_patch_oof}" <<'PY'
import json,sys
route=json.load(open(sys.argv[1],encoding="utf-8")); focus=json.load(open(sys.argv[2],encoding="utf-8"))
if route.get("decision")!="QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME": raise SystemExit("mixed route did not qualify")
if route.get("index_sha256")!=focus.get("index_sha256"): raise SystemExit("mixed route/Focus index mismatch")
if set(route.get("source_to_route",{}).values())!={"b32","focus"}: raise SystemExit("mixed route must use both model families")
if set(route.get("groups",{}))!={"mature","weak_sources","lung_cells"}: raise SystemExit("mixed route group coverage mismatch")
input_mode=route.get("focus_input_mode")
if input_mode!=focus.get("input_mode") or input_mode not in {"soft","soft_seed","soft_lung_seed"}: raise SystemExit("mixed route Focus input mismatch")
rows=focus.get("fold_records",[])
if sorted(int(row["fold"]) for row in rows)!=list(range(5)): raise SystemExit("invalid mixed-route Focus folds")
print(f"INPUT\t-\t-\t{input_mode}")
for row in sorted(rows,key=lambda value:int(value["fold"])):
    print(f"FOCUS\t{int(row['fold'])}\t-\t{row['checkpoint']}")
for source,family in sorted(route["source_to_route"].items()): print(f"SOURCE\t-\t{source}\t{family}")
PY
  )
  [[ "${#focus_route_specs[@]}" -gt 0 ]] || exit 2
  focus_input_mode=""
  for specification in "${focus_route_specs[@]}"; do
    IFS=$'\t' read -r kind fold_or_dash name value <<< "${specification}"
    if [[ "${kind}" == INPUT ]]; then
      focus_input_mode="${value}"
    elif [[ "${kind}" == FOCUS ]]; then
      [[ -s "${value}" ]] || { echo "missing mixed-route Focus checkpoint: ${value}" >&2; exit 2; }
    elif [[ "${kind}" != SOURCE ]]; then
      echo "invalid mixed route specification: ${specification}" >&2; exit 2
    fi
  done
elif [[ -n "${focus_route_summary}" ]]; then
  focus_routed=1
  mapfile -t focus_route_specs < <("${python_bin}" - "${focus_route_summary}" "${focus_patch_oof}" <<'PY'
import json,sys
route=json.load(open(sys.argv[1],encoding="utf-8")); baseline=json.load(open(sys.argv[2],encoding="utf-8"))
if route.get("index_sha256")!=baseline.get("index_sha256"): raise SystemExit("Focus route/global index mismatch")
rows=route.get("fold_records",[]); groups={"mature","weak_sources","lung_cells"}
if sorted(int(row["fold"]) for row in rows)!=list(range(5)): raise SystemExit("invalid Focus route folds")
source_to_route=route.get("source_to_route",{})
if set(source_to_route.values())!=groups: raise SystemExit("invalid Focus route source mapping")
input_mode=route.get("training_recipe",{}).get("input_mode")
if input_mode not in {"soft","soft_seed","soft_lung_seed"}: raise SystemExit("invalid routed Focus input mode")
print(f"INPUT\t-\t-\t{input_mode}")
for row in sorted(rows,key=lambda value:int(value["fold"])):
    fold=int(row["fold"]); selected=row.get("selected",{})
    if set(selected)!=groups: raise SystemExit(f"fold {fold} route coverage mismatch")
    for group in sorted(groups): print(f"ROUTE\t{fold}\t{group}\t{selected[group]['checkpoint']}")
for source,group in sorted(source_to_route.items()): print(f"SOURCE\t-\t{source}\t{group}")
PY
  )
  [[ "${#focus_route_specs[@]}" -gt 0 ]] || exit 2
  focus_input_mode=""
  for specification in "${focus_route_specs[@]}"; do
    IFS=$'\t' read -r kind fold_or_dash name value <<< "${specification}"
    if [[ "${kind}" == INPUT ]]; then
      focus_input_mode="${value}"
    elif [[ "${kind}" == ROUTE ]]; then
      [[ -s "${value}" ]] || { echo "missing routed Focus checkpoint: ${value}" >&2; exit 2; }
    elif [[ "${kind}" != SOURCE ]]; then
      echo "invalid Focus route specification: ${specification}" >&2; exit 2
    fi
  done
else
  mapfile -t focus_checkpoints < <("${python_bin}" - "${focus_patch_oof}" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))["fold_records"]
if sorted(int(row["fold"]) for row in rows)!=list(range(5)):
    raise SystemExit("invalid Focus fold records")
for row in sorted(rows,key=lambda value:int(value["fold"])):
    print(row["checkpoint"])
PY
  )
  [[ "${#focus_checkpoints[@]}" -eq 5 ]] || exit 2
  for focus in "${focus_checkpoints[@]}"; do [[ -s "${focus}" ]] || exit 2; done
  focus_input_mode=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("input_mode","soft"))' "${focus_patch_oof}")
fi
case "${focus_input_mode}" in soft|soft_seed|soft_lung_seed);; *) echo "invalid Focus input mode" >&2; exit 2;; esac

# Each distinct exact-707 Focus inference cache is approximately 145 GiB on
# this host. Serialize only branches that have already passed their model gate,
# then require a conservative free-space floor before allocating a new cache.
# The descriptor remains open through scoring, so another Focus branch cannot
# race the reservation check and fill the filesystem concurrently.
exec 8>"${storage_lock}"
printf '%s waiting for Focus full-volume storage reservation\n' "$(date -Ins)"
flock 8
required_free_kib=$((minimum_free_gib * 1024 * 1024))
while :; do
  available_kib=$(df -Pk "${output}" | awk 'NR==2 {print $4}')
  [[ "${available_kib}" =~ ^[0-9]+$ ]] \
    || { echo "unable to measure Focus full-volume free space" >&2; exit 2; }
  if [[ "${available_kib}" -ge "${required_free_kib}" ]]; then
    printf '%s acquired Focus storage reservation: available=%s KiB required=%s KiB\n' \
      "$(date -Ins)" "${available_kib}" "${required_free_kib}"
    break
  fi
  printf '%s waiting for Focus storage headroom: available=%s KiB required=%s KiB\n' \
    "$(date -Ins)" "${available_kib}" "${required_free_kib}"
  sleep 60
done

# Reuse the exact prepared OOF patch manifests without altering their caches.
for fold in 0 1 2 3 4; do
  source_work="$(primary_fold_work "${fold}")"
  work="$(focus_fold_work "${fold}")"
  mkdir -p "${work}"
  [[ -s "${source_work}/manifest.json" && -s "${source_work}/PREPARE_COMPLETE.txt" ]] || {
    echo "missing prepared OOF cache marker: ${source_work}" >&2
    exit 2
  }
  cp "${source_work}/manifest.json" "${work}/manifest.json"
  cp "${source_work}/PREPARE_COMPLETE.txt" "${work}/PREPARE_COMPLETE.txt"
done

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  free0=0; free1=0
  [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 1024 ]] && free0=1
  [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 1024 ]] && free1=1
  [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && break
  printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

build_focus_args() {
  local requested_fold="$1" specification kind fold_or_dash name value
  FOCUS_ARGS=()
  FOCUS_VALIDATOR_ARGS=()
  if [[ "${focus_routed}" -eq 2 ]]; then
    local mixed_focus="" mixed_b32
    mixed_b32="$(small_fold_dir "${requested_fold}")/${checkpoint_name}"
    [[ -s "${mixed_b32}" ]] || { echo "missing mixed-route B32 checkpoint: ${mixed_b32}" >&2; return 2; }
    for specification in "${focus_route_specs[@]}"; do
      IFS=$'\t' read -r kind fold_or_dash name value <<< "${specification}"
      if [[ "${kind}" == FOCUS && "${fold_or_dash}" == "${requested_fold}" ]]; then
        mixed_focus="${value}"
      elif [[ "${kind}" == SOURCE ]]; then
        FOCUS_ARGS+=(--source-contour-route "${name}=${value}")
      fi
    done
    [[ -s "${mixed_focus}" ]] || { echo "missing fold ${requested_fold} mixed Focus route" >&2; return 2; }
    FOCUS_ARGS+=(--candidate-contour-route "b32=${mixed_b32}")
    FOCUS_ARGS+=(--focus-contour-route "focus=${mixed_focus}")
    FOCUS_VALIDATOR_ARGS+=(--focus-routed-input-mode "${focus_input_mode}")
  elif [[ "${focus_routed}" -eq 1 ]]; then
    for specification in "${focus_route_specs[@]}"; do
      IFS=$'\t' read -r kind fold_or_dash name value <<< "${specification}"
      if [[ "${kind}" == ROUTE && "${fold_or_dash}" == "${requested_fold}" ]]; then
        FOCUS_ARGS+=(--focus-contour-route "${name}=${value}")
      elif [[ "${kind}" == SOURCE ]]; then
        FOCUS_ARGS+=(--source-contour-route "${name}=${value}")
      fi
    done
    FOCUS_VALIDATOR_ARGS+=(--focus-routed-input-mode "${focus_input_mode}")
  else
    FOCUS_ARGS+=(--contour-checkpoint "${focus_checkpoints[$requested_fold]}")
    FOCUS_VALIDATOR_ARGS+=(--contour-input-mode "${focus_input_mode}")
  fi
  [[ -z "${classifier_fusion}" ]] || FOCUS_ARGS+=(--classifier-fusion "${classifier_fusion}")
}
inference_matches() {
  local work="$1" small="$2"; shift 2
  "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    --marker "${work}/INFERENCE_COMPLETE.json" --checkpoint "${small}" "$@" \
    "${FOCUS_VALIDATOR_ARGS[@]}" \
    > /dev/null 2>&1
}
infer_fold() {
  local gpu="$1" fold="$2" work small
  work="$(focus_fold_work "${fold}")"
  small="$(small_fold_dir "${fold}")/${checkpoint_name}"
  [[ -s "${small}" ]] || return 2
  build_focus_args "${fold}"
  inference_matches "${work}" "${small}" "${FOCUS_ARGS[@]}" && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" --checkpoint "${small}" "${FOCUS_ARGS[@]}" \
    --batch-size 4 --workers 2 --resume > "${output}/fold${fold}.infer.log" 2>&1
  inference_matches "${work}" "${small}" "${FOCUS_ARGS[@]}"
}
infer_sequence() { local gpu="$1"; shift; local fold; for fold in "$@"; do infer_fold "${gpu}" "${fold}"; done; }
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  infer_sequence 0 0 2 4 & pid0=$!
  infer_sequence 1 1 3 & pid1=$!
  status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
elif [[ "${free0}" -eq 1 ]]; then
  infer_sequence 0 0 1 2 3 4
else
  infer_sequence 1 0 1 2 3 4
fi
for fold in 0 1 2 3 4; do
  work="$(focus_fold_work "${fold}")"
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
    --allow-external-patches --require-predictions \
    > "${output}/fold${fold}.audit.log" 2>&1
done

classification_input="${primary_patch_oof}"
[[ -z "${classifier_fusion}" ]] || classification_input="${classifier_fusion}"
read -r classification_center classification_values < <(PYTHONPATH="${root}" "${python_bin}" - "${classification_input}" "${classifier_fusion:+fusion}" <<'PY'
import json,sys
from summarize_candidate_refiner_oof import deployment_classification_thresholds
p=json.load(open(sys.argv[1])); fusion=bool(sys.argv[2])
if fusion:
    center=float(p["candidate"]["threshold"])
    values=deployment_classification_thresholds(center)
else:
    center=float(p["classification"]["threshold"])
    values=p["classification"]["deployment_thresholds"]
if len(values)!=3 or len(set(map(float,values)))!=3: raise SystemExit("expected three classifier thresholds")
print(f'{center:.2f}', ",".join(f"{float(value):.2f}" for value in values))
PY
)
classification_values="${classification_values//,/ }"
source_threshold_json=$("${python_bin}" - "${classification_input}" "${expected_sources}" "${classifier_fusion:+fusion}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
fusion=bool(sys.argv[3])
if fusion:
    grouped=payload.get("by_source",{})
    group_for=lambda name: "mature" if name in {"epoch105","currentgf"} else ("weak_sources" if name in {"autopet_weak","lndb_weak"} else "lung_cells")
    expected={value for value in sys.argv[2].split(",") if value}
    if set(grouped)!={"mature","weak_sources","lung_cells"}: raise SystemExit("fusion source calibration mismatch")
    reports={name:grouped[group_for(name)]["candidate"] for name in expected}
else:
    reports=payload.get("source_classification",{})
if not reports:
    print("{}")
    raise SystemExit(0)
expected={value for value in sys.argv[2].split(",") if value}
if set(reports)!=expected: raise SystemExit(f"source calibration mismatch: {sorted(reports)}")
thresholds={name:round(float(reports[name]["threshold"]),2) for name in sorted(reports)}
if any(not 0 < value < 1 for value in thresholds.values()): raise SystemExit("invalid source threshold")
print(json.dumps(thresholds,sort_keys=True,separators=(",",":")))
PY
)
source_segmentation_input="${focus_patch_oof}"
[[ "${focus_routed}" -eq 0 ]] || source_segmentation_input="${focus_route_summary}"
[[ "${focus_routed}" -ne 2 ]] || source_segmentation_input="${mixed_route_decision}"
source_segmentation_threshold_json=$("${python_bin}" - "${source_segmentation_input}" "${expected_sources}" "${focus_routed}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
route_mode=int(sys.argv[3])
if route_mode==2:
    reports={
        source:{"threshold":value}
        for source,value in payload.get("source_segmentation_thresholds",{}).items()
    }
elif route_mode==1:
    groups=payload.get("source_segmentation_groups",{})
    source_to_route=payload.get("source_to_route",{})
    reports={
        source:{"threshold":groups[route]["threshold"]}
        for source,route in source_to_route.items()
        if route in groups
    }
else:
    reports=payload.get("source_segmentation",{})
if not reports:
    print("{}")
    raise SystemExit(0)
expected={value for value in sys.argv[2].split(",") if value}
if set(reports)!=expected:
    raise SystemExit(f"Focus source contour calibration mismatch: {sorted(reports)}")
thresholds={name:round(float(reports[name]["threshold"]),2) for name in sorted(reports)}
if any(not 0 < value < 1 for value in thresholds.values()):
    raise SystemExit("invalid Focus source contour threshold")
lung={value for name,value in thresholds.items() if name.startswith("lung_cell")}
if lung and len(lung)!=1:
    raise SystemExit(f"pooled lung-cell Focus thresholds diverged: {sorted(lung)}")
print(json.dumps(thresholds,sort_keys=True,separators=(",",":")))
PY
)
mapfile -t source_threshold_entries < <("${python_bin}" - "${source_threshold_json}" <<'PY'
import json,sys
for name,value in sorted(json.loads(sys.argv[1]).items()): print(f"{name}={float(value):.2f}")
PY
)
segmentation_center_input="${focus_patch_oof}"
segmentation_center_key=focus
if [[ "${focus_routed}" -eq 2 ]]; then
  segmentation_center_input="${primary_patch_oof}"
  segmentation_center_key=b32
fi
read -r segmentation_center segmentation_values < <(PYTHONPATH="${root}" "${python_bin}" - "${segmentation_center_input}" "${segmentation_center_key}" <<'PY'
import json,sys
from summarize_candidate_refiner_oof import deployment_segmentation_thresholds
payload=json.load(open(sys.argv[1])); center=(
    float(payload["segmentation_threshold"])
    if sys.argv[2]=="focus" else float(payload["segmentation"]["threshold"])
)
values=deployment_segmentation_thresholds(center)
print(f"{center:.2f}", ",".join(f"{value:.2f}" for value in values))
PY
)
segmentation_values="${segmentation_values//,/ }"

materialize_grid_all_folds() {
  local configurations="$1" segmentation_value="$2" log_tag="$3"
  local fold work base status=0 pid configuration_count fold_group
  local pids=() fold_groups=() empty_arg=()
  [[ "${primary_deployment}" == standalone ]] && empty_arg+=(--empty-base)
  configuration_count=$("${python_bin}" -c \
    'import json,sys;p=json.load(open(sys.argv[1]));print(len(p.get("configurations",p)))' \
    "${configurations}")
  if [[ "${configuration_count}" -le 1 ]]; then
    # Five concurrent real-data materializers triggered systemd-oomd while
    # host swap was saturated. One configuration reduces per-worker memory but
    # does not remove the full CT/probability working set, so retain a bounded
    # three-plus-two schedule.
    fold_groups=("0 1 2" "3 4")
  else
    fold_groups=("0 1" "2 3" "4")
  fi
  for fold_group in "${fold_groups[@]}"; do
    pids=()
    for fold in ${fold_group}; do
      work="$(focus_fold_work "${fold}")"
      if [[ "${primary_deployment}" == anchor_currentgf ]]; then
        base="${primary_root}/staging/fold${fold}/proposals/currentgf"
      else
        base="${primary_root}/staging/fold${fold}/${base_proposal_rel}"
      fi
      "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
        --work "${work}" --base "${base}" "${empty_arg[@]}" --output-root "${output}" \
        --configurations "${configurations}" --fold "${fold}" \
        --segmentation-threshold "${segmentation_value}" \
        --min-seed-coverage "${min_seed_coverage}" \
        --min-refined-overlap "${min_refined_overlap}" \
        --min-volume-ratio "${min_volume_ratio}" \
        --max-volume-ratio "${max_volume_ratio}" \
        > "${output}/${log_tag}.fold${fold}.materialize_grid.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
    [[ "${status}" -eq 0 ]] || return "${status}"
  done
}
score_variant() {
  local classification="$1" segmentation_value="$2" mode="$3" tag="$4" summary="$5"
  local source_cell="$6" source_segmentation_cell="$7"
  local combined_dir metrics score_log dsc nsd combined
  combined_dir="${output}/${tag}/combined"; mkdir -p "${combined_dir}"
  "${python_bin}" - "${output}/${tag}" "${combined_dir}" "${primary_root}/staging/all/labels" "${expected_cases}" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:4]); expected_cases=int(sys.argv[4]); observed={}
for fold in range(5):
    for source in (root/f"fold{fold}").glob("*.nii.gz"):
        if source.name in observed: raise SystemExit(f"duplicate {source.name}")
        target=combined/source.name
        if target.is_symlink() or target.exists():
            if target.resolve(strict=True)!=source.resolve(strict=True):
                raise SystemExit(f"combined OOF collision: {target}")
        else:
            target.symlink_to(source.resolve())
        observed[source.name]=source
expected={path.name for path in labels.glob("*.nii.gz")}
if len(expected)!=expected_cases or set(observed)!=expected: raise SystemExit("OOF coverage mismatch")
PY
  metrics="${output}/${tag}.metrics.csv"; score_log="${output}/${tag}.score.log"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${combined_dir}" --gt "${primary_root}/staging/all/labels" \
    --csv "${metrics}" --workers 4 > "${score_log}" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${classification}" \
    "${segmentation_value}" "${mode}" "${source_cell}" "${source_segmentation_cell}" \
    "${dsc}" "${nsd}" "${combined}" "${metrics}" >> "${summary}"
}

segmentation_sweep="${output}/segmentation_threshold_sweep.tsv"
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tsource_classification_thresholds\tsource_segmentation_thresholds\tDSC\tNSD\tcombined\tmetrics\n' > "${segmentation_sweep}"
for segmentation_value in ${segmentation_values}; do
  tag="focus_segcal_c${classification_center/./p}_s${segmentation_value/./p}_refined"
  grid="${output}/${tag}.grid.json"
  "${python_bin}" - "${tag}" "${classification_center}" "${grid}" <<'PY'
import json,sys
from pathlib import Path
tag=sys.argv[1]; threshold=float(sys.argv[2]); output=Path(sys.argv[3])
output.write_text(json.dumps({"schema_version":1,"configurations":[{
 "tag":tag,"classification_threshold":threshold,"mode":"refined",
 "source_classification_thresholds":{}}]},indent=2)+"\n")
PY
  materialize_grid_all_folds "${grid}" "${segmentation_value}" "${tag}"
  score_variant "${classification_center}" "${segmentation_value}" refined "${tag}" "${segmentation_sweep}" '{}' '{}'
done
segmentation=$("${python_bin}" -u "${root}/select_segmentation_threshold_sweep.py" \
  --summary "${segmentation_sweep}" --patch-oof-center "${segmentation_center}" \
  --expected-cases "${expected_cases}" \
  --output "${output}/SEGMENTATION_THRESHOLD_SELECTION.json")

printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tsource_classification_thresholds\tsource_segmentation_thresholds\tDSC\tNSD\tcombined\tmetrics\n' > "${output}/summary.tsv"
main_grid="${output}/materialization_grid.json"
"${python_bin}" - "${classification_values}" "${classification_center}" "${segmentation}" \
  "${source_threshold_json}" "${source_segmentation_threshold_json}" "${main_grid}" <<'PY'
import json,sys
from pathlib import Path
values=[float(value) for value in sys.argv[1].split()]
center=float(sys.argv[2]); segmentation=float(sys.argv[3])
sources=json.loads(sys.argv[4]); source_segmentation=json.loads(sys.argv[5]); output=Path(sys.argv[6])
configs=[]
for classification in values:
    for mode in ("refined",):
        configs.append({
          "tag":f"focus_c{classification:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p")+f"_{mode}",
          "classification_threshold":classification,"mode":mode,
          "source_classification_thresholds":{},"source_segmentation_thresholds":{},
        })
if sources:
    for mode in ("refined",):
        configs.append({
          "tag":f"focus_src_c{center:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p")+f"_{mode}",
          "classification_threshold":center,"mode":mode,
          "source_classification_thresholds":sources,"source_segmentation_thresholds":{},
        })
if source_segmentation:
    configs.append({
      "tag":f"focus_srcseg_c{center:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p")+"_refined",
      "classification_threshold":center,"mode":"refined",
      "source_classification_thresholds":{},
      "source_segmentation_thresholds":source_segmentation,
    })
if sources and source_segmentation:
    configs.append({
      "tag":f"focus_srcboth_c{center:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p")+"_refined",
      "classification_threshold":center,"mode":"refined",
      "source_classification_thresholds":sources,
      "source_segmentation_thresholds":source_segmentation,
    })
output.write_text(json.dumps({"schema_version":1,"configurations":configs},indent=2)+"\n")
PY
materialize_grid_all_folds "${main_grid}" "${segmentation}" main
for classification in ${classification_values}; do
  for mode in refined; do
    tag="focus_c${classification/./p}_s${segmentation/./p}_${mode}"
    score_variant "${classification}" "${segmentation}" "${mode}" "${tag}" "${output}/summary.tsv" '{}' '{}'
  done
done
if [[ "${source_threshold_json}" != '{}' ]]; then
  for mode in refined; do
    tag="focus_src_c${classification_center/./p}_s${segmentation/./p}_${mode}"
    score_variant "${classification_center}" "${segmentation}" "${mode}" "${tag}" "${output}/summary.tsv" "${source_threshold_json}" '{}'
  done
fi
if [[ "${source_segmentation_threshold_json}" != '{}' ]]; then
  tag="focus_srcseg_c${classification_center/./p}_s${segmentation/./p}_refined"
  score_variant "${classification_center}" "${segmentation}" refined "${tag}" \
    "${output}/summary.tsv" '{}' "${source_segmentation_threshold_json}"
fi
if [[ "${source_threshold_json}" != '{}' && "${source_segmentation_threshold_json}" != '{}' ]]; then
  tag="focus_srcboth_c${classification_center/./p}_s${segmentation/./p}_refined"
  score_variant "${classification_center}" "${segmentation}" refined "${tag}" \
    "${output}/summary.tsv" "${source_threshold_json}" "${source_segmentation_threshold_json}"
fi
{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k7,7gr -k8,8gr; } > "${output}/summary_by_dsc.tsv"
"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${output}/summary_by_dsc.tsv" --checkpoint-selection "${selection}" \
  --manifest "${manifest}" --output "${calibration}" --bootstrap-samples 5000 \
  --seed 20260823 > "${output}/summarize.log" 2>&1
"${python_bin}" - "${calibration}" "${primary_calibration}" "${geometry_gate_json}" \
  "${primary_deployment}" <<'PY'
import json,os,sys
from pathlib import Path
output=Path(sys.argv[1]); source=Path(sys.argv[2]); gates=json.loads(sys.argv[3]); deployment=sys.argv[4]
payload=json.loads(output.read_text(encoding="utf-8"))
payload["geometry_gates"]=gates
payload["calibration_deployment"]=deployment
payload["fixed_geometry_source"]=str(source.resolve())
payload["fixed_deployment_source"]=str(source.resolve())
payload["selection_basis"] += "; geometry gates fixed from primary exact OOF; deployment family fixed from primary exact OOF"
payload["selected"]["geometry_gates"]=gates
payload["selected"]["calibration_deployment"]=deployment
for row in payload["all_variants"]:
    row["geometry_gates"]=gates
    row["calibration_deployment"]=deployment
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
anchor_metrics=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["metrics"])' "${primary_calibration}")
candidate_metrics=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["metrics"])' "${calibration}")
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${candidate_metrics}" \
  --anchor-name primary_refiner_oof --candidate-name "${candidate_name}" \
  --min-dsc-gain 0.015 --min-nsd-gain 0.015 --min-win-rate 0.55 \
  --max-catastrophic-rate 0.10 --bootstrap 10000 --seed 20260823 \
  --output "${robust}" > "${output}/robust.log"
decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
"${python_bin}" - "${calibration}" "${robust}" "${decision}" "${terminal}" "${classifier_fusion}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
calibration,robust=map(Path,sys.argv[1:3]); decision=sys.argv[3]; output=Path(sys.argv[4])
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision,"calibration":str(calibration.resolve()),
 "robust":str(robust.resolve()),
 "classifier_fusion":str(Path(sys.argv[5]).resolve()) if sys.argv[5] else None,
 "public_read":False,"hidden_read":False}
temporary=output.with_name(f".{output.name}.tmp"); temporary.write_text(json.dumps(payload,indent=2)+"\n"); temporary.replace(output)
PY
cat "${output}/summary_by_dsc.tsv"
