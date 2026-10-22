#!/usr/bin/env bash
set -euo pipefail

# Exact707 local-only full-volume OOF for the composite deployment:
# B32 on B32-qualified sources, per-fold/per-source Focus epochs elsewhere,
# and x/y TTA only on groups that pass the pooled five-fold gate. Thresholds
# and geometry are inherited from held-out patch OOF and primary full-volume
# OOF respectively. No public/hidden data, ZIP creation, or upload is present.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
composite_root="${root}/analysis/focus_composite_routes_20260824"
composite="${composite_root}/DECISION.json"
composite_terminal="${composite_root}/TERMINAL.json"
primary_train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
primary_root="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
primary_complete="${primary_root}/FULL_VOLUME_OOF_COMPLETE.json"
primary_calibration="${primary_root}/deployment_calibration.json"
primary_patch_oof="${primary_train}/five_fold_oof_summary.json"
selection="${primary_train}/checkpoint_selection.json"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
output="${root}/analysis/focus_composite_fullvolume_oof_20260824"
terminal="${output}/TERMINAL.json"
storage_lock="${root}/runs/focus_fullvolume_storage_reservation_20260824.lock"
controller_lock="${root}/runs/run_focus_composite_fullvolume_oof_20260824.lock"
expected_cases=707
expected_sources="epoch105,currentgf,autopet_weak,lndb_weak,lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7"

small_fold_dir() {
  printf '%s/runs/candidate_refiner_v4_cell707_b32_t05_fold%s_20260823' \
    "${root}" "$1"
}
primary_fold_work() { printf '%s/fold%s/work_bbox24' "${primary_root}" "$1"; }
composite_fold_work() { printf '%s/fold%s/work_bbox24' "${output}" "$1"; }

mkdir -p "${output}" "${output}/checkpoint_aliases" "${output}/matplotlib" "${root}/runs"
exec 9>"${controller_lock}"
flock -n 9 || { echo "composite full-volume OOF already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${terminal}" "${status}" "${decision}" "${reason}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
output=Path(sys.argv[1]); payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":sys.argv[2],"decision":sys.argv[3],"reason":sys.argv[4],
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_PRIMARY \
    "failed at line ${line}, status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${composite_terminal}" || ! -s "${primary_complete}" ]]; do
  printf '%s waiting for composite route and primary exact OOF\n' "$(date -Ins)"
  sleep 30
done
composite_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
  "${composite_terminal}")
composite_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
  "${composite_terminal}")
if [[ "${composite_status}" != COMPLETE || \
      "${composite_decision}" != QUALIFY_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME ]]; then
  write_terminal SKIPPED KEEP_PRIMARY \
    "composite architecture/epoch/TTA route did not qualify"
  trap - EXIT
  exit 0
fi
for required in "${composite}" "${primary_calibration}" "${primary_patch_oof}" \
    "${selection}" "${manifest}"; do
  [[ -s "${required}" ]] || { echo "missing composite OOF input: ${required}" >&2; exit 2; }
done

checkpoint_name=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' \
  "${selection}")
case "${checkpoint_name}" in
  checkpoint_best.pt|checkpoint_best_classification.pt|checkpoint_best_segmentation.pt|checkpoint_best_cohort_segmentation.pt) ;;
  *) echo "invalid primary checkpoint selection: ${checkpoint_name}" >&2; exit 2 ;;
esac

read -r min_seed_coverage min_refined_overlap min_volume_ratio max_volume_ratio primary_deployment <<< "$(
  "${python_bin}" - "${primary_calibration}" <<'PY'
import json,math,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); g=p.get("geometry_gates",{})
values=[float(g.get(name,default)) for name,default in (
 ("min_seed_coverage",0.0),("min_refined_overlap",0.0),
 ("min_volume_ratio",0.0),("max_volume_ratio",1e12))]
deployment=p.get("calibration_deployment","standalone")
if deployment not in {"standalone","anchor_currentgf"}: raise SystemExit("invalid deployment")
if not all(math.isfinite(value) and value>=0 for value in values): raise SystemExit("invalid geometry")
print(*(f"{value:.12g}" for value in values),deployment)
PY
)"
read -r classification_center classification_threshold_json <<< "$(
  "${python_bin}" - "${primary_patch_oof}" "${expected_sources}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); expected=set(sys.argv[2].split(","))
center=float(p["classification"]["threshold"]); reports=p.get("source_classification",{})
if set(reports)!=expected: raise SystemExit("primary source classification mismatch")
thresholds={name:round(float(reports[name]["threshold"]),2) for name in sorted(reports)}
if any(not 0<value<1 for value in thresholds.values()): raise SystemExit("invalid classification threshold")
print(f"{center:.2f}",json.dumps(thresholds,sort_keys=True,separators=(",",":")))
PY
)"
segmentation_threshold_json=$("${python_bin}" - "${composite}" "${expected_sources}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); expected=set(sys.argv[2].split(","))
values=p.get("source_segmentation_thresholds",{})
if set(values)!=expected: raise SystemExit("composite source segmentation mismatch")
thresholds={name:round(float(values[name]),2) for name in sorted(values)}
if any(not 0<value<1 for value in thresholds.values()): raise SystemExit("invalid segmentation threshold")
print(json.dumps(thresholds,sort_keys=True,separators=(",",":")))
PY
)

# Serialize dense probability caches and reserve enough room for the one
# approximately 145-GiB branch plus downstream masks.
exec 8>"${storage_lock}"
printf '%s waiting for Focus full-volume storage reservation\n' "$(date -Ins)"
flock 8
required_free_kib=$((180 * 1024 * 1024))
while :; do
  available_kib=$(df -Pk "${output}" | awk 'NR==2 {print $4}')
  [[ "${available_kib}" =~ ^[0-9]+$ ]] || exit 2
  [[ "${available_kib}" -ge "${required_free_kib}" ]] && break
  printf '%s waiting for composite storage headroom: %s KiB\n' \
    "$(date -Ins)" "${available_kib}"
  sleep 60
done

for fold in 0 1 2 3 4; do
  source_work="$(primary_fold_work "${fold}")"
  work="$(composite_fold_work "${fold}")"
  mkdir -p "${work}"
  [[ -s "${source_work}/manifest.json" && -s "${source_work}/PREPARE_COMPLETE.txt" ]] || exit 2
  cp "${source_work}/manifest.json" "${work}/manifest.json"
  cp "${source_work}/PREPARE_COMPLETE.txt" "${work}/PREPARE_COMPLETE.txt"
done

build_route_args() {
  local fold="$1" small specification kind name value expected_hash alias
  small="$(small_fold_dir "${fold}")/${checkpoint_name}"
  [[ -s "${small}" ]] || return 2
  ROUTE_ARGS=(--candidate-contour-route "b32=${small}")
  TTA_MARKER_ARGS=()
  VALIDATOR_MODE_ARGS=()
  mapfile -t specifications < <("${python_bin}" - "${composite}" "${fold}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); fold=int(sys.argv[2])
rows={int(row["fold"]):row for row in p["fold_records"]}; row=rows[fold]
for name,route in sorted(row["routes"].items()):
 if name!="b32": print("FOCUS\t%s\t%s\t%s"%(name,route["checkpoint"],route["checkpoint_sha256"]))
for source,route in sorted(p["source_to_route"].items()): print(f"SOURCE\t{source}\t{route}\t-")
print(f"INPUT\t-\t{p['focus_input_mode']}\t-")
PY
  )
  for specification in "${specifications[@]}"; do
    IFS=$'\t' read -r kind name value expected_hash <<< "${specification}"
    if [[ "${kind}" == FOCUS ]]; then
      [[ -s "${value}" ]] || return 2
      checkpoint="${value}"
      if [[ "${name}" == *_tta_xy ]]; then
        mkdir -p "${output}/checkpoint_aliases/fold${fold}"
        alias="${output}/checkpoint_aliases/fold${fold}/${name}.pt"
        if [[ ! -e "${alias}" ]]; then ln "${value}" "${alias}"; fi
        observed=$(sha256sum "${alias}" | awk '{print $1}')
        [[ "${observed}" == "${expected_hash}" ]] || return 2
        checkpoint="${alias}"
        TTA_MARKER_ARGS+=(--tta-route "${name}")
      fi
      ROUTE_ARGS+=(--focus-contour-route "${name}=${checkpoint}")
    elif [[ "${kind}" == SOURCE ]]; then
      ROUTE_ARGS+=(--source-contour-route "${name}=${value}")
    elif [[ "${kind}" == INPUT ]]; then
      VALIDATOR_MODE_ARGS+=(--focus-routed-input-mode "${value}")
    else
      return 2
    fi
  done
  [[ "${#TTA_MARKER_ARGS[@]}" -gt 0 ]] || return 2
}

inference_matches() {
  local work="$1" small="$2"; shift 2
  "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    --marker "${work}/INFERENCE_COMPLETE.json" --checkpoint "${small}" "$@" \
    "${VALIDATOR_MODE_ARGS[@]}" \
    > /dev/null 2>&1 || return 1
  "${python_bin}" "${root}/validate_focus_tta_xy_marker.py" \
    --work "${work}" "${TTA_MARKER_ARGS[@]}" > /dev/null 2>&1
}
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
wait_for_gpu() {
  local gpu="$1" used
  while :; do
    used=$(gpu_memory_mib "${gpu}")
    [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] && return 0
    printf '%s composite OOF waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
    sleep 30
  done
}
infer_fold() {
  local gpu="$1" fold="$2" work small
  work="$(composite_fold_work "${fold}")"
  small="$(small_fold_dir "${fold}")/${checkpoint_name}"
  build_route_args "${fold}"
  inference_matches "${work}" "${small}" "${ROUTE_ARGS[@]}" && return 0
  wait_for_gpu "${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
    "${python_bin}" -u "${root}/apply_candidate_refiner_focus_tta_xy.py" infer \
    --work "${work}" --checkpoint "${small}" "${ROUTE_ARGS[@]}" \
    --batch-size 4 --workers 2 --resume > "${output}/fold${fold}.infer.log" 2>&1
  inference_matches "${work}" "${small}" "${ROUTE_ARGS[@]}"
}
infer_sequence() { local gpu="$1"; shift; local fold; for fold in "$@"; do infer_fold "${gpu}" "${fold}"; done; }
infer_sequence 0 0 2 4 & pid0=$!
infer_sequence 1 1 3 & pid1=$!
status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

for fold in 0 1 2 3 4; do
  work="$(composite_fold_work "${fold}")"
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
    --allow-external-patches --require-predictions \
    > "${output}/fold${fold}.audit.log" 2>&1
done

grid="${output}/materialization_grid.json"
"${python_bin}" - "${classification_center}" "${classification_threshold_json}" \
  "${segmentation_threshold_json}" "${grid}" <<'PY'
import json,sys
from pathlib import Path
center=float(sys.argv[1]); classification=json.loads(sys.argv[2]); segmentation=json.loads(sys.argv[3]); output=Path(sys.argv[4])
payload={"schema_version":1,"configurations":[{
 "tag":"composite","classification_threshold":center,"mode":"refined",
 "source_classification_thresholds":classification,
 "source_segmentation_thresholds":segmentation,
}]}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
empty_arg=(); [[ "${primary_deployment}" == standalone ]] && empty_arg+=(--empty-base)
for fold_group in "0 1 2" "3 4"; do
  pids=(); status=0
  for fold in ${fold_group}; do
    work="$(composite_fold_work "${fold}")"
    if [[ "${primary_deployment}" == anchor_currentgf ]]; then
      base="${primary_root}/staging/fold${fold}/proposals/currentgf"
    else
      base="${primary_root}/staging/fold${fold}/proposals/epoch105"
    fi
    "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
      --work "${work}" --base "${base}" "${empty_arg[@]}" --output-root "${output}" \
      --configurations "${grid}" --fold "${fold}" --segmentation-threshold 0.5 \
      --min-seed-coverage "${min_seed_coverage}" \
      --min-refined-overlap "${min_refined_overlap}" \
      --min-volume-ratio "${min_volume_ratio}" --max-volume-ratio "${max_volume_ratio}" \
      > "${output}/fold${fold}.materialize.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]] || exit "${status}"
done

combined="${output}/composite/combined"; mkdir -p "${combined}"
"${python_bin}" - "${output}/composite" "${combined}" \
  "${primary_root}/staging/all/labels" "${expected_cases}" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:4]); expected=int(sys.argv[4]); observed={}
for fold in range(5):
 for source in (root/f"fold{fold}").glob("*.nii.gz"):
  if source.name in observed: raise SystemExit(f"duplicate {source.name}")
  target=combined/source.name
  if not target.exists(): target.symlink_to(source.resolve())
  observed[source.name]=source
truth={path.name for path in labels.glob("*.nii.gz")}
if len(truth)!=expected or set(observed)!=truth: raise SystemExit("composite OOF coverage mismatch")
PY
metrics="${output}/composite.metrics.csv"
"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${combined}" --gt "${primary_root}/staging/all/labels" \
  --csv "${metrics}" --workers 4 > "${output}/score.log" 2>&1
anchor_metrics=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["metrics"])' \
  "${primary_calibration}")
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${metrics}" \
  --anchor-name primary_refiner_oof --candidate-name composite_focus_b32_tta_oof \
  --min-dsc-gain 0.005 --min-nsd-gain 0.0 --min-win-rate 0.52 \
  --max-catastrophic-rate 0.10 --bootstrap 10000 --seed 20260824 \
  --output "${output}/paired_vs_primary.robust.json" > "${output}/robust.log"
decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' \
  "${output}/paired_vs_primary.robust.json")
write_terminal COMPLETE "${decision}" \
  "exact707 local full-volume composite route evaluation complete"
trap - EXIT
cat "${output}/score.log"
