#!/usr/bin/env bash
set -euo pipefail

# Exact-707 full-volume OOF for a five-fold-qualified controlled ablation.
# Prepared case-held-out patches are reused from the primary cache, but every
# probability is inferred by the variant's excluded-fold checkpoint. No
# public/hidden inference, packaging, or upload is reachable from this script.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
source_routed=0
classifier_routed=0
fivefold_terminal_override=""
discovery_run_prefix=""
discovery_run_suffix="20260823"
discovery_checkpoint_name="checkpoint_best_cohort_segmentation.pt"
currentgf_checkpoint_name="checkpoint_best_currentgf_segmentation.pt"
epoch105_checkpoint_name="checkpoint_best_epoch105_segmentation.pt"
autopet_checkpoint_name="checkpoint_best_autopet_segmentation.pt"
lndb_checkpoint_name="checkpoint_best_lndb_segmentation.pt"
lung_checkpoint_name="checkpoint_best_lung_segmentation.pt"
variant="${1:-}"
train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${train}/patches_bbox24_ss_min5_v4_cell48/index.csv"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
primary="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
primary_complete="${primary}/FULL_VOLUME_OOF_COMPLETE.json"
primary_calibration="${primary}/deployment_calibration.json"
primary_save_complete="${root}/public_sweeps/candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
primary_decoupled_chain_complete="${root}/public_sweeps/candidate_refiner_v4_cell707_decoupled_saved_20260824/VARIANT_CHAIN_TERMINAL.json"
staging="${primary}/staging"

case "${variant}" in
  prompt_pool)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_prompt_pool_fivefold_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_promptpool_fold"
    output="${root}/analysis/candidate_refiner_v4_cell707_prompt_pool_fullvolume_oof_20260824"
    gpu="${PROMPT_POOL_OOF_GPU:-1}"
    variant_prefix="promptpool"
    ;;
  primary_decoupled)
    fivefold="${train}/decoupled_oof"
    fivefold_terminal_override="${fivefold}/TERMINAL.json"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_fold"
    output="${root}/analysis/candidate_refiner_v4_cell707_decoupled_fullvolume_oof_20260824"
    gpu="${PRIMARY_DECOUPLED_OOF_GPU:-0}"
    variant_prefix="dec707"
    ;;
  classifier_routed)
    fivefold="${train}/source_classifier_routed_oof"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_fold"
    output="${root}/analysis/candidate_refiner_v4_cell707_classifier_routed_fullvolume_oof_20260824"
    gpu="${CLASSIFIER_ROUTED_OOF_GPU:-1}"
    variant_prefix="cr707"
    classifier_routed=1
    ;;
  source_condition)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_source_conditioning_fivefold_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_srccond_fold"
    output="${root}/analysis/candidate_refiner_v4_cell707_source_conditioning_fullvolume_oof_20260824"
    gpu="${SOURCE_CONDITION_OOF_GPU:-1}"
    variant_prefix="srccond"
    ;;
  small_lesion)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_small_loss_fivefold_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_smallloss30_fold"
    output="${root}/analysis/candidate_refiner_v4_cell707_small_loss_fullvolume_oof_20260824"
    gpu="${SMALL_LESION_OOF_GPU:-0}"
    variant_prefix="small100"
    ;;
  uncertain_negative)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_uncertain_negative_fivefold_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_uneg025_fold"
    output="${root}/analysis/candidate_refiner_v4_cell707_uncertain_negative_fullvolume_oof_20260824"
    gpu="${UNCERTAIN_NEGATIVE_OOF_GPU:-0}"
    variant_prefix="uneg025"
    ;;
  source_routed)
    fivefold="${train}/source_routed_oof"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_fold"
    output="${root}/analysis/candidate_refiner_v4_cell707_source_routed_fullvolume_oof_20260824"
    gpu="${SOURCE_ROUTED_OOF_GPU:-0}"
    variant_prefix="srroute"
    source_routed=1
    ;;
  discovery_specialist)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_discovery_specialist_fivefold_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_fold"
    discovery_run_prefix="candidate_refiner_v4_cell707_b32_t05_discovery_fold"
    discovery_run_suffix="20260824"
    discovery_checkpoint_name="checkpoint_best_routed_segmentation.pt"
    output="${root}/analysis/candidate_refiner_v4_cell707_discovery_specialist_fullvolume_oof_20260824"
    gpu="${DISCOVERY_SPECIALIST_OOF_GPU:-0}"
    variant_prefix="discsp"
    source_routed=1
    ;;
  *) echo "usage: $0 prompt_pool|primary_decoupled|classifier_routed|source_condition|small_lesion|uncertain_negative|source_routed|discovery_specialist" >&2; exit 2;;
esac
[[ "${gpu}" == 0 || "${gpu}" == 1 ]] || { echo "invalid GPU: ${gpu}" >&2; exit 2; }

if [[ "${source_routed}" -eq 1 ]]; then
  fivefold_terminal="${fivefold}/TERMINAL.json"
  selection="${output}/checkpoint_selection.json"
else
  fivefold_terminal="${fivefold}/FIVEFOLD_TERMINAL.json"
  selection="${fivefold}/checkpoint_selection.json"
fi
if [[ -n "${fivefold_terminal_override}" ]]; then
  fivefold_terminal="${fivefold_terminal_override}"
  selection="${output}/checkpoint_selection.json"
fi
if [[ "${classifier_routed}" -eq 1 ]]; then
  fivefold_terminal="${fivefold}/TERMINAL.json"
  selection="${fivefold}/checkpoint_selection.json"
fi
patch_oof="${fivefold}/five_fold_oof_summary.json"
calibration="${output}/deployment_calibration.json"
robust="${output}/paired_vs_primary.robust.json"
terminal="${output}/FULL_VOLUME_OOF_TERMINAL.json"
fold_dir() { printf '%s/runs/%s%s_20260823' "${root}" "${run_prefix}" "$1"; }
discovery_fold_dir() {
  local fold="$1" prefix="${discovery_run_prefix:-${run_prefix}}"
  printf '%s/runs/%s%s_%s' "${root}" "${prefix}" "${fold}" "${discovery_run_suffix}"
}
source_work() { printf '%s/fold%s/work_bbox24' "${primary}" "$1"; }
fold_work() { printf '%s/fold%s/work_bbox24' "${output}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_v4_${variant}_fullvolume_oof_20260824.lock"
flock -n 9 || { echo "${variant} full-volume OOF controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${fivefold_terminal}" ]]; do
  printf '%s waiting for %s five-fold terminal\n' "$(date -Ins)" "${variant}"
  sleep 30
done
fivefold_decision=$("${python_bin}" -c \
  'import json,sys;p=json.load(open(sys.argv[1]));print(p.get("decision_payload",{}).get("decision",p.get("decision","")))' \
  "${fivefold_terminal}")
expected_fivefold_decision="QUALIFY_VARIANT_FULL_VOLUME"
[[ "${source_routed}" -eq 1 ]] && expected_fivefold_decision="QUALIFY_SOURCE_ROUTED_FULL_VOLUME"
[[ "${variant}" == primary_decoupled ]] && expected_fivefold_decision="QUALIFY_DECOUPLED_FULL_VOLUME"
[[ "${classifier_routed}" -eq 1 ]] && expected_fivefold_decision="QUALIFY_CLASSIFIER_ROUTED_FULL_VOLUME"
if [[ "${fivefold_decision}" != "${expected_fivefold_decision}" ]]; then
  "${python_bin}" - "${variant}" "${fivefold_terminal}" "${fivefold_decision}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
variant=sys.argv[1]; source=Path(sys.argv[2]); decision=sys.argv[3]; output=Path(sys.argv[4])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","variant":variant,"decision":"KEEP_PRIMARY",
 "reason":f"five-fold decision was {decision}","fivefold_terminal":str(source.resolve()),
 "public_read":False,"hidden_read":False,"submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY
  exit 0
fi
if [[ "${variant}" == primary_decoupled && ! -s "${selection}" ]]; then
  "${python_bin}" - "${fivefold_terminal}" "${patch_oof}" "${selection}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
terminal,summary,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
terminal_payload=json.loads(terminal.read_text(encoding="utf-8"))
decision=terminal_payload.get("decision_payload",{}).get("decision","")
if decision!="QUALIFY_DECOUPLED_FULL_VOLUME":
    raise SystemExit(f"decoupled patch gate did not qualify: {decision}")
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "selection_basis":"exact-707 grouped five-fold decoupled patch OOF only",
 "selected_strategy":"decoupled",
 "checkpoint_filename":"checkpoint_best_classification.pt",
 "prediction_filename":"validation_predictions_best_classification.npz",
 "contour_checkpoint_filename":"checkpoint_best_segmentation.pt",
 "contour_prediction_filename":"validation_predictions_best_segmentation.npz",
 "classification_eligible":True,
 "fivefold_terminal":str(terminal.resolve()),"fivefold_terminal_sha256":sha(terminal),
 "patch_oof":str(summary.resolve()),"patch_oof_sha256":sha(summary),
 "public_read":False,"hidden_read":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
fi
while [[ ! -s "${primary_complete}" ]]; do
  printf '%s waiting for primary exact-707 full-volume calibration\n' "$(date -Ins)"
  sleep 30
done
if [[ "${source_routed}" -eq 1 && ! -s "${selection}" ]]; then
  discovery_family=primary
  [[ -z "${discovery_run_prefix}" ]] || discovery_family=discovery
  partial_selection=NONE
  [[ "${variant}" == source_routed ]] && \
    partial_selection="${fivefold}/PARTIAL_ROUTE_SELECTION.json"
  "${python_bin}" - "${fivefold_terminal}" "${patch_oof}" \
    "${variant}" "${currentgf_checkpoint_name}" "${epoch105_checkpoint_name}" \
    "${discovery_checkpoint_name}" "${discovery_family}" \
    "${autopet_checkpoint_name}" "${lndb_checkpoint_name}" \
    "${lung_checkpoint_name}" "${partial_selection}" "${selection}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
terminal,summary=map(Path,sys.argv[1:3]); variant=sys.argv[3]
current_name,epoch_name,discovery_name=sys.argv[4:7]
discovery_family=sys.argv[7]
autopet_name,lndb_name,lung_name=sys.argv[8:11]
partial_name=sys.argv[11]; output=Path(sys.argv[12])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
if variant=="source_routed":
    partial=Path(partial_name)
    if not partial.is_file(): raise SystemExit("partial source-route selection is missing")
    selected=json.loads(partial.read_text(encoding="utf-8"))
    terminal_payload=json.loads(terminal.read_text(encoding="utf-8"))
    if selected.get("decision")!="QUALIFY_GROUP_SELECTED_SOURCE_ROUTE":
        raise SystemExit("partial source-route selector did not qualify")
    if terminal_payload.get("partial_route_selection_sha256")!=sha(partial):
        raise SystemExit("five-fold terminal/partial selection digest mismatch")
    contour_filenames=selected.get("contour_checkpoint_filenames",{})
    contour_families=selected.get("contour_checkpoint_run_families",{})
    source_routes=selected.get("source_contour_routes",{})
    if (not contour_filenames or set(contour_families)!=set(contour_filenames)
            or set(contour_families.values())!={"primary"}):
        raise SystemExit("invalid partial route checkpoint inventory")
    expected={"epoch105","currentgf","autopet_weak","lndb_weak"}|{f"lung_cell{i}" for i in range(8)}
    if set(source_routes)!=expected or set(source_routes.values())!=set(contour_filenames):
        raise SystemExit("invalid partial route source inventory")
    partial_provenance={
      "path":str(partial.resolve()),"sha256":sha(partial),
      "selected_units":selected.get("selected_units",[]),
      "fallback_units":selected.get("fallback_units",[]),
    }
else:
    contour_filenames={"currentgf":current_name,"epoch105":epoch_name,
      "discovery":discovery_name}
    contour_families={"currentgf":"primary","epoch105":"primary",
      "discovery":discovery_family}
    source_routes={"epoch105":"epoch105","currentgf":"currentgf",
      "autopet_weak":"discovery","lndb_weak":"discovery",
      **{f"lung_cell{i}":"discovery" for i in range(8)}}
    partial_provenance=None
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "selection_basis":("exact-707 five-fold independently selected source-route patch OOF"
   if variant=="source_routed" else
   "predeclared exact-707 five-fold source-routed patch OOF only"),
 "selected_strategy":"source_routed",
 "classification_checkpoint_filename":"checkpoint_best_classification.pt",
 "contour_checkpoint_filenames":contour_filenames,
 "contour_checkpoint_run_families":contour_families,
 "source_contour_routes":source_routes,
 "fivefold_terminal":str(terminal.resolve()),"fivefold_terminal_sha256":sha(terminal),
 "patch_oof":str(summary.resolve()),"patch_oof_sha256":sha(summary),
 "partial_route_selection":partial_provenance,
 "public_read":False,"hidden_read":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY
fi
for required in "${selection}" "${patch_oof}" "${index}" "${manifest}" \
  "${primary_calibration}" "${staging}/STAGING_COMPLETE.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 3; }
done
read -r min_seed_coverage min_refined_overlap min_volume_ratio max_volume_ratio <<< "$("${python_bin}" - "${primary_calibration}" <<'PY'
import json,math,sys
g=json.load(open(sys.argv[1],encoding="utf-8")).get("geometry_gates",{})
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
print(json.dumps(dict(zip(names,map(float,sys.argv[1:]))),sort_keys=True,separators=(",",":")))
PY
)
"${python_bin}" - "${patch_oof}" "${selection}" "${index}" "${staging}/STAGING_COMPLETE.json" <<'PY'
import csv,json,sys
patch=json.load(open(sys.argv[1])); selection=json.load(open(sys.argv[2]))
with open(sys.argv[3],newline="",encoding="utf-8") as stream:
    cases={row["case"] for row in csv.DictReader(stream)}
stage=json.load(open(sys.argv[4]))
if len(cases)!=707 or patch.get("coverage",{}).get("cases")!=707 or not patch.get("coverage",{}).get("every_patch_exactly_once"):
    raise SystemExit("variant patch OOF lacks exact 707-case coverage")
if selection.get("selected_strategy") not in {"composite","segmentation","cohort_segmentation","classification","decoupled","source_routed","classifier_routed"}:
    raise SystemExit("invalid variant checkpoint selection")
if stage.get("cases")!=707 or not stage.get("exact_proposal_coverage"):
    raise SystemExit("primary deployment staging is incomplete")
PY

for fold in 0 1 2 3 4; do
  source="$(source_work "${fold}")"; work="$(fold_work "${fold}")"
  [[ -s "${source}/manifest.json" && -s "${source}/PREPARE_COMPLETE.txt" ]] || {
    echo "missing primary prepared cache: ${source}" >&2; exit 3;
  }
  mkdir -p "${work}"
  cp "${source}/manifest.json" "${work}/manifest.json"
  cp "${source}/PREPARE_COMPLETE.txt" "${work}/PREPARE_COMPLETE.txt"
done

checkpoint_name=""
contour_checkpoint_name="NONE"
if [[ "${source_routed}" -eq 0 ]]; then
  checkpoint_name=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
  contour_checkpoint_name=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1])).get("contour_checkpoint_filename","NONE"))' \
    "${selection}")
else
  route_specs_text=$("${python_bin}" "${root}/candidate_refiner_route_selection.py" \
    --selection "${selection}") || {
      echo "invalid source-routed selection: ${selection}" >&2; exit 3;
    }
  mapfile -t route_specs <<< "${route_specs_text}"
fi
# Preserve the shortest path to tonight's first qualified ZIP.  The
# source-routed arm reuses the primary model and is an incremental follow-up;
# it must not occupy a GPU while primary fixed-public and hidden save-only
# inference are still pending.  Newly trained variants already inherit an
# equivalent prerequisite from their five-fold controllers.
if [[ "${variant}" == source_routed ]]; then
  while [[ ! -s "${primary_save_complete}" \
      || ! -s "${primary_decoupled_chain_complete}" ]]; do
    printf '%s waiting for primary save and direct decoupled chain\n' \
      "$(date -Ins)"
    sleep 30
  done
elif [[ "${variant}" == primary_decoupled ]]; then
  while [[ ! -s "${primary_save_complete}" ]]; do
    printf '%s waiting for primary exact-707 save-or-skip terminal\n' "$(date -Ins)"
    sleep 30
  done
elif [[ "${classifier_routed}" -eq 1 ]]; then
  while [[ ! -s "${primary_save_complete}" \
      || ! -s "${primary_decoupled_chain_complete}" ]]; do
    printf '%s waiting for primary save and direct decoupled chain before classifier routing\n' \
      "$(date -Ins)"
    sleep 30
  done
fi
while :; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done
inference_matches() {
  local work="$1" checkpoint="$2" contour_checkpoint="$3"
  local marker_args=(--marker "${work}/INFERENCE_COMPLETE.json" --checkpoint "${checkpoint}")
  [[ "${contour_checkpoint}" == NONE ]] || marker_args+=(--contour-checkpoint "${contour_checkpoint}")
  "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    "${marker_args[@]}" > /dev/null 2>&1
}
for fold in 0 1 2 3 4; do
  work="$(fold_work "${fold}")"
  if [[ "${source_routed}" -eq 1 ]]; then
    IFS=$'\t' read -r kind class_name <<< "${route_specs[0]}"
    [[ "${kind}" == CLASS && -n "${class_name}" ]] || { echo "invalid routed classifier specification" >&2; exit 3; }
    classification="$(fold_dir "${fold}")/${class_name}"
    [[ -s "${classification}" ]] || { echo "missing routed checkpoint: ${classification}" >&2; exit 3; }
    routed_args=()
    for specification in "${route_specs[@]:1}"; do
      IFS=$'\t' read -r kind name value family <<< "${specification}"
      if [[ "${kind}" == ROUTE ]]; then
        if [[ "${family}" == discovery ]]; then
          route_checkpoint="$(discovery_fold_dir "${fold}")/${value}"
        else
          route_checkpoint="$(fold_dir "${fold}")/${value}"
        fi
        [[ -s "${route_checkpoint}" ]] || { echo "missing routed checkpoint: ${route_checkpoint}" >&2; exit 3; }
        routed_args+=(--candidate-contour-route "${name}=${route_checkpoint}")
      elif [[ "${kind}" == SOURCE ]]; then
        routed_args+=(--source-contour-route "${name}=${value}")
      else
        echo "invalid routed specification: ${specification}" >&2; exit 3
      fi
    done
    if ! "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
      --marker "${work}/INFERENCE_COMPLETE.json" --checkpoint "${classification}" \
      "${routed_args[@]}" > /dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
        --work "${work}" --checkpoint "${classification}" \
        "${routed_args[@]}" --batch-size 8 --workers 2 \
        > "${output}/fold${fold}.infer.log" 2>&1
    fi
  else
    checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
    [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 3; }
    contour_checkpoint=NONE; contour_args=()
    if [[ "${contour_checkpoint_name}" != NONE ]]; then
      contour_checkpoint="$(fold_dir "${fold}")/${contour_checkpoint_name}"
      [[ -s "${contour_checkpoint}" ]] || { echo "missing contour checkpoint: ${contour_checkpoint}" >&2; exit 3; }
      contour_args+=(--contour-checkpoint "${contour_checkpoint}")
    fi
    classifier_args=()
    if [[ "${classifier_routed}" -eq 1 ]]; then
      mapfile -t classifier_specs < <("${python_bin}" \
        "${root}/candidate_refiner_classifier_route_selection.py" \
        --decision "${fivefold}/DECISION.json" --fold "${fold}")
      for specification in "${classifier_specs[@]}"; do
        IFS=$'\t' read -r kind route_fold name value <<< "${specification}"
        if [[ "${kind}" == ROUTE ]]; then
          [[ "${route_fold}" == "${fold}" && -s "${value}" ]] || {
            echo "invalid classifier route: ${specification}" >&2; exit 3;
          }
          classifier_args+=(--classifier-route "${name}=${value}")
        elif [[ "${kind}" == SOURCE ]]; then
          classifier_args+=(--source-classifier-route "${name}=${value}")
        else
          echo "invalid classifier route specification: ${specification}" >&2; exit 3
        fi
      done
      if ! "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
        --marker "${work}/INFERENCE_COMPLETE.json" --checkpoint "${checkpoint}" \
        "${contour_args[@]}" "${classifier_args[@]}" > /dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
          --work "${work}" --checkpoint "${checkpoint}" "${contour_args[@]}" \
          "${classifier_args[@]}" --batch-size 32 --workers 2 --resume \
          > "${output}/fold${fold}.infer.log" 2>&1
      fi
      "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
        --marker "${work}/INFERENCE_COMPLETE.json" --checkpoint "${checkpoint}" \
        "${contour_args[@]}" "${classifier_args[@]}" > /dev/null
    elif ! inference_matches "${work}" "${checkpoint}" "${contour_checkpoint}"; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
        --work "${work}" --checkpoint "${checkpoint}" "${contour_args[@]}" \
        --batch-size 8 --workers 2 \
        > "${output}/fold${fold}.infer.log" 2>&1
    fi
  fi
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
    --allow-external-patches --require-predictions --require-inference-plan \
    > "${output}/fold${fold}.audit.log" 2>&1
done

read -r classification_center classification_values < <("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); values=p["classification"]["deployment_thresholds"]
if len(values)!=3 or len(set(map(float,values)))!=3: raise SystemExit("expected three classifier thresholds")
print(f'{float(p["classification"]["threshold"]):.2f}', ",".join(f"{float(v):.2f}" for v in values))
PY
)
classification_values="${classification_values//,/ }"
read -r segmentation_center segmentation_values < <(PYTHONPATH="${root}" "${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
from summarize_candidate_refiner_oof import deployment_segmentation_thresholds
center=float(json.load(open(sys.argv[1]))["segmentation"]["threshold"])
print(f"{center:.2f}", ",".join(f"{v:.2f}" for v in deployment_segmentation_thresholds(center)))
PY
)
segmentation_values="${segmentation_values//,/ }"
source_threshold_json=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); reports=p.get("source_classification",{})
expected={"epoch105","currentgf","autopet_weak","lndb_weak"}|{f"lung_cell{i}" for i in range(8)}
if set(reports)!=expected: raise SystemExit("source threshold inventory mismatch")
print(json.dumps({name:round(float(reports[name]["threshold"]),2) for name in sorted(reports)},sort_keys=True,separators=(",",":")))
PY
)
source_segmentation_threshold_json=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); reports=p.get("source_segmentation",{})
expected={"epoch105","currentgf","autopet_weak","lndb_weak"}|{f"lung_cell{i}" for i in range(8)}
if set(reports)!=expected: raise SystemExit("source contour threshold inventory mismatch")
values={name:round(float(reports[name]["threshold"]),2) for name in sorted(reports)}
if any(not 0 < value < 1 for value in values.values()): raise SystemExit("invalid source contour threshold")
if len({values[f"lung_cell{i}"] for i in range(8)})!=1: raise SystemExit("pooled cell contour thresholds diverged")
print(json.dumps(values,sort_keys=True,separators=(",",":")))
PY
)
lower_cells() {
  "${python_bin}" - "$1" "$2" <<'PY'
import json,sys
p=json.loads(sys.argv[1]); delta=float(sys.argv[2]); cells=[f"lung_cell{i}" for i in range(8)]
values={float(p[name]) for name in cells}
if len(values)!=1: raise SystemExit("pooled lung-cell thresholds diverged")
value=max(0.01,round(next(iter(values))-delta,2))
for name in cells:p[name]=value
print(json.dumps(p,sort_keys=True,separators=(",",":")))
PY
}
source_recall_threshold_json=$(lower_cells "${source_threshold_json}" 0.10)
source_high_recall_threshold_json=$(lower_cells "${source_threshold_json}" 0.20)

materialize_grid_all_folds() {
  local configurations="$1" segmentation="$2" log_tag="$3" fold group pid status=0 work base
  local -a empty_arg=()
  [[ "${primary_deployment}" == standalone ]] && empty_arg+=(--empty-base)
  for group in "0 1" "2 3" "4"; do
    pids=()
    for fold in ${group}; do
      work="$(fold_work "${fold}")"
      if [[ "${primary_deployment}" == anchor_currentgf ]]; then
        base="${staging}/fold${fold}/proposals/currentgf"
      else
        base="${staging}/fold${fold}/proposals/epoch105"
      fi
      "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
        --work "${work}" --base "${base}" "${empty_arg[@]}" --output-root "${output}" \
        --configurations "${configurations}" --fold "${fold}" \
        --segmentation-threshold "${segmentation}" \
        --min-seed-coverage "${min_seed_coverage}" \
        --min-refined-overlap "${min_refined_overlap}" \
        --min-volume-ratio "${min_volume_ratio}" \
        --max-volume-ratio "${max_volume_ratio}" \
        > "${output}/${log_tag}.fold${fold}.materialize.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
    [[ "${status}" -eq 0 ]] || return "${status}"
  done
}
score_variant() {
  local tag="$1" classification="$2" segmentation="$3" source_json="$4"
  local source_segmentation_json="$5" source_mode_json="$6" summary="$7"
  local combined="${output}/${tag}/combined" metrics="${output}/${tag}.metrics.csv"
  mkdir -p "${combined}"
  "${python_bin}" - "${output}/${tag}" "${combined}" "${staging}/all/labels" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:]); seen={}
for fold in range(5):
    for source in (root/f"fold{fold}").glob("*.nii.gz"):
        if source.name in seen: raise SystemExit(f"duplicate: {source.name}")
        target=combined/source.name
        if not target.exists(): target.symlink_to(source.resolve(strict=True))
        elif target.resolve(strict=True)!=source.resolve(strict=True): raise SystemExit(f"collision: {target}")
        seen[source.name]=source
expected={p.name for p in labels.glob("*.nii.gz")}
if len(expected)!=707 or set(seen)!=expected: raise SystemExit("OOF coverage mismatch")
PY
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${combined}" --gt "${staging}/all/labels" --csv "${metrics}" --workers 6 \
    > "${output}/${tag}.score.log" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${output}/${tag}.score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${output}/${tag}.score.log")
  combined_score=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t%s\trefined\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${classification}" \
    "${segmentation}" "${source_json}" "${source_segmentation_json}" "${source_mode_json}" \
    "${dsc}" "${nsd}" "${combined_score}" "${metrics}" >> "${summary}"
}

header='variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tsource_classification_thresholds\tsource_segmentation_thresholds\tsource_deployment_modes\tDSC\tNSD\tcombined\tmetrics'
segmentation_sweep="${output}/segmentation_threshold_sweep.tsv"; printf '%b\n' "${header}" > "${segmentation_sweep}"
for segmentation_candidate in ${segmentation_values}; do
  tag="${variant_prefix}_segcal_c${classification_center/./p}_s${segmentation_candidate/./p}_refined"
  grid="${output}/${tag}.grid.json"
  "${python_bin}" - "${tag}" "${classification_center}" "${grid}" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[3]).write_text(json.dumps({"schema_version":1,"configurations":[{
 "tag":sys.argv[1],"classification_threshold":float(sys.argv[2]),"mode":"refined",
 "source_classification_thresholds":{},"source_segmentation_thresholds":{}}]},indent=2)+"\n")
PY
  materialize_grid_all_folds "${grid}" "${segmentation_candidate}" "${tag}"
  score_variant "${tag}" "${classification_center}" "${segmentation_candidate}" '{}' '{}' '{}' "${segmentation_sweep}"
done
segmentation=$("${python_bin}" -u "${root}/select_segmentation_threshold_sweep.py" \
  --summary "${segmentation_sweep}" --patch-oof-center "${segmentation_center}" \
  --expected-cases 707 --output "${output}/SEGMENTATION_THRESHOLD_SELECTION.json")

summary_tsv="${output}/summary.tsv"; printf '%b\n' "${header}" > "${summary_tsv}"
grid="${output}/materialization_grid.json"
"${python_bin}" - "${classification_values}" "${classification_center}" "${segmentation}" \
  "${source_threshold_json}" "${source_recall_threshold_json}" \
  "${source_high_recall_threshold_json}" "${source_segmentation_threshold_json}" \
  "${variant_prefix}" "${grid}" <<'PY'
import json,sys
from pathlib import Path
values=[float(v) for v in sys.argv[1].split()]; center=float(sys.argv[2]); seg=float(sys.argv[3])
source=[json.loads(v) for v in sys.argv[4:7]]; source_seg=json.loads(sys.argv[7]); prefix=sys.argv[8]; output=Path(sys.argv[9]); configs=[]
for value in values:
    tag=(f"{prefix}_c{value:.2f}_s{seg:.2f}_refined").replace(".","p")
    configs.append({"tag":tag,"classification_threshold":value,"mode":"refined","source_classification_thresholds":{},"source_segmentation_thresholds":{}})
for name,thresholds in zip(("src","srclo","srclo2"),source):
    tag=(f"{prefix}_{name}_c{center:.2f}_s{seg:.2f}_refined").replace(".","p")
    configs.append({"tag":tag,"classification_threshold":center,"mode":"refined","source_classification_thresholds":thresholds,"source_segmentation_thresholds":{}})
tag=(f"{prefix}_srcseg_c{center:.2f}_s{seg:.2f}_refined").replace(".","p")
configs.append({"tag":tag,"classification_threshold":center,"mode":"refined","source_classification_thresholds":source[0],"source_segmentation_thresholds":source_seg})
for suffix,modes in (
    ("seedgf",{"epoch105":"seed","currentgf":"seed"}),
    ("seedcropgf",{"epoch105":"seed_crop","currentgf":"seed_crop"}),
):
    tag=(f"{prefix}_{suffix}_c{center:.2f}_s{seg:.2f}").replace(".","p")
    configs.append({"tag":tag,"classification_threshold":center,"mode":"refined",
      "source_classification_thresholds":source[0],"source_segmentation_thresholds":source_seg,
      "source_deployment_modes":modes})
output.write_text(json.dumps({"schema_version":1,"configurations":configs},indent=2)+"\n")
PY
materialize_grid_all_folds "${grid}" "${segmentation}" main
for classification in ${classification_values}; do
  tag="${variant_prefix}_c${classification/./p}_s${segmentation/./p}_refined"
  score_variant "${tag}" "${classification}" "${segmentation}" '{}' '{}' '{}' "${summary_tsv}"
done
for name in src srclo srclo2; do
  case "${name}" in src) source_json="${source_threshold_json}";; srclo) source_json="${source_recall_threshold_json}";; *) source_json="${source_high_recall_threshold_json}";; esac
  tag="${variant_prefix}_${name}_c${classification_center/./p}_s${segmentation/./p}_refined"
  score_variant "${tag}" "${classification_center}" "${segmentation}" "${source_json}" '{}' '{}' "${summary_tsv}"
done
tag="${variant_prefix}_srcseg_c${classification_center/./p}_s${segmentation/./p}_refined"
score_variant "${tag}" "${classification_center}" "${segmentation}" "${source_threshold_json}" \
  "${source_segmentation_threshold_json}" '{}' "${summary_tsv}"
for mode_variant in seedgf seedcropgf; do
  tag="${variant_prefix}_${mode_variant}_c${classification_center/./p}_s${segmentation/./p}"
  if [[ "${mode_variant}" == seedgf ]]; then
    source_mode_json='{"currentgf":"seed","epoch105":"seed"}'
  else
    source_mode_json='{"currentgf":"seed_crop","epoch105":"seed_crop"}'
  fi
  score_variant "${tag}" "${classification_center}" "${segmentation}" "${source_threshold_json}" \
    "${source_segmentation_threshold_json}" "${source_mode_json}" "${summary_tsv}"
done
{ head -1 "${summary_tsv}"; tail -n +2 "${summary_tsv}" | sort -t $'\t' -k8,8gr -k9,9gr; } > "${output}/summary_by_dsc.tsv"
"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${output}/summary_by_dsc.tsv" --checkpoint-selection "${selection}" \
  --manifest "${manifest}" --output "${calibration}" --bootstrap-samples 5000 --seed 20260824 \
  > "${output}/summarize.log" 2>&1
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
  --anchor-name primary_exact707 --candidate-name "${variant}" \
  --min-dsc-gain 0.005 --min-nsd-gain 0.005 --min-win-rate 0.52 \
  --max-catastrophic-rate 0.10 --bootstrap 20000 --seed 20260824 --output "${robust}" \
  > "${output}/robust.log" 2>&1
robust_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
"${python_bin}" - "${variant}" "${fivefold_terminal}" "${calibration}" "${robust}" \
  "${robust_decision}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
variant=sys.argv[1]; fivefold,calibration,robust=map(Path,sys.argv[2:5]); decision=sys.argv[5]; output=Path(sys.argv[6])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","variant":variant,
 "decision":"QUALIFY_FOR_FIXED_PUBLIC" if decision=="PROMOTE" else "KEEP_PRIMARY",
 "selection_basis":"exact-707 case-held-out grouped-fold full-volume OOF on identical prepared proposals",
 "fivefold_terminal":str(fivefold.resolve()),"calibration":str(calibration.resolve()),
 "calibration_payload":json.loads(calibration.read_text()),"paired_comparison":str(robust.resolve()),
 "paired_payload":json.loads(robust.read_text()),"public_read":False,"hidden_read":False,
 "submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY
