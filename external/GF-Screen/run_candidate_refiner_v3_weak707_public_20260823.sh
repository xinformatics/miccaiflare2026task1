#!/usr/bin/env bash
set -euo pipefail

# Fixed public evaluation of the exact-707 source-separated refiner. OOF
# qualification, checkpoint family, thresholds, output mode, prompts, NMS, and
# existence threshold are all frozen before public labels are read. No hidden,
# packaging, submission, or upload path exists.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
qualification="${WEAK707_PUBLIC_QUALIFICATION:-${root}/analysis/candidate_refiner_v4_cell707_qualification_20260823/QUALIFICATION.json}"
required_decision="${WEAK707_PUBLIC_REQUIRED_DECISION:-QUALIFY_FOR_FIXED_PUBLIC}"
scaled_oof="${WEAK707_PUBLIC_OOF_ROOT:-${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823}"
calibration="${WEAK707_PUBLIC_CALIBRATION:-${scaled_oof}/deployment_calibration.json}"
contour_summary="${WEAK707_PUBLIC_CONTOUR_SUMMARY:-}"
classifier_fusion="${WEAK707_PUBLIC_CLASSIFIER_FUSION:-}"
classifier_route_decision="${WEAK707_PUBLIC_CLASSIFIER_ROUTE_DECISION:-}"
focus_route_summary="${WEAK707_PUBLIC_FOCUS_ROUTE_SUMMARY:-}"
focus_route_decision="${WEAK707_PUBLIC_FOCUS_ROUTE_DECISION:-}"
mixed_focus_summary="${WEAK707_PUBLIC_MIXED_FOCUS_SUMMARY:-}"
mixed_route_decision="${WEAK707_PUBLIC_MIXED_ROUTE_DECISION:-}"
composite_route_decision="${WEAK707_PUBLIC_COMPOSITE_ROUTE_DECISION:-}"
margin_decision="${root}/analysis/weak707_bbox_margin_headroom_20260823/DECISION.json"
train="${WEAK707_PUBLIC_TRAIN_ROOT:-${root}/analysis/candidate_refiner_v4_cell707_training_20260823}"
selection="${WEAK707_PUBLIC_SELECTION:-${train}/checkpoint_selection.json}"
classifier_checkpoint_override="${WEAK707_PUBLIC_CLASSIFIER_CHECKPOINT_NAME:-}"
plan="${biomed}/analysis/weak_domain_biomedparse_707_selected_20260823/SELECTION_PLAN.json"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
inputs="${biomed}/public_a160_240_inputs"
epoch105="${root}/public_sweeps/epoch0105/abdomen_g09_l05/predictions"
currentgf="${root}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
base="${WEAK707_PUBLIC_BASE:-${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25}"
allow_prune_base="${WEAK707_PUBLIC_ALLOW_PRUNE_BASE:-0}"
source_routed="${WEAK707_PUBLIC_SOURCE_ROUTED:-0}"
anchor_metrics="${WEAK707_PUBLIC_ANCHOR_METRICS:-${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv}"
cell_scale="${root}/analysis/lung_cell707_scale_20260823"
cell_decision="${cell_scale}/DECISION.json"
output="${WEAK707_PUBLIC_OUTPUT:-${root}/public_sweeps/candidate_refiner_v4_cell707_20260823}"
work="${output}/work_bbox24_ss_v4_cell48"
proposal_cache="${WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT:-${output}}"
public_manifest="${proposal_cache}/public_manifest.json"
public_cells="${proposal_cache}/cells48"
complete="${output}/PUBLIC_ONLY_COMPLETE.txt"
candidate_prefix="${WEAK707_PUBLIC_TAG_PREFIX:-c707}"
run_prefix="${WEAK707_PUBLIC_RUN_PREFIX:-candidate_refiner_v4_cell707_b32_t05_fold}"
discovery_run_prefix="${WEAK707_PUBLIC_DISCOVERY_RUN_PREFIX:-${run_prefix}}"
discovery_run_suffix="${WEAK707_PUBLIC_DISCOVERY_RUN_SUFFIX:-20260823}"
controller_lock="${WEAK707_PUBLIC_LOCK:-${root}/runs/run_candidate_refiner_v3_weak707_public_20260823.lock}"

[[ "${allow_prune_base}" == 0 || "${allow_prune_base}" == 1 ]] || {
  echo "WEAK707_PUBLIC_ALLOW_PRUNE_BASE must be 0 or 1" >&2; exit 2;
}
[[ "${source_routed}" == 0 || "${source_routed}" == 1 ]] || {
  echo "WEAK707_PUBLIC_SOURCE_ROUTED must be 0 or 1" >&2; exit 2;
}
[[ -z "${focus_route_summary}" || -n "${focus_route_decision}" ]] || {
  echo "routed Focus summary requires WEAK707_PUBLIC_FOCUS_ROUTE_DECISION" >&2; exit 2;
}
[[ -z "${focus_route_decision}" || -n "${focus_route_summary}" ]] || {
  echo "routed Focus decision requires WEAK707_PUBLIC_FOCUS_ROUTE_SUMMARY" >&2; exit 2;
}
[[ -z "${mixed_focus_summary}" || -n "${mixed_route_decision}" ]] || {
  echo "mixed Focus summary requires WEAK707_PUBLIC_MIXED_ROUTE_DECISION" >&2; exit 2;
}
[[ -z "${mixed_route_decision}" || -n "${mixed_focus_summary}" ]] || {
  echo "mixed route decision requires WEAK707_PUBLIC_MIXED_FOCUS_SUMMARY" >&2; exit 2;
}
[[ -z "${composite_route_decision}" || -s "${composite_route_decision}" ]] || {
  echo "missing composite route decision: ${composite_route_decision}" >&2; exit 2;
}
[[ -z "${contour_summary}" || -z "${focus_route_summary}" ]] || {
  echo "global and routed Focus contours are mutually exclusive" >&2; exit 2;
}
[[ "${source_routed}" -eq 0 || -z "${focus_route_summary}" ]] || {
  echo "candidate-refiner and Focus contour routing are mutually exclusive" >&2; exit 2;
}
[[ -z "${classifier_fusion}" || -n "${contour_summary}" ]] || {
  echo "classifier fusion requires a global Focus contour summary" >&2; exit 2;
}
[[ -z "${classifier_fusion}" || -z "${focus_route_summary}" ]] || {
  echo "classifier fusion and routed Focus contours are mutually exclusive" >&2; exit 2;
}
[[ -z "${classifier_fusion}" || -z "${classifier_route_decision}" ]] || {
  echo "classifier fusion and routed classifiers are mutually exclusive" >&2; exit 2;
}
[[ -z "${mixed_route_decision}" || ( -z "${contour_summary}" && -z "${focus_route_summary}" \
    && "${source_routed}" -eq 0 && -z "${classifier_fusion}" \
    && -z "${classifier_checkpoint_override}" ) ]] || {
  echo "mixed B32/Focus routing is exclusive with other contour/classifier overrides" >&2; exit 2;
}
[[ -z "${composite_route_decision}" || ( -z "${contour_summary}" && -z "${focus_route_summary}" \
    && -z "${mixed_route_decision}" && "${source_routed}" -eq 0 \
    && -z "${classifier_fusion}" && -z "${classifier_checkpoint_override}" ) ]] || {
  echo "composite Focus/B32 routing is exclusive with other contour/classifier overrides" >&2; exit 2;
}

fold_dir() { printf '%s/runs/%s%s_20260823' "${root}" "${run_prefix}" "$1"; }
discovery_fold_dir() { printf '%s/runs/%s%s_%s' "${root}" "${discovery_run_prefix}" "$1" "${discovery_run_suffix}"; }

mkdir -p "${output}" "${proposal_cache}"
exec 9>"${controller_lock}"
flock -n 9 || { echo "weak707 public controller already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
while [[ ! -s "${qualification}" ]]; do
  printf '%s waiting for exact-707 OOF qualification\n' "$(date -Ins)"
  sleep 30
done
decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${qualification}")
if [[ "${decision}" != "${required_decision}" ]]; then
  printf '%s %s\n' "$(date -Ins)" "${decision}" > "${output}/OOF_DID_NOT_QUALIFY.txt"
  date -Ins > "${complete}"
  exit 0
fi
for required in "${calibration}" "${selection}" "${plan}" "${anchor_metrics}" \
  "${margin_decision}" "${cell_decision}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
bbox_margin=$("${python_bin}" - "${margin_decision}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
value=int(payload.get("effective_margin_mm", payload["baseline_margin_mm"] if payload["decision"]=="KEEP_MARGIN_24" else payload["selected"]["margin_mm"]))
if value not in {16,24,32,48,72,96}: raise SystemExit("invalid frozen bbox margin")
print(value)
PY
)
[[ -z "${contour_summary}" || -s "${contour_summary}" ]] || { echo "missing contour summary: ${contour_summary}" >&2; exit 2; }
[[ -z "${classifier_fusion}" || -s "${classifier_fusion}" ]] || { echo "missing classifier fusion: ${classifier_fusion}" >&2; exit 2; }
[[ -z "${classifier_route_decision}" || -s "${classifier_route_decision}" ]] || { echo "missing classifier route decision: ${classifier_route_decision}" >&2; exit 2; }
[[ -z "${focus_route_summary}" || -s "${focus_route_summary}" ]] || { echo "missing Focus route summary: ${focus_route_summary}" >&2; exit 2; }
[[ -z "${focus_route_decision}" || -s "${focus_route_decision}" ]] || { echo "missing Focus route decision: ${focus_route_decision}" >&2; exit 2; }
[[ -z "${mixed_focus_summary}" || -s "${mixed_focus_summary}" ]] || { echo "missing mixed Focus summary: ${mixed_focus_summary}" >&2; exit 2; }
[[ -z "${mixed_route_decision}" || -s "${mixed_route_decision}" ]] || { echo "missing mixed route decision: ${mixed_route_decision}" >&2; exit 2; }
for directory in "${images}" "${labels}" "${inputs}" "${epoch105}" "${currentgf}" "${base}"; do
  [[ -d "${directory}" ]] || { echo "missing directory: ${directory}" >&2; exit 2; }
done
[[ $(find "${images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${inputs}" -maxdepth 1 -name '*.npz' | wc -l) -eq 50 ]]
[[ $(find "${epoch105}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${currentgf}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

cell_gate=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${cell_decision}")
[[ "${cell_gate}" == PROMOTE_LUNG_CELLS_707 ]] || {
  echo "exact-707 lung-cell gate did not promote: ${cell_gate}" >&2
  exit 2
}
"${python_bin}" - "${images}" "${public_manifest}" <<'PY'
import json,sys
from pathlib import Path
images,output=map(Path,sys.argv[1:]); items=[]
for image in sorted(images.glob("*_0000.nii.gz")):
    case=image.name[:-12]
    items.append({"name":case,"prediction_name":case+".nii.gz",
                  "image":str(image.resolve()),"complete_label":False})
if len(items)!=50 or len({item["name"] for item in items})!=50:
    raise SystemExit("expected exact 50 public images")
output.write_text(json.dumps({"description":"label-blind public cell manifest",
                              "items":items},indent=2)+"\n",encoding="utf-8")
PY
if [[ ! -s "${public_cells}/PROPOSALS_COMPLETE.json" ]]; then
  "${python_bin}" -u "${root}/generate_lung_cell_proposals.py" \
    --manifest "${public_manifest}" --all-cases --output "${public_cells}" \
    --target-spacing-mm 2.5 --cell-stride-mm 48 --expected-count 50 --workers 2 --resume \
    > "${output}/generate_cells.log" 2>&1
fi

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
gpu0_limit_mib="${WEAK707_PUBLIC_GPU0_MAX_MIB:-1024}"
gpu1_limit_mib="${WEAK707_PUBLIC_GPU1_MAX_MIB:-1024}"
[[ "${gpu0_limit_mib}" =~ ^[0-9]+$ && "${gpu1_limit_mib}" =~ ^[0-9]+$ ]] || {
  echo "invalid weak707 public GPU admission limit" >&2; exit 2;
}
wait_any_gpu() {
  while :; do
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    free0=0; free1=0
    [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le "${gpu0_limit_mib}" ]] && free0=1
    [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le "${gpu1_limit_mib}" ]] && free1=1
    [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && return
    printf '%s waiting for any GPU: gpu0=%s/%s MiB gpu1=%s/%s MiB\n' \
      "$(date -Ins)" "${used0}" "${gpu0_limit_mib}" \
      "${used1}" "${gpu1_limit_mib}"
    sleep 30
  done
}

run_bank() {
  local bank="$1" gpu="$2"
  bash "${biomed}/run_selected_weak_prompt_bank.sh" \
    "${inputs}" "${images}" "${proposal_cache}/${bank}_weak" "${plan}" "${bank}" 50 "${gpu}" \
    > "${output}/${bank}_weak.controller.log" 2>&1
}
wait_any_gpu
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  run_bank autopet 0 & p0=$!; run_bank lndb 1 & p1=$!
  status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
elif [[ "${free0}" -eq 1 ]]; then
  run_bank autopet 0; run_bank lndb 0
else
  run_bank autopet 1; run_bank lndb 1
fi
autopet="${proposal_cache}/autopet_weak/predictions_nifti"
lndb="${proposal_cache}/lndb_weak/predictions_nifti"

if [[ ! -s "${work}/PREPARE_COMPLETE.txt" ]]; then
  cell_public_args=()
  for colour in 0 1 2 3 4 5 6 7; do
    cell_public_args+=(--proposal "lung_cell${colour}=${public_cells}/colour${colour}")
  done
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${images}" --proposal "epoch105=${epoch105}" \
    --proposal "currentgf=${currentgf}" \
    --proposal "autopet_weak=${autopet}" --proposal "lndb_weak=${lndb}" \
    "${cell_public_args[@]}" \
    --component-policy source_separate --base "${base}" --work "${work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 \
    --max-mm3 1000000000000 --bbox-margin-mm "${bbox_margin}" --tile-stride-mm 72 --resume \
    > "${output}/prepare.log" 2>&1
fi
checkpoint_args=(); checkpoint_paths=(); contour_args=(); contour_paths=(); contour_input_mode=""; focus_routed_input_mode=""; infer_batch=8
routed_args=()
if [[ "${source_routed}" -eq 1 ]]; then
  [[ -z "${contour_summary}" ]] || { echo "source routing cannot use a Focus contour summary" >&2; exit 2; }
  route_specs_text=$("${python_bin}" "${root}/candidate_refiner_route_selection.py" \
    --selection "${selection}") || {
      echo "invalid source-routed selection: ${selection}" >&2; exit 2;
    }
  mapfile -t route_specs <<< "${route_specs_text}"
  IFS=$'\t' read -r kind class_name <<< "${route_specs[0]}"
  [[ "${kind}" == CLASS && -n "${class_name}" ]] || { echo "invalid routed classifier specification" >&2; exit 2; }
  for fold in 0 1 2 3 4; do
    classification="$(fold_dir "${fold}")/${class_name}"
    [[ -s "${classification}" ]] || { echo "missing routed checkpoint: ${classification}" >&2; exit 2; }
    checkpoint_args+=(--checkpoint "${classification}"); checkpoint_paths+=("${classification}")
  done
  for specification in "${route_specs[@]:1}"; do
    IFS=$'\t' read -r kind name value family <<< "${specification}"
    if [[ "${kind}" == ROUTE ]]; then
      for fold in 0 1 2 3 4; do
        if [[ "${family}" == discovery ]]; then
          route_checkpoint="$(discovery_fold_dir "${fold}")/${value}"
        else
          route_checkpoint="$(fold_dir "${fold}")/${value}"
        fi
        [[ -s "${route_checkpoint}" ]] || { echo "missing routed checkpoint: ${route_checkpoint}" >&2; exit 2; }
        routed_args+=(--candidate-contour-route "${name}=${route_checkpoint}")
      done
    elif [[ "${kind}" == SOURCE ]]; then
      routed_args+=(--source-contour-route "${name}=${value}")
    else
      echo "invalid routed specification: ${specification}" >&2; exit 2
    fi
  done
else
  checkpoint_name="${classifier_checkpoint_override}"
  if [[ -n "${checkpoint_name}" ]]; then
    [[ "${checkpoint_name}" == checkpoint_best_classification.pt ]] || {
      echo "external classifier override must be checkpoint_best_classification.pt" >&2; exit 2;
    }
    selection_contour_name=NONE
  else
    checkpoint_name=$("${python_bin}" -c \
      'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
    selection_contour_name=$("${python_bin}" -c \
      'import json,sys;v=json.load(open(sys.argv[1])).get("contour_checkpoint_filename");print(v if isinstance(v,str) and v else "NONE")' \
      "${selection}")
  fi
  [[ "${selection_contour_name}" == NONE || -z "${contour_summary}" ]] || {
    echo "selection and external contour summary are mutually exclusive" >&2; exit 2;
  }
  for fold in 0 1 2 3 4; do
    checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
    [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
    checkpoint_args+=(--checkpoint "${checkpoint}"); checkpoint_paths+=("${checkpoint}")
    if [[ "${selection_contour_name}" != NONE ]]; then
      contour="$(fold_dir "${fold}")/${selection_contour_name}"
      [[ -s "${contour}" ]] || { echo "missing contour checkpoint: ${contour}" >&2; exit 2; }
      contour_args+=(--contour-checkpoint "${contour}"); contour_paths+=("${contour}")
    fi
  done
fi
if [[ -n "${mixed_route_decision}" ]]; then
  infer_batch=2
  [[ "${selection_contour_name:-NONE}" == NONE ]] || {
    echo "mixed B32/Focus routing requires the primary selected checkpoint family" >&2; exit 2;
  }
  expected_sources="epoch105,currentgf,autopet_weak,lndb_weak,lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7"
  mapfile -t mixed_route_specs < <("${python_bin}" "${root}/focus_b32_hybrid_route_selection.py" \
    --summary "${mixed_focus_summary}" --decision "${mixed_route_decision}" \
    --expected-sources "${expected_sources}")
  for checkpoint in "${checkpoint_paths[@]}"; do
    routed_args+=(--candidate-contour-route "b32=${checkpoint}")
  done
  for specification in "${mixed_route_specs[@]}"; do
    IFS=$'\t' read -r kind fold_or_dash name value <<< "${specification}"
    if [[ "${kind}" == INPUT ]]; then
      focus_routed_input_mode="${value}"
    elif [[ "${kind}" == FOCUS ]]; then
      [[ -s "${value}" ]] || { echo "missing mixed-route Focus checkpoint: ${value}" >&2; exit 2; }
      routed_args+=(--focus-contour-route "focus=${value}")
    elif [[ "${kind}" == SOURCE ]]; then
      routed_args+=(--source-contour-route "${name}=${value}")
    else
      echo "invalid mixed B32/Focus route record: ${specification}" >&2; exit 2
    fi
  done
  [[ -n "${focus_routed_input_mode}" ]] || { echo "missing mixed Focus input mode" >&2; exit 2; }
fi
if [[ -n "${composite_route_decision}" ]]; then
  infer_batch=2
  [[ "${selection_contour_name:-NONE}" == NONE ]] || {
    echo "composite routing requires the primary selected classifier family" >&2; exit 2;
  }
  expected_sources="epoch105,currentgf,autopet_weak,lndb_weak,lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7"
  for checkpoint in "${checkpoint_paths[@]}"; do
    routed_args+=(--candidate-contour-route "b32=${checkpoint}")
  done
  mapfile -t composite_specs < <("${python_bin}" "${root}/focus_composite_route_selection.py" \
    --decision "${composite_route_decision}" --expected-sources "${expected_sources}")
  for specification in "${composite_specs[@]}"; do
    IFS=$'\t' read -r kind fold_or_dash name value <<< "${specification}"
    if [[ "${kind}" == INPUT ]]; then
      focus_routed_input_mode="${value}"
    elif [[ "${kind}" == FOCUS ]]; then
      [[ "${fold_or_dash}" =~ ^[0-4]$ && -s "${value}" ]] || {
        echo "invalid composite Focus route: ${specification}" >&2; exit 2;
      }
      routed_args+=(--focus-contour-route "${name}=${value}")
    elif [[ "${kind}" == SOURCE ]]; then
      routed_args+=(--source-contour-route "${name}=${value}")
    else
      echo "invalid composite route record: ${specification}" >&2; exit 2
    fi
  done
  [[ -n "${focus_routed_input_mode}" ]] || { echo "missing composite Focus input mode" >&2; exit 2; }
fi
if [[ -n "${contour_summary}" ]]; then
  infer_batch=4
  mapfile -t contour_paths < <("${python_bin}" - "${contour_summary}" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))["fold_records"]
if sorted(int(row["fold"]) for row in rows)!=list(range(5)): raise SystemExit("invalid contour folds")
for row in sorted(rows,key=lambda value:int(value["fold"])): print(row["checkpoint"])
PY
)
  [[ "${#contour_paths[@]}" -eq 5 ]] || exit 2
  contour_input_mode=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["input_mode"])' "${contour_summary}")
  for contour in "${contour_paths[@]}"; do
    [[ -s "${contour}" ]] || { echo "missing contour checkpoint: ${contour}" >&2; exit 2; }
    contour_args+=(--contour-checkpoint "${contour}")
  done
fi
if [[ -n "${focus_route_summary}" ]]; then
  infer_batch=4
  expected_sources="epoch105,currentgf,autopet_weak,lndb_weak,lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7"
  mapfile -t focus_route_specs < <("${python_bin}" "${root}/focus_epoch_route_selection.py" \
    --summary "${focus_route_summary}" --decision "${focus_route_decision}" \
    --expected-sources "${expected_sources}")
  for specification in "${focus_route_specs[@]}"; do
    IFS=$'\t' read -r kind fold_or_dash name value <<< "${specification}"
    if [[ "${kind}" == INPUT ]]; then
      focus_routed_input_mode="${value}"
    elif [[ "${kind}" == ROUTE ]]; then
      routed_args+=(--focus-contour-route "${name}=${value}")
    elif [[ "${kind}" == SOURCE ]]; then
      routed_args+=(--source-contour-route "${name}=${value}")
    else
      echo "invalid Focus route record: ${specification}" >&2; exit 2
    fi
  done
  [[ -n "${focus_routed_input_mode}" ]] || { echo "missing routed Focus input mode" >&2; exit 2; }
fi
if [[ -n "${classifier_route_decision}" ]]; then
  # Batch 32 is already exhaustively validated by every primary B32 fold;
  # route ensembles execute sequentially and do not multiply activations.
  infer_batch=32
  mapfile -t classifier_route_specs < <("${python_bin}" \
    "${root}/candidate_refiner_classifier_route_selection.py" \
    --decision "${classifier_route_decision}")
  for specification in "${classifier_route_specs[@]}"; do
    IFS=$'\t' read -r kind route_fold name value <<< "${specification}"
    if [[ "${kind}" == ROUTE ]]; then
      [[ "${route_fold}" =~ ^[0-4]$ && -s "${value}" ]] || {
        echo "invalid classifier route: ${specification}" >&2; exit 2;
      }
      routed_args+=(--classifier-route "${name}=${value}")
    elif [[ "${kind}" == SOURCE ]]; then
      routed_args+=(--source-classifier-route "${name}=${value}")
    else
      echo "invalid classifier route record: ${specification}" >&2; exit 2
    fi
  done
fi
fusion_args=()
[[ -z "${classifier_fusion}" ]] || fusion_args+=(--classifier-fusion "${classifier_fusion}")
inference_matches() {
  local marker_args=("${checkpoint_args[@]}" "${contour_args[@]}" "${routed_args[@]}")
  [[ -z "${contour_input_mode}" ]] || marker_args+=(--contour-input-mode "${contour_input_mode}")
  [[ -z "${focus_routed_input_mode}" ]] || marker_args+=(--focus-routed-input-mode "${focus_routed_input_mode}")
  [[ -z "${classifier_fusion}" ]] || marker_args+=(--classifier-fusion "${classifier_fusion}")
  "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    --marker "${work}/INFERENCE_COMPLETE.json" "${marker_args[@]}" \
    > /dev/null 2>&1
}
if ! inference_matches; then
  wait_any_gpu
  if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
    if [[ "${used0}" -le "${used1}" ]]; then inference_gpu=0; else inference_gpu=1; fi
  elif [[ "${free0}" -eq 1 ]]; then
    inference_gpu=0
  else
    inference_gpu=1
  fi
  CUDA_VISIBLE_DEVICES="${inference_gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" "${checkpoint_args[@]}" "${contour_args[@]}" "${routed_args[@]}" \
    "${fusion_args[@]}" \
    --batch-size "${infer_batch}" --workers 2 --resume \
    > "${output}/infer.log" 2>&1
fi
inference_matches
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${work}" --output "${work}/AUDIT.json" --workers 4 \
  --require-predictions --require-inference-plan \
  > "${output}/audit_cache.log" 2>&1

classification=$("${python_bin}" -c \
  'import json,sys;print("{:.2f}".format(float(json.load(open(sys.argv[1]))["classification_threshold"])))' "${calibration}")
segmentation=$("${python_bin}" -c \
  'import json,sys;print("{:.2f}".format(float(json.load(open(sys.argv[1]))["segmentation_threshold"])))' "${calibration}")
deployment_mode=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("deployment_mode","refined"))' "${calibration}")
calibration_deployment=$("${python_bin}" - "${calibration}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8")).get("calibration_deployment")
if value not in {"standalone","anchor_currentgf"}:
    raise SystemExit(f"missing or invalid frozen calibration deployment: {value}")
print(value)
PY
)
read -r min_seed_coverage min_refined_overlap min_volume_ratio max_volume_ratio <<< "$("${python_bin}" - "${calibration}" <<'PY'
import json,math,sys
gates=json.load(open(sys.argv[1],encoding="utf-8")).get("geometry_gates",{})
values=[float(gates.get(name,default)) for name,default in (
 ("min_seed_coverage",0.0),("min_refined_overlap",0.0),
 ("min_volume_ratio",0.0),("max_volume_ratio",1e12))]
if not all(math.isfinite(value) for value in values) or min(values)<0:
    raise SystemExit("invalid frozen geometry gates")
if values[0]>1 or values[1]>1 or values[2]>values[3]:
    raise SystemExit("invalid frozen geometry gate bounds")
print(*(f"{value:.12g}" for value in values))
PY
)"
mapfile -t source_threshold_entries < <("${python_bin}" - "${calibration}" <<'PY'
import json,sys
mapping=json.load(open(sys.argv[1],encoding="utf-8")).get("source_classification_thresholds",{})
expected={"epoch105","currentgf","autopet_weak","lndb_weak"} | {f"lung_cell{i}" for i in range(8)}
if mapping and set(mapping)!=expected:
    raise SystemExit(f"invalid frozen source thresholds: {sorted(mapping)}")
for name,value in sorted(mapping.items()):
    value=float(value)
    if not 0 < value < 1: raise SystemExit(f"invalid threshold: {name}={value}")
    print(f"{name}={value:.2f}")
PY
)
source_args=(); source_tag=""; fusion_tag=""
[[ -z "${classifier_fusion}" ]] || fusion_tag="_ff"
for entry in "${source_threshold_entries[@]}"; do
  source_args+=(--source-classification-threshold "${entry}")
done
[[ ${#source_threshold_entries[@]} -eq 0 ]] || source_tag="_src"
mapfile -t source_segmentation_entries < <("${python_bin}" - "${calibration}" <<'PY'
import json,sys
mapping=json.load(open(sys.argv[1],encoding="utf-8")).get("source_segmentation_thresholds",{})
expected={"epoch105","currentgf","autopet_weak","lndb_weak"} | {f"lung_cell{i}" for i in range(8)}
if mapping and set(mapping)!=expected: raise SystemExit(f"invalid frozen source contour thresholds: {sorted(mapping)}")
for name,value in sorted(mapping.items()):
    value=float(value)
    if not 0 < value < 1: raise SystemExit(f"invalid contour threshold: {name}={value}")
    print(f"{name}={value:.2f}")
PY
)
source_segmentation_args=(); source_segmentation_tag=""
for entry in "${source_segmentation_entries[@]}"; do
  source_segmentation_args+=(--source-segmentation-threshold "${entry}")
done
[[ ${#source_segmentation_entries[@]} -eq 0 ]] || source_segmentation_tag="_sg"
mapfile -t source_mode_entries < <("${python_bin}" - "${calibration}" <<'PY'
import json,sys
mapping=json.load(open(sys.argv[1],encoding="utf-8")).get("source_deployment_modes",{})
expected={"epoch105","currentgf","autopet_weak","lndb_weak"} | {f"lung_cell{i}" for i in range(8)}
allowed={"refined","refined_or_seed","seed","seed_crop","seed_union","prune_base"}
if set(mapping)-expected: raise SystemExit(f"invalid frozen source modes: {sorted(mapping)}")
for name,mode in sorted(mapping.items()):
    if mode not in allowed: raise SystemExit(f"invalid source mode: {name}={mode}")
    print(f"{name}={mode}")
PY
)
source_mode_args=(); source_mode_tag=""
has_prune_base=0
for entry in "${source_mode_entries[@]}"; do
  source_mode_args+=(--source-mode "${entry}")
  [[ "${entry}" == *=prune_base ]] && has_prune_base=1
done
[[ ${#source_mode_entries[@]} -eq 0 ]] || source_mode_tag="_sm"
if [[ "${calibration_deployment}" == anchor_currentgf ]]; then
  deployments=(anchor)
else
  deployments=(standalone)
fi
if [[ "${has_prune_base}" -eq 1 ]]; then
  [[ "${allow_prune_base}" -eq 1 ]] || {
    echo "prune_base calibration requires explicit WEAK707_PUBLIC_ALLOW_PRUNE_BASE=1" >&2; exit 2;
  }
  deployments=(anchor)
fi
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tdeployment\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' > "${output}/summary.tsv"
for deployment in "${deployments[@]}"; do
  tag="${candidate_prefix}_${deployment}_${deployment_mode}${fusion_tag}${source_tag}${source_segmentation_tag}${source_mode_tag}_c${classification/./p}_s${segmentation/./p}"
  prediction="${output}/${tag}"; empty_arg=()
  [[ "${deployment}" == standalone ]] && empty_arg+=(--empty-base)
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
    --work "${work}" --base "${base}" --output "${prediction}" --resume "${empty_arg[@]}" \
    --classification-threshold "${classification}" \
    "${source_args[@]}" \
    "${source_segmentation_args[@]}" \
    "${source_mode_args[@]}" \
    --segmentation-threshold "${segmentation}" --mode "${deployment_mode}" \
    --min-seed-coverage "${min_seed_coverage}" \
    --min-refined-overlap "${min_refined_overlap}" \
    --min-volume-ratio "${min_volume_ratio}" \
    --max-volume-ratio "${max_volume_ratio}" > "${prediction}.materialize.log" 2>&1
  metrics="${prediction}.metrics.csv"; score_log="${prediction}.score.log"
  robust="${prediction}.robust.json"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${prediction}" \
    --gt "${labels}" --csv "${metrics}" --workers 4 > "${score_log}" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor_metrics}" --candidate "${metrics}" \
    --anchor-name gf_champ_new2k_s25 --candidate-name "${tag}" \
    --output "${robust}" > "${prediction}.robust.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
  nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  gate=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${classification}" "${segmentation}" "${deployment_mode}" \
    "${deployment}" "${dsc}" "${nsd}" "${combined}" "${metrics}" \
    "${robust}" "${gate}" >> "${output}/summary.tsv"
done
{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k6,6gr -k7,7gr; } \
  > "${output}/summary_by_dsc.tsv"
date -Ins > "${complete}"
cat "${output}/summary_by_dsc.tsv"
