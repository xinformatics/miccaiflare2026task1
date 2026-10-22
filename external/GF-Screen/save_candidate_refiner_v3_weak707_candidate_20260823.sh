#!/usr/bin/env bash
set -euo pipefail

# Save-only hidden consumer for a strictly promoted exact-707 refiner. Hidden
# labels are unavailable and never referenced. The script validates and hashes
# an exact flat 100-mask ZIP and never uploads it.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public="${WEAK707_SAVE_PUBLIC_ROOT:-${root}/public_sweeps/candidate_refiner_v4_cell707_20260823}"
train="${WEAK707_SAVE_TRAIN_ROOT:-${root}/analysis/candidate_refiner_v4_cell707_training_20260823}"
selection="${WEAK707_SAVE_SELECTION:-${train}/checkpoint_selection.json}"
classifier_checkpoint_override="${WEAK707_SAVE_CLASSIFIER_CHECKPOINT_NAME:-}"
calibration="${WEAK707_SAVE_CALIBRATION:-${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823/deployment_calibration.json}"
contour_summary="${WEAK707_SAVE_CONTOUR_SUMMARY:-}"
classifier_fusion="${WEAK707_SAVE_CLASSIFIER_FUSION:-}"
classifier_route_decision="${WEAK707_SAVE_CLASSIFIER_ROUTE_DECISION:-}"
focus_route_summary="${WEAK707_SAVE_FOCUS_ROUTE_SUMMARY:-}"
focus_route_decision="${WEAK707_SAVE_FOCUS_ROUTE_DECISION:-}"
mixed_focus_summary="${WEAK707_SAVE_MIXED_FOCUS_SUMMARY:-}"
mixed_route_decision="${WEAK707_SAVE_MIXED_ROUTE_DECISION:-}"
composite_route_decision="${WEAK707_SAVE_COMPOSITE_ROUTE_DECISION:-}"
margin_decision="${root}/analysis/weak707_bbox_margin_headroom_20260823/DECISION.json"
plan="${biomed}/analysis/weak_domain_biomedparse_707_selected_20260823/SELECTION_PLAN.json"
cell_scale="${root}/analysis/lung_cell707_scale_20260823"
cell_decision="${cell_scale}/DECISION.json"
saved_root="${WEAK707_SAVE_OUTPUT:-${root}/public_sweeps/candidate_refiner_v4_cell707_saved_20260823}"
hidden_images="${project}/validation/Validation-Hidden-Images"
hidden_inputs="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_biomedparse_direct_whole_body_lesion_e030_confirm_distance_mm_0_rescue_min_mm3_inf/inputs"
hidden_root="${WEAK707_SAVE_HIDDEN_ROOT:-${root}/hidden_predictions/candidate_refiner_v4_cell707_20260823}"
proposal_cache="${WEAK707_SAVE_PROPOSAL_CACHE_ROOT:-${hidden_root}}"
epoch105_root="${root}/hidden_predictions/candidate_refiner_v2_bbox24_20260823/epoch105_abd_g090_l050"
epoch105="${epoch105_root}/predictions"
currentgf="${root}/predictions/gf_a055_bbox_pu_best_ep280/hidden_raw"
hidden_base="${WEAK707_SAVE_BASE:-${root}/hidden_predictions/candidate_refiner_v2_bbox24_20260823/champion_base}"
hidden_work="${hidden_root}/work_bbox24_ss_v4_cell48"
hidden_manifest="${proposal_cache}/hidden_manifest.json"
hidden_cells="${proposal_cache}/cells48"
champion_zip="${project}/submissions/gf_champ_new2k_s25.zip"
gf_checkpoint="${root}/runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt"
packager="${project}/nnunet/package_codabench.py"
preflight="${saved_root}/SUBMISSION_CHAIN_PREFLIGHT.json"
ready="${project}/submissions/READY_20260822.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"
complete="${saved_root}/SAVE_ONLY_COMPLETE.txt"
tag_prefix="${WEAK707_SAVE_TAG_PREFIX:-gf_refc707}"
run_prefix="${WEAK707_SAVE_RUN_PREFIX:-candidate_refiner_v4_cell707_b32_t05_fold}"
discovery_run_prefix="${WEAK707_SAVE_DISCOVERY_RUN_PREFIX:-${run_prefix}}"
discovery_run_suffix="${WEAK707_SAVE_DISCOVERY_RUN_SUFFIX:-20260823}"
family="${WEAK707_SAVE_FAMILY:-candidate_refiner_v4_cell707}"
allow_anchor_fallback="${WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK:-1}"
allow_prune_base="${WEAK707_SAVE_ALLOW_PRUNE_BASE:-0}"
source_routed="${WEAK707_SAVE_SOURCE_ROUTED:-0}"
controller_lock="${WEAK707_SAVE_LOCK:-${root}/runs/save_candidate_refiner_v3_weak707_candidate_20260823.lock}"

[[ "${allow_anchor_fallback}" == 0 || "${allow_anchor_fallback}" == 1 ]] || {
  echo "WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK must be 0 or 1" >&2; exit 2;
}
[[ "${allow_prune_base}" == 0 || "${allow_prune_base}" == 1 ]] || {
  echo "WEAK707_SAVE_ALLOW_PRUNE_BASE must be 0 or 1" >&2; exit 2;
}
[[ "${source_routed}" == 0 || "${source_routed}" == 1 ]] || {
  echo "WEAK707_SAVE_SOURCE_ROUTED must be 0 or 1" >&2; exit 2;
}
[[ -z "${focus_route_summary}" || -n "${focus_route_decision}" ]] || {
  echo "routed Focus summary requires WEAK707_SAVE_FOCUS_ROUTE_DECISION" >&2; exit 2;
}
[[ -z "${focus_route_decision}" || -n "${focus_route_summary}" ]] || {
  echo "routed Focus decision requires WEAK707_SAVE_FOCUS_ROUTE_SUMMARY" >&2; exit 2;
}
[[ -z "${mixed_focus_summary}" || -n "${mixed_route_decision}" ]] || {
  echo "mixed Focus summary requires WEAK707_SAVE_MIXED_ROUTE_DECISION" >&2; exit 2;
}
[[ -z "${mixed_route_decision}" || -n "${mixed_focus_summary}" ]] || {
  echo "mixed route decision requires WEAK707_SAVE_MIXED_FOCUS_SUMMARY" >&2; exit 2;
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
    && -z "${classifier_checkpoint_override}" && -z "${composite_route_decision}" ) ]] || {
  echo "mixed B32/Focus routing is exclusive with other contour/classifier overrides" >&2; exit 2;
}
[[ -z "${composite_route_decision}" || ( -z "${contour_summary}" \
    && -z "${focus_route_summary}" && -z "${mixed_route_decision}" \
    && "${source_routed}" -eq 0 && -z "${classifier_fusion}" \
    && -z "${classifier_checkpoint_override}" && -z "${classifier_route_decision}" ) ]] || {
  echo "composite B32/Focus routing is exclusive with other contour/classifier overrides" >&2; exit 2;
}
fold_dir() { printf '%s/runs/%s%s_20260823' "${root}" "${run_prefix}" "$1"; }
discovery_fold_dir() { printf '%s/runs/%s%s_%s' "${root}" "${discovery_run_prefix}" "$1" "${discovery_run_suffix}"; }

mkdir -p "${saved_root}" "${hidden_root}" "${proposal_cache}" "${epoch105}" "${hidden_base}"
exec 9>"${controller_lock}"
flock -n 9 || { echo "weak707 save-only consumer already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
while [[ ! -s "${public}/PUBLIC_ONLY_COMPLETE.txt" ]]; do
  printf '%s waiting for weak707 public qualification\n' "$(date -Ins)"
  sleep 30
done
if [[ ! -s "${public}/summary.tsv" ]]; then
  date -Ins > "${saved_root}/NO_ELIGIBLE_WEAK707_CANDIDATE.txt"
  date -Ins > "${complete}"
  exit 0
fi
for required in "${selection}" "${calibration}" "${plan}" "${champion_zip}" \
  "${gf_checkpoint}" "${margin_decision}" "${cell_decision}"; do
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
[[ -z "${composite_route_decision}" || -s "${composite_route_decision}" ]] || { echo "missing composite route decision: ${composite_route_decision}" >&2; exit 2; }
[[ $(find "${hidden_images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 100 ]]
[[ $(find "${hidden_inputs}" -maxdepth 1 -name '*.npz' | wc -l) -eq 100 ]]
[[ $(find "${currentgf}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

cell_gate=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${cell_decision}")
[[ "${cell_gate}" == PROMOTE_LUNG_CELLS_707 ]] || {
  echo "exact-707 lung-cell gate did not promote: ${cell_gate}" >&2
  exit 2
}
"${python_bin}" - "${hidden_images}" "${hidden_manifest}" <<'PY'
import json,sys
from pathlib import Path
images,output=map(Path,sys.argv[1:]); items=[]
for image in sorted(images.glob("*_0000.nii.gz")):
    case=image.name[:-12]
    items.append({"name":case,"prediction_name":case+".nii.gz",
                  "image":str(image.resolve()),"complete_label":False})
if len(items)!=100 or len({item["name"] for item in items})!=100:
    raise SystemExit("expected exact 100 hidden images")
output.write_text(json.dumps({"description":"label-blind hidden cell manifest",
                              "items":items},indent=2)+"\n",encoding="utf-8")
PY
if [[ ! -s "${hidden_cells}/PROPOSALS_COMPLETE.json" ]]; then
  "${python_bin}" -u "${root}/generate_lung_cell_proposals.py" \
    --manifest "${hidden_manifest}" --all-cases --output "${hidden_cells}" \
    --target-spacing-mm 2.5 --cell-stride-mm 48 --expected-count 100 --workers 2 --resume \
    > "${saved_root}/generate_cells.log" 2>&1
fi
preflight_extra_args=()
[[ -z "${classifier_route_decision}" ]] || \
  preflight_extra_args+=(--extra-asset "classifier_route_decision=${classifier_route_decision}")
[[ -z "${composite_route_decision}" ]] || \
  preflight_extra_args+=(--extra-asset "composite_route_decision=${composite_route_decision}")
"${python_bin}" -u "${root}/preflight_candidate_refiner_submission_chain.py" \
  --public-images "${project}/validation/Validation-Public-Images" \
  --public-labels "${project}/validation/Validation-Public-Labels" \
  --public-inputs "${biomed}/public_a160_240_inputs" \
  --public-epoch105 "${root}/public_sweeps/epoch0105/abdomen_g09_l05/predictions" \
  --public-currentgf "${root}/predictions/gf_a055_bbox_pu_best_ep280/public_raw" \
  --public-base "${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25" \
  --hidden-images "${hidden_images}" --hidden-inputs "${hidden_inputs}" \
  --hidden-currentgf "${currentgf}" --champion-zip "${champion_zip}" \
  --public-cells "${public}/cells48/PROPOSALS_COMPLETE.json" \
  --hidden-cells "${hidden_cells}/PROPOSALS_COMPLETE.json" \
  --selection "${selection}" --calibration "${calibration}" --weak-plan "${plan}" \
  --margin-decision "${margin_decision}" --cell-decision "${cell_decision}" \
  --gf-checkpoint "${gf_checkpoint}" --packager "${packager}" \
  --public-summary "${public}/summary.tsv" "${preflight_extra_args[@]}" \
  --output "${preflight}" \
  > "${saved_root}/submission_chain_preflight.log" 2>&1

selection_args=(--summary "${public}/summary.tsv" --expected-cases 50)
[[ "${allow_anchor_fallback}" == 0 ]] || selection_args+=(--allow-anchor-fallback)
selected=$("${python_bin}" "${root}/select_promoted_refiner_summary.py" "${selection_args[@]}")
if [[ -z "${selected}" ]]; then
  date -Ins > "${saved_root}/NO_ELIGIBLE_WEAK707_CANDIDATE.txt"
  date -Ins > "${complete}"
  exit 0
fi
IFS=$'\t' read -r variant classification segmentation deployment_mode deployment dsc nsd combined metrics robust decision <<< "${selected}"
expected_deployment=$("${python_bin}" - "${calibration}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8")).get("calibration_deployment")
mapping={"standalone":"standalone","anchor_currentgf":"anchor"}
if value not in mapping: raise SystemExit(f"missing or invalid frozen calibration deployment: {value}")
print(mapping[value])
PY
)
[[ "${deployment}" == "${expected_deployment}" ]] || {
  echo "public selection changed frozen deployment: expected ${expected_deployment}, got ${deployment}" >&2
  exit 2
}

folds=(); contour_folds=(); contour_args=(); contour_input_mode=""; focus_routed_input_mode=""; infer_batch=8
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
    folds+=("${classification}")
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
    folds+=("${checkpoint}")
    if [[ "${selection_contour_name}" != NONE ]]; then
      contour="$(fold_dir "${fold}")/${selection_contour_name}"
      [[ -s "${contour}" ]] || { echo "missing contour checkpoint: ${contour}" >&2; exit 2; }
      contour_folds+=("${contour}"); contour_args+=(--contour-checkpoint "${contour}")
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
  for checkpoint in "${folds[@]}"; do
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
  for checkpoint in "${folds[@]}"; do
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
  mapfile -t contour_folds < <("${python_bin}" - "${contour_summary}" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))["fold_records"]
if sorted(int(row["fold"]) for row in rows)!=list(range(5)): raise SystemExit("invalid contour folds")
for row in sorted(rows,key=lambda value:int(value["fold"])): print(row["checkpoint"])
PY
)
  [[ "${#contour_folds[@]}" -eq 5 ]] || exit 2
  contour_input_mode=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["input_mode"])' "${contour_summary}")
  for contour in "${contour_folds[@]}"; do
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
  # Match the proven primary B32 inference capacity. Route ensembles execute
  # sequentially, so only checkpoint weights—not activation graphs—accumulate.
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

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
wait_any_gpu() {
  while :; do
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    free0=0; free1=0
    [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 1024 ]] && free0=1
    [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 1024 ]] && free1=1
    [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && return
    printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
}

proposal_marker="${epoch105_root}/PROPOSALS_COMPLETE.json"
if [[ ! -s "${proposal_marker}" ]]; then
  wait_any_gpu
  mkdir -p "${epoch105_root}/gpu0" "${epoch105_root}/gpu1"
  run_epoch105_shard() {
    local gpu="$1" shard="$2"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${hidden_images}" --save_prediction_path "${epoch105_root}/gpu${shard}" \
      --trained_pth "${gf_checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.90 --lesion_threshold 0.50 --focus_batch_size 16 \
      --workers 2 --num_shards 2 --shard_id "${shard}" --skip_existing \
      > "${saved_root}/epoch105_gpu${shard}.log" 2>&1
  }
  if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
    run_epoch105_shard 0 0 & p0=$!; run_epoch105_shard 1 1 & p1=$!
    status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?
    [[ "${status}" -eq 0 ]] || exit "${status}"
  elif [[ "${free0}" -eq 1 ]]; then
    run_epoch105_shard 0 0; run_epoch105_shard 0 1
  else
    run_epoch105_shard 1 0; run_epoch105_shard 1 1
  fi
  cp "${epoch105_root}/gpu0/"*.nii.gz "${epoch105}/"
  cp "${epoch105_root}/gpu1/"*.nii.gz "${epoch105}/"
  checkpoint_sha=$(sha256sum "${gf_checkpoint}" | awk '{print $1}')
  "${python_bin}" - "${hidden_images}" "${epoch105}" "${proposal_marker}" \
    "${gf_checkpoint}" "${checkpoint_sha}" <<'PY'
import datetime,json,sys
from pathlib import Path
images,predictions,marker,checkpoint=map(Path,sys.argv[1:5]); digest=sys.argv[5]
expected={path.name[:-12]+".nii.gz" for path in images.glob("*_0000.nii.gz")}
observed={path.name for path in predictions.glob("*.nii.gz")}
if len(expected)!=100 or observed!=expected: raise SystemExit("hidden epoch105 coverage mismatch")
marker.write_text(json.dumps({
 "created_at":datetime.datetime.now().astimezone().isoformat(),"cases":100,
 "labels_read":False,"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":digest,
 "window_hu":[-175,250],"glance_threshold":0.90,"lesion_threshold":0.50,
},indent=2)+"\n",encoding="utf-8")
PY
fi
[[ $(find "${epoch105}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

run_bank() {
  local bank="$1" gpu="$2"
  bash "${biomed}/run_selected_weak_prompt_bank.sh" \
    "${hidden_inputs}" "${hidden_images}" "${proposal_cache}/${bank}_weak" \
    "${plan}" "${bank}" 100 "${gpu}" > "${saved_root}/${bank}_weak.controller.log" 2>&1
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

if [[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
  unzip -nq "${champion_zip}" -d "${hidden_base}"
fi
[[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
if [[ ! -s "${hidden_work}/PREPARE_COMPLETE.txt" ]]; then
  cell_hidden_args=()
  for colour in 0 1 2 3 4 5 6 7; do
    cell_hidden_args+=(--proposal "lung_cell${colour}=${hidden_cells}/colour${colour}")
  done
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${hidden_images}" --proposal "epoch105=${epoch105}" \
    --proposal "currentgf=${currentgf}" \
    --proposal "autopet_weak=${autopet}" --proposal "lndb_weak=${lndb}" \
    "${cell_hidden_args[@]}" \
    --component-policy source_separate --base "${hidden_base}" --work "${hidden_work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 \
    --max-mm3 1000000000000 --bbox-margin-mm "${bbox_margin}" --tile-stride-mm 72 --resume \
    > "${saved_root}/prepare_hidden.log" 2>&1
fi
checkpoint_args=(); for checkpoint in "${folds[@]}"; do checkpoint_args+=(--checkpoint "${checkpoint}"); done
inference_matches() {
  local marker_args=("${checkpoint_args[@]}" "${contour_args[@]}" "${routed_args[@]}")
  [[ -z "${contour_input_mode}" ]] || marker_args+=(--contour-input-mode "${contour_input_mode}")
  [[ -z "${focus_routed_input_mode}" ]] || marker_args+=(--focus-routed-input-mode "${focus_routed_input_mode}")
  [[ -z "${classifier_fusion}" ]] || marker_args+=(--classifier-fusion "${classifier_fusion}")
  "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    --marker "${hidden_work}/INFERENCE_COMPLETE.json" "${marker_args[@]}" \
    > /dev/null 2>&1
}
if ! inference_matches; then
  wait_any_gpu
  if [[ "${free0}" -eq 1 ]]; then inference_gpu=0; else inference_gpu=1; fi
  CUDA_VISIBLE_DEVICES="${inference_gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${hidden_work}" "${checkpoint_args[@]}" "${contour_args[@]}" "${routed_args[@]}" \
    "${fusion_args[@]}" --batch-size "${infer_batch}" --workers 2 --resume \
    > "${saved_root}/infer_hidden.log" 2>&1
fi
inference_matches
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${hidden_work}" --output "${hidden_work}/AUDIT.json" --workers 4 \
  --require-predictions --require-inference-plan \
  > "${saved_root}/audit_hidden.log" 2>&1

case "${deployment_mode}" in
  refined) mode_tag=r ;;
  refined_or_seed) mode_tag=q ;;
  seed) mode_tag=s ;;
  seed_union) mode_tag=u ;;
  *) echo "invalid deployment mode: ${deployment_mode}" >&2; exit 2 ;;
esac
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
if [[ "${has_prune_base}" -eq 1 ]]; then
  [[ "${allow_prune_base}" -eq 1 ]] || {
    echo "prune_base calibration requires explicit WEAK707_SAVE_ALLOW_PRUNE_BASE=1" >&2; exit 2;
  }
  [[ "${deployment}" == anchor ]] || {
    echo "prune_base cannot be materialized with an empty/standalone base" >&2; exit 2;
  }
fi
tag="${tag_prefix}_${deployment:0:3}_${mode_tag}${fusion_tag}${source_tag}${source_segmentation_tag}${source_mode_tag}_c${classification/./p}_s${segmentation/./p}"
[[ ${#tag} -lt 60 ]] || { echo "submission tag is too long: ${tag}" >&2; exit 2; }
prediction="${hidden_root}/${tag}"; empty_arg=()
[[ "${deployment}" == standalone ]] && empty_arg+=(--empty-base)
"${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
  --work "${hidden_work}" --base "${hidden_base}" --output "${prediction}" "${empty_arg[@]}" \
  --classification-threshold "${classification}" --segmentation-threshold "${segmentation}" \
  "${source_args[@]}" \
  "${source_segmentation_args[@]}" \
  "${source_mode_args[@]}" \
  --mode "${deployment_mode}" --min-seed-coverage "${min_seed_coverage}" \
  --min-refined-overlap "${min_refined_overlap}" \
  --min-volume-ratio "${min_volume_ratio}" --max-volume-ratio "${max_volume_ratio}" \
  > "${saved_root}/materialize.log" 2>&1
[[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

submission="${project}/submissions/${tag}.zip"
"${python_bin}" -u "${packager}" --images "${hidden_images}" --pred-dir "${prediction}" \
  --output "${submission}" --expected-count 100 > "${saved_root}/package.log" 2>&1
unzip -tq "${submission}"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${submission}" > "${submission}.sha256"

saved="${saved_root}/saved_weak707_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' > "${saved}"
checkpoints=$(IFS=,; echo "${folds[*]}")
[[ -z "${composite_route_decision}" ]] || checkpoints="${checkpoints},route=${composite_route_decision}"
printf '1\t%s\t%s\t%s\t0.90+weak+cell48\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${family}" "${checkpoints}" "${segmentation}" "${dsc}" "${nsd}" "${metrics}" \
  "${robust}" "${decision}" "${submission}" "${digest}" >> "${saved}"

"${python_bin}" "${root}/refresh_all_saved_candidates_20260822.py" \
  --manifest-root "${root}/public_sweeps" \
  --manifest "${saved}" --submitted "${submitted}" --output "${ready}" \
  --audit-output "${project}/submissions/READY_20260822.audit.json" > "${saved_root}/refresh_ready.log"
"${python_bin}" "${root}/rank_submission_candidates_calibrated.py" \
  --history "${root}/codabench_history.tsv" --candidates "${ready}" \
  --output "${project}/submissions/READY_20260822.calibrated.tsv" \
  --json-output "${project}/submissions/READY_20260822.calibrated.json" \
  > "${saved_root}/calibrated_rank.log"
date -Ins > "${complete}"
